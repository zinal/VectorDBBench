"""Orchestrate YDB vector_kmeans_tree parameter tuning for VectorDBBench."""

from __future__ import annotations

import csv
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import ujson
import yaml

from vectordb_bench import config as bench_config
from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.models import ResultLabel

log = logging.getLogger(__name__)

DEFAULT_SEARCH_TOP_SIZE_VALUES = [1, 2, 5, 10, 20, 32, 64]
DEFAULT_BASELINE_LABELS = ["standard_20260403"]
CASE_NAME_BY_ID = {member.value: member.name for member in CaseType}


@dataclass
class TuneRunRecord:
    phase: str
    case_type: str
    db_label: str
    kmeans_tree_search_top_size: int
    levels: int | None
    clusters: int | None
    overlap_clusters: int | None
    recall: float | None = None
    ndcg: float | None = None
    qps: float | None = None
    serial_latency_p99: float | None = None
    load_duration: float | None = None
    optimize_duration: float | None = None
    result_file: str = ""
    elapsed_seconds: float = 0.0


@dataclass
class TuneConfig:
    case_type: str = "Performance768D1M"
    task_label: str = "ydb_tune"
    db_label: str = "ydb-tune"
    endpoint: str = ""
    database: str = ""
    target_recall: float = 0.92
    min_recall: float = 0.88
    search_top_size_values: list[int] = field(default_factory=lambda: list(DEFAULT_SEARCH_TOP_SIZE_VALUES))
    index_build_grid: list[dict[str, Any]] = field(default_factory=lambda: [{}])
    cover_embedding: bool = True
    finalize_top_n: int = 2
    baseline_task_labels: list[str] = field(default_factory=lambda: list(DEFAULT_BASELINE_LABELS))
    output_dir: str = "./ydb_tune_results"
    extra_cli_args: list[str] = field(default_factory=list)
    skip_load: bool = False
    skip_drop_old: bool = False
    table_name: str = ""


def _load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = f"Config file must contain a YAML mapping: {path}"
        raise click.BadParameter(msg)
    return data


def _resolve_config_file(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    fallback = Path(bench_config.CONFIG_LOCAL_DIR, path)
    if fallback.exists():
        return fallback
    msg = f"Config file not found: {path}"
    raise click.BadParameter(msg)


def _table_name_for_build(tune: TuneConfig, build_idx: int) -> str:
    if tune.table_name:
        return tune.table_name
    base = CaseType[tune.case_type].name.lower()
    return f"{base}_b{build_idx}"


def _case_id(case_type: str) -> int:
    try:
        return CaseType[case_type].value
    except KeyError as exc:
        valid = ", ".join(sorted(CASE_NAME_BY_ID.values()))
        msg = f"Unknown case_type={case_type!r}. Valid values: {valid}"
        raise click.BadParameter(msg) from exc


def _build_cli_args(
    tune: TuneConfig,
    *,
    db_label: str,
    top_size: int,
    build: dict[str, Any],
    load: bool,
    drop_old: bool,
    search_serial: bool,
    search_concurrent: bool,
    table_name: str = "",
) -> list[str]:
    args = [
        "ydb",
        "--case-type",
        tune.case_type,
        "--db-label",
        db_label,
        "--task-label",
        tune.task_label,
        "--kmeans-tree-search-top-size",
        str(top_size),
    ]
    if tune.endpoint:
        args.extend(["--endpoint", tune.endpoint])
    if tune.database:
        args.extend(["--database", tune.database])
    effective_table_name = table_name or tune.table_name
    if effective_table_name:
        args.extend(["--table-name", effective_table_name])

    levels = build.get("levels")
    clusters = build.get("clusters")
    overlap = build.get("overlap_clusters")
    if levels is not None:
        args.extend(["--levels", str(levels)])
    if clusters is not None:
        args.extend(["--clusters", str(clusters)])
    if overlap is not None:
        args.extend(["--overlap-clusters", str(overlap)])

    cover = build.get("cover_embedding", tune.cover_embedding)
    if cover:
        args.append("--cover-embedding")
    else:
        args.append("--no-cover-embedding")

    if load:
        args.append("--load")
    else:
        args.append("--skip-load")

    if drop_old:
        args.append("--drop-old")
    else:
        args.append("--skip-drop-old")

    if search_serial:
        args.append("--search-serial")
    else:
        args.append("--skip-search-serial")

    if search_concurrent:
        args.append("--search-concurrent")
    else:
        args.append("--skip-search-concurrent")

    args.extend(tune.extra_cli_args)
    return args


def _run_vectordbbench(cli_args: list[str]) -> None:
    cmd = [sys.executable, "-m", "vectordb_bench.cli.vectordbbench", *cli_args]
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _latest_result_mtime() -> float:
    result_dir = Path(bench_config.RESULTS_LOCAL_DIR) / DB.YDB.value
    if not result_dir.exists():
        return 0.0
    files = list(result_dir.glob("result_*.json"))
    if not files:
        return 0.0
    return max(path.stat().st_mtime for path in files)


def _wait_for_new_result(since_mtime: float, timeout: float = 7200.0) -> Path | None:
    result_dir = Path(bench_config.RESULTS_LOCAL_DIR) / DB.YDB.value
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if result_dir.exists():
            candidates = [path for path in result_dir.glob("result_*.json") if path.stat().st_mtime > since_mtime]
            if candidates:
                return max(candidates, key=lambda path: path.stat().st_mtime)
        time.sleep(2.0)
    return None


def _extract_case_entry(result_file: Path, case_type: str, db_label: str) -> dict[str, Any] | None:
    case_id = _case_id(case_type)
    payload = ujson.loads(result_file.read_text())
    for entry in payload.get("results", []):
        task_config = entry.get("task_config", {})
        case_config = task_config.get("case_config", {})
        if case_config.get("case_id") != case_id:
            continue
        db_config = task_config.get("db_config", {})
        if db_config.get("db_label") != db_label:
            continue
        return entry
    return None


def _extract_case_metrics(result_file: Path, case_type: str, db_label: str) -> dict[str, Any] | None:
    entry = _extract_case_entry(result_file, case_type, db_label)
    if entry is None:
        return None
    return entry.get("metrics", {})


def _ensure_benchmark_succeeded(
    result_file: Path | None,
    *,
    case_type: str,
    db_label: str,
    phase: str,
) -> None:
    if result_file is None:
        msg = f"No result file produced for {db_label} ({phase})"
        raise RuntimeError(msg)

    entry = _extract_case_entry(result_file, case_type, db_label)
    if entry is None:
        msg = f"No matching result entry for {db_label} in {result_file}"
        raise RuntimeError(msg)

    label = entry.get("label")
    if label != ResultLabel.NORMAL.value:
        metrics = entry.get("metrics", {})
        msg = f"Benchmark {db_label} ({phase}) failed with label={label!r}, metrics={metrics}"
        raise RuntimeError(msg)

    if phase == "load":
        metrics = entry.get("metrics", {})
        load_duration = metrics.get("load_duration") or 0.0
        if load_duration <= 0:
            msg = f"Load phase for {db_label} did not complete successfully (load_duration={load_duration})"
            raise RuntimeError(msg)


def _run_benchmark(
    tune: TuneConfig,
    *,
    phase: str,
    db_label: str,
    top_size: int,
    build: dict[str, Any],
    load: bool,
    drop_old: bool,
    search_serial: bool,
    search_concurrent: bool,
    table_name: str = "",
) -> TuneRunRecord:
    since = _latest_result_mtime()
    started = time.monotonic()
    cli_args = _build_cli_args(
        tune,
        db_label=db_label,
        top_size=top_size,
        build=build,
        load=load,
        drop_old=drop_old,
        search_serial=search_serial,
        search_concurrent=search_concurrent,
        table_name=table_name,
    )
    _run_vectordbbench(cli_args)
    result_file = _wait_for_new_result(since)
    _ensure_benchmark_succeeded(
        result_file,
        case_type=tune.case_type,
        db_label=db_label,
        phase=phase,
    )
    elapsed = time.monotonic() - started

    record = TuneRunRecord(
        phase=phase,
        case_type=tune.case_type,
        db_label=db_label,
        kmeans_tree_search_top_size=top_size,
        levels=build.get("levels"),
        clusters=build.get("clusters"),
        overlap_clusters=build.get("overlap_clusters"),
        elapsed_seconds=round(elapsed, 2),
    )
    if result_file:
        metrics = _extract_case_metrics(result_file, tune.case_type, db_label)
        record.result_file = str(result_file)
        if metrics:
            record.recall = metrics.get("recall")
            record.ndcg = metrics.get("ndcg")
            record.qps = metrics.get("qps")
            record.serial_latency_p99 = metrics.get("serial_latency_p99")
            record.load_duration = metrics.get("load_duration")
            record.optimize_duration = metrics.get("optimize_duration")
    return record


def _serial_sweep(
    tune: TuneConfig,
    build: dict[str, Any],
    *,
    load_first: bool,
    drop_old_first: bool,
    values: list[int],
    table_name: str,
) -> list[TuneRunRecord]:
    records: list[TuneRunRecord] = []
    for idx, top_size in enumerate(values):
        record = _run_benchmark(
            tune,
            phase="search_sweep",
            db_label=f"{tune.db_label}-top{top_size}",
            top_size=top_size,
            build=build,
            load=load_first and idx == 0,
            drop_old=drop_old_first and idx == 0,
            search_serial=True,
            search_concurrent=False,
            table_name=table_name,
        )
        records.append(record)
        log.info(
            "Sweep top_size=%s recall=%s qps=%s",
            top_size,
            record.recall,
            record.qps,
        )
    return records


def _binary_search_top_size(
    tune: TuneConfig,
    build: dict[str, Any],
    bracket: tuple[int, int],
    target: float,
    *,
    table_name: str,
) -> list[TuneRunRecord]:
    records: list[TuneRunRecord] = []
    lo, hi = bracket
    tested: dict[int, float | None] = {}

    def recall_for(top_size: int) -> float | None:
        if top_size in tested:
            return tested[top_size]
        record = _run_benchmark(
            tune,
            phase="binary_search",
            db_label=f"{tune.db_label}-bin{top_size}",
            top_size=top_size,
            build=build,
            load=False,
            drop_old=False,
            search_serial=True,
            search_concurrent=False,
            table_name=table_name,
        )
        records.append(record)
        tested[top_size] = record.recall
        return record.recall

    recall_for(lo)
    recall_for(hi)

    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mid in tested:
            if tested[mid] is not None and tested[mid] >= target:
                hi = mid
            else:
                lo = mid
            continue
        mid_recall = recall_for(mid)
        if mid_recall is not None and mid_recall >= target:
            hi = mid
        else:
            lo = mid

    return records


def _pareto_score(record: TuneRunRecord) -> float:
    recall = record.recall or 0.0
    qps = record.qps or 0.0
    return qps * recall


def _select_finalists(records: list[TuneRunRecord], tune: TuneConfig) -> list[TuneRunRecord]:
    serial_runs = [record for record in records if record.phase in {"search_sweep", "binary_search"} and record.recall]
    if not serial_runs:
        return []

    at_target = [record for record in serial_runs if record.recall and record.recall >= tune.target_recall]
    pool = at_target or [max(serial_runs, key=lambda record: record.recall or 0.0)]

    by_top_size: dict[int, TuneRunRecord] = {}
    for record in pool:
        existing = by_top_size.get(record.kmeans_tree_search_top_size)
        if existing is None or (record.recall or 0) > (existing.recall or 0):
            by_top_size[record.kmeans_tree_search_top_size] = record

    finalists = sorted(by_top_size.values(), key=_pareto_score, reverse=True)
    return finalists[: max(tune.finalize_top_n, 1)]


def _finalize_runs(
    tune: TuneConfig,
    build: dict[str, Any],
    finalists: list[TuneRunRecord],
    *,
    table_name: str,
) -> list[TuneRunRecord]:
    records: list[TuneRunRecord] = []
    for finalist in finalists:
        record = _run_benchmark(
            tune,
            phase="finalize",
            db_label=f"{tune.db_label}-final-top{finalist.kmeans_tree_search_top_size}",
            top_size=finalist.kmeans_tree_search_top_size,
            build=build,
            load=False,
            drop_old=False,
            search_serial=True,
            search_concurrent=True,
            table_name=table_name,
        )
        records.append(record)
    return records


def _load_baselines(case_type: str, task_labels: list[str]) -> list[dict[str, Any]]:
    case_id = _case_id(case_type)
    baselines: list[dict[str, Any]] = []
    results_root = Path(bench_config.RESULTS_LOCAL_DIR)
    for db_dir in sorted(results_root.iterdir()):
        if not db_dir.is_dir() or db_dir.name == DB.YDB.value:
            continue
        for result_file in db_dir.glob("result_*.json"):
            if not any(label in result_file.name for label in task_labels):
                continue
            try:
                payload = ujson.loads(result_file.read_text())
            except Exception:
                continue
            if payload.get("task_label") not in task_labels:
                continue
            for entry in payload.get("results", []):
                case_config = entry.get("task_config", {}).get("case_config", {})
                if case_config.get("case_id") != case_id:
                    continue
                metrics = entry.get("metrics", {})
                baselines.append(
                    {
                        "db": entry.get("task_config", {}).get("db"),
                        "task_label": payload.get("task_label"),
                        "source_file": str(result_file),
                        "db_case_config": entry.get("task_config", {}).get("db_case_config", {}),
                        "recall": metrics.get("recall"),
                        "qps": metrics.get("qps"),
                        "serial_latency_p99": metrics.get("serial_latency_p99"),
                    }
                )
    return baselines


def _recommend(records: list[TuneRunRecord], tune: TuneConfig) -> dict[str, Any] | None:
    finalized = [record for record in records if record.phase == "finalize" and record.recall and record.qps]
    if finalized:
        best = max(finalized, key=_pareto_score)
    else:
        serial = [record for record in records if record.recall]
        if not serial:
            return None
        best = max(serial, key=lambda record: (record.recall or 0.0, record.qps or 0.0))

    return {
        "kmeans_tree_search_top_size": best.kmeans_tree_search_top_size,
        "levels": best.levels,
        "clusters": best.clusters,
        "overlap_clusters": best.overlap_clusters,
        "recall": best.recall,
        "qps": best.qps,
        "serial_latency_p99": best.serial_latency_p99,
        "db_label": best.db_label,
        "meets_target_recall": (best.recall or 0.0) >= tune.target_recall,
        "meets_min_recall": (best.recall or 0.0) >= tune.min_recall,
    }


def _write_report(
    output_dir: Path,
    records: list[TuneRunRecord],
    recommendation: dict[str, Any] | None,
    baselines: list[dict[str, Any]],
    tune: TuneConfig,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"ydb_tune_report_{timestamp}.json"
    csv_path = output_dir / f"ydb_tune_report_{timestamp}.csv"

    report = {
        "generated_at": datetime.now().isoformat(),
        "config": asdict(tune),
        "runs": [asdict(record) for record in records],
        "recommendation": recommendation,
        "baselines": baselines,
    }
    json_path.write_text(ujson.dumps(report, indent=2) + "\n")

    empty = TuneRunRecord("", "", "", 0, None, None, None)
    fieldnames = list(asdict(records[0]).keys()) if records else list(asdict(empty).keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    return json_path, csv_path


def run_tuning(tune: TuneConfig, *, phase: str = "full") -> Path:
    all_records: list[TuneRunRecord] = []

    for build_idx, build in enumerate(tune.index_build_grid):
        build_suffix = f"b{build_idx}"
        original_label = tune.db_label
        tune.db_label = f"{original_label}-{build_suffix}"
        build_table_name = _table_name_for_build(tune, build_idx)

        load_first = not tune.skip_load
        drop_old_first = not tune.skip_drop_old

        if phase in {"full", "load"} and not tune.skip_load:
            load_record = _run_benchmark(
                tune,
                phase="load",
                db_label=f"{tune.db_label}-loaded",
                top_size=tune.search_top_size_values[0],
                build=build,
                load=True,
                drop_old=True,
                search_serial=False,
                search_concurrent=False,
                table_name=build_table_name,
            )
            all_records.append(load_record)
            load_first = False
            drop_old_first = False

        if phase in {"full", "search_sweep"}:
            sweep_records = _serial_sweep(
                tune,
                build,
                load_first=load_first,
                drop_old_first=drop_old_first,
                values=tune.search_top_size_values,
                table_name=build_table_name,
            )
            all_records.extend(sweep_records)

            measured = [
                (record.kmeans_tree_search_top_size, record.recall)
                for record in sweep_records
                if record.recall is not None
            ]
            if measured and phase == "full":
                measured.sort(key=lambda item: item[0])
                below = [item for item in measured if item[1] < tune.target_recall]
                above = [item for item in measured if item[1] >= tune.target_recall]
                if below and above:
                    lo = below[-1][0]
                    hi = above[0][0]
                    if lo < hi:
                        all_records.extend(
                            _binary_search_top_size(
                                tune,
                                build,
                                (lo, hi),
                                tune.target_recall,
                                table_name=build_table_name,
                            )
                        )

        if phase in {"full", "finalize"}:
            finalists = _select_finalists(all_records, tune)
            all_records.extend(
                _finalize_runs(tune, build, finalists, table_name=build_table_name)
            )

        tune.db_label = original_label

    baselines = _load_baselines(tune.case_type, tune.baseline_task_labels)
    recommendation = _recommend(all_records, tune)
    output_dir = Path(tune.output_dir)
    json_path, csv_path = _write_report(output_dir, all_records, recommendation, baselines, tune)

    log.info("Report written to %s and %s", json_path, csv_path)
    if recommendation:
        log.info("Recommended config: %s", recommendation)
    return json_path


@click.command("ydb-tune")
@click.option(
    "--config-file",
    type=str,
    default="",
    help="YAML config (see vectordb_bench/config-files/ydb_tune_config.yml)",
)
@click.option("--case-type", type=str, default="", help="VectorDBBench case type")
@click.option("--task-label", type=str, default="", help="Task label for result files")
@click.option("--db-label", type=str, default="", help="Prefix for db_label in runs")
@click.option("--endpoint", type=str, default="", envvar="YDB_ENDPOINT", help="YDB endpoint")
@click.option("--database", type=str, default="", envvar="YDB_DATABASE", help="YDB database path")
@click.option("--target-recall", type=float, default=None, help="Target recall for binary search")
@click.option(
    "--search-top-size-values",
    type=str,
    default="",
    help="Comma-separated KMeansTreeSearchTopSize values for serial sweep",
)
@click.option(
    "--phase",
    type=click.Choice(["full", "load", "search-sweep", "finalize"], case_sensitive=False),
    default="full",
    show_default=True,
    help="Which tuning phase(s) to run",
)
@click.option("--skip-load", is_flag=True, help="Skip data loading (reuse existing table/index)")
@click.option("--skip-drop-old", is_flag=True, help="Do not drop table before load")
@click.option("--output-dir", type=str, default="", help="Directory for tune reports")
@click.option("--table-name", type=str, default="", help="Fixed YDB table name")
def ydb_tune(
    config_file: str,
    case_type: str,
    task_label: str,
    db_label: str,
    endpoint: str,
    database: str,
    target_recall: float | None,
    search_top_size_values: str,
    phase: str,
    skip_load: bool,
    skip_drop_old: bool,
    output_dir: str,
    table_name: str,
) -> None:
    """Automate YDB vector index parameter tuning (QPS + Recall tradeoff)."""
    logging.basicConfig(level=bench_config.LOG_LEVEL)

    tune = TuneConfig()
    if config_file:
        yaml_data = _load_yaml_config(_resolve_config_file(config_file))
        for key, value in yaml_data.items():
            key_normalized = key.replace("-", "_")
            if hasattr(tune, key_normalized):
                setattr(tune, key_normalized, value)

    if case_type:
        tune.case_type = case_type
    if task_label:
        tune.task_label = task_label
    if db_label:
        tune.db_label = db_label
    if endpoint:
        tune.endpoint = endpoint
    if database:
        tune.database = database
    if target_recall is not None:
        tune.target_recall = target_recall
    if search_top_size_values:
        tune.search_top_size_values = [int(value.strip()) for value in search_top_size_values.split(",") if value.strip()]
    if output_dir:
        tune.output_dir = output_dir
    if table_name:
        tune.table_name = table_name
    tune.skip_load = skip_load or tune.skip_load
    tune.skip_drop_old = skip_drop_old or tune.skip_drop_old

    normalized_phase = phase.replace("-", "_")
    report_path = run_tuning(tune, phase=normalized_phase)
    click.echo(f"Tuning complete. Report: {report_path}")
