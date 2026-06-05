"""Unit tests for YDB index tuning helpers."""

from pathlib import Path

import pytest

from vectordb_bench.backend.clients.ydb.tune import (
    TuneConfig,
    TuneRunRecord,
    _build_cli_args,
    _case_id,
    _ensure_benchmark_succeeded,
    _recommend,
    _select_finalists,
    _table_name_for_build,
)


def test_case_id_valid():
    assert _case_id("Performance768D1M") == 5


def test_case_id_invalid():
    with pytest.raises(Exception):
        _case_id("NotACase")


def test_build_cli_args_skip_load():
    tune = TuneConfig(case_type="Performance768D1M", endpoint="grpc://host:2136", database="/db")
    args = _build_cli_args(
        tune,
        db_label="test",
        top_size=20,
        build={"levels": 2, "clusters": 256},
        load=False,
        drop_old=False,
        search_serial=True,
        search_concurrent=False,
        table_name="performance768d1m_b1",
    )
    assert "--skip-load" in args
    assert "--skip-drop-old" in args
    assert "--search-serial" in args
    assert "--skip-search-concurrent" in args
    assert "--table-name" in args
    assert args[args.index("--table-name") + 1] == "performance768d1m_b1"
    assert "--kmeans-tree-search-top-size" in args
    idx = args.index("--kmeans-tree-search-top-size")
    assert args[idx + 1] == "20"


def test_table_name_for_build_is_unique_per_grid_entry():
    tune = TuneConfig(case_type="Performance768D1M")
    assert _table_name_for_build(tune, 0) == "performance768d1m_b0"
    assert _table_name_for_build(tune, 2) == "performance768d1m_b2"


def test_table_name_for_build_respects_explicit_override():
    tune = TuneConfig(case_type="Performance768D1M", table_name="fixed_table")
    assert _table_name_for_build(tune, 3) == "fixed_table"


def test_ensure_benchmark_succeeded_rejects_failed_label(tmp_path):
    result_file = tmp_path / "result_test.json"
    result_file.write_text(
        """
        {
          "results": [
            {
              "label": "x",
              "metrics": {"load_duration": 0.0},
              "task_config": {
                "db_config": {"db_label": "ydb-b2-loaded"},
                "case_config": {"case_id": 5}
              }
            }
          ]
        }
        """
    )
    with pytest.raises(RuntimeError, match="failed with label='x'"):
        _ensure_benchmark_succeeded(
            result_file,
            case_type="Performance768D1M",
            db_label="ydb-b2-loaded",
            phase="load",
        )


def test_ensure_benchmark_succeeded_requires_load_duration(tmp_path):
    result_file = tmp_path / "result_test.json"
    result_file.write_text(
        """
        {
          "results": [
            {
              "label": ":)",
              "metrics": {"load_duration": 0.0},
              "task_config": {
                "db_config": {"db_label": "ydb-b2-loaded"},
                "case_config": {"case_id": 5}
              }
            }
          ]
        }
        """
    )
    with pytest.raises(RuntimeError, match="Load phase for ydb-b2-loaded did not complete"):
        _ensure_benchmark_succeeded(
            result_file,
            case_type="Performance768D1M",
            db_label="ydb-b2-loaded",
            phase="load",
        )


def test_select_finalists_prefers_target_recall():
    tune = TuneConfig(target_recall=0.92, finalize_top_n=1)
    records = [
        TuneRunRecord("search_sweep", "Performance768D1M", "a", 10, None, None, None, recall=0.85),
        TuneRunRecord("search_sweep", "Performance768D1M", "b", 32, None, None, None, recall=0.93),
        TuneRunRecord("search_sweep", "Performance768D1M", "c", 64, None, None, None, recall=0.95),
    ]
    finalists = _select_finalists(records, tune)
    assert len(finalists) == 1
    assert finalists[0].kmeans_tree_search_top_size == 32


def test_recommend_from_finalize():
    tune = TuneConfig(target_recall=0.9)
    records = [
        TuneRunRecord("finalize", "Performance768D1M", "final", 20, None, None, None, recall=0.91, qps=1500.0),
        TuneRunRecord("search_sweep", "Performance768D1M", "sweep", 10, None, None, None, recall=0.95),
    ]
    rec = _recommend(records, tune)
    assert rec is not None
    assert rec["kmeans_tree_search_top_size"] == 20
    assert rec["qps"] == 1500.0
