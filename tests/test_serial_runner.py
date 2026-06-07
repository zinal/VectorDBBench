from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from vectordb_bench.backend.filter import non_filter
from vectordb_bench.backend.runner.serial_runner import SerialSearchRunner


class _ThreadSafeDB:
    thread_safe = True
    name = "MockDB"

    def supports_payload_profile(self, payload_profile):
        return True

    @contextmanager
    def init(self):
        yield

    def prepare_filter(self, filters):
        return None

    def search_embedding(self, query, k):
        return [0]


class _NotThreadSafeDB(_ThreadSafeDB):
    thread_safe = False


@pytest.mark.parametrize("db_cls", [_ThreadSafeDB, _NotThreadSafeDB])
def test_serial_search_runner_run(db_cls):
    runner = SerialSearchRunner(
        db=db_cls(),
        test_data=[[0.1, 0.2]],
        ground_truth=[[0]],
        k=1,
        filters=non_filter,
    )
    recall, ndcg, p99, p95 = runner.run()[0]
    assert recall == 1.0
    assert ndcg == 1.0
    assert p99 >= 0
    assert p95 >= 0


def test_serial_search_runs_in_process_when_thread_safe():
    runner = SerialSearchRunner(
        db=_ThreadSafeDB(),
        test_data=[[0.1]],
        ground_truth=[[0]],
        k=1,
    )
    with patch.object(SerialSearchRunner, "search", return_value=(0.5, 0.5, 0.01, 0.02)) as mock_search, patch(
        "vectordb_bench.backend.runner.serial_runner.concurrent.futures.ProcessPoolExecutor"
    ) as mock_pool:
        result, _ = runner.run()

    mock_pool.assert_not_called()
    mock_search.assert_called_once()
    assert result == (0.5, 0.5, 0.01, 0.02)


def test_serial_search_uses_spawn_subprocess_when_not_thread_safe():
    import multiprocessing as mp

    runner = SerialSearchRunner(
        db=_NotThreadSafeDB(),
        test_data=[[0.1]],
        ground_truth=[[0]],
        k=1,
    )
    with patch(
        "vectordb_bench.backend.runner.serial_runner.concurrent.futures.ProcessPoolExecutor"
    ) as mock_pool:
        mock_executor = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_executor
        mock_executor.submit.return_value.result.return_value = (0.5, 0.5, 0.01, 0.02)

        result, _ = runner.run()

    mock_pool.assert_called_once_with(
        mp_context=mp.get_context("spawn"),
        max_workers=1,
    )
    assert result == (0.5, 0.5, 0.01, 0.02)
