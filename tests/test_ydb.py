from unittest.mock import MagicMock, patch
import os

import numpy as np
import pytest

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.ydb.config import (
    YDBConfig,
    YDBIndexConfig,
    compute_kmeans_tree_params,
)
from vectordb_bench.backend.clients.ydb.ydb_client import YDB


def _integration_db_config() -> dict:
    return YDBConfig(
        endpoint=os.environ.get("YDB_ENDPOINT", "grpc://localhost:2136"),
        database=os.environ.get("YDB_DATABASE", "/Root/test"),
        auth_mode=os.environ.get("YDB_AUTH_MODE", "env"),
    ).to_dict()


class TestYDBConfig:
    def test_compute_kmeans_tree_params_small_dataset(self):
        levels, clusters = compute_kmeans_tree_params(50_000)
        assert levels == 1
        assert 20 <= clusters <= 512

    def test_compute_kmeans_tree_params_medium_dataset(self):
        levels, clusters = compute_kmeans_tree_params(500_000)
        assert levels == 2
        assert clusters >= 20

    def test_compute_kmeans_tree_params_large_dataset(self):
        levels, clusters = compute_kmeans_tree_params(10_000_000)
        assert levels == 3
        assert clusters >= 20

    def test_metric_mapping(self):
        cosine = YDBIndexConfig(metric_type=MetricType.COSINE)
        assert cosine.index_strategy() == "similarity=cosine"
        assert cosine.knn_function() == "CosineSimilarity"

        l2 = YDBIndexConfig(metric_type=MetricType.L2)
        assert l2.index_strategy() == "distance=euclidean"
        assert l2.knn_function() == "EuclideanDistance"

        ip = YDBIndexConfig(metric_type=MetricType.IP)
        assert ip.index_strategy() == "similarity=inner_product"
        assert ip.knn_function() == "InnerProductSimilarity"


class TestYDBAuth:
    def test_build_credentials_login_from_env(self, monkeypatch):
        monkeypatch.setenv("YDB_USER", "bench")
        monkeypatch.setenv("YDB_PASSWORD", "secret")

        with patch("ydb.StaticCredentials") as static_credentials:
            YDB._build_credentials({"auth_mode": "env", "user": "", "password": ""})
            static_credentials.assert_called_once_with("bench", "secret")

    def test_build_credentials_login_mode_cli_overrides_env(self, monkeypatch):
        monkeypatch.setenv("YDB_USER", "env-user")
        monkeypatch.setenv("YDB_PASSWORD", "env-pass")

        with patch("ydb.StaticCredentials") as static_credentials:
            YDB._build_credentials({"auth_mode": "login", "user": "cli-user", "password": "cli-pass"})
            static_credentials.assert_called_once_with("cli-user", "cli-pass")

    def test_build_credentials_login_mode_requires_user(self):
        with pytest.raises(ValueError, match="YDB_USER"):
            YDB._build_credentials({"auth_mode": "login", "user": "", "password": ""})

    def test_build_credentials_anonymous(self):
        with patch("ydb.AnonymousCredentials") as anonymous_credentials:
            YDB._build_credentials({"auth_mode": "anonymous"})
            anonymous_credentials.assert_called_once_with()

    def test_build_credentials_env_defaults_to_anonymous(self, monkeypatch):
        monkeypatch.delenv("YDB_USER", raising=False)
        monkeypatch.delenv("YDB_PASSWORD", raising=False)
        monkeypatch.delenv("YDB_ACCESS_TOKEN_CREDENTIALS", raising=False)
        monkeypatch.delenv("YDB_SERVICE_ACCOUNT_KEY_FILE_CREDENTIALS", raising=False)
        monkeypatch.delenv("YDB_OAUTH2_KEY_FILE", raising=False)
        monkeypatch.delenv("YDB_ANONYMOUS_CREDENTIALS", raising=False)
        monkeypatch.delenv("YDB_METADATA_CREDENTIALS", raising=False)

        with patch("ydb.AnonymousCredentials") as anonymous_credentials:
            YDB._build_credentials({"auth_mode": "env", "user": "", "password": ""})
            anonymous_credentials.assert_called_once_with()

    def test_build_credentials_env_uses_sdk_when_configured(self, monkeypatch):
        monkeypatch.delenv("YDB_USER", raising=False)
        monkeypatch.setenv("YDB_ACCESS_TOKEN_CREDENTIALS", "token-value")

        with patch("ydb.credentials_from_env_variables") as from_env:
            from_env.return_value = MagicMock()
            YDB._build_credentials({"auth_mode": "env", "user": "", "password": ""})
            from_env.assert_called_once_with()


@pytest.mark.integration
class TestYDBClient:
    @pytest.fixture
    def db_client(self):
        db_cls = DB.YDB.init_cls
        db_config = _integration_db_config()

        dim = 16
        try:
            client = db_cls(
                dim=dim,
                db_config=db_config,
                db_case_config=YDBIndexConfig(
                    metric_type=MetricType.COSINE,
                    levels=1,
                    clusters=8,
                    kmeans_tree_search_top_size=3,
                ),
                collection_name="vdbbench_ydb_test",
                drop_old=True,
            )
        except (TimeoutError, OSError) as e:
            pytest.skip(f"YDB is not available at {db_config['endpoint']}{db_config['database']}: {e}")
        except Exception as e:
            if e.__class__.__module__.startswith("ydb"):
                pytest.skip(f"YDB is not available at {db_config['endpoint']}{db_config['database']}: {e}")
            raise
        return client, dim

    def test_insert_optimize_and_search(self, db_client):
        client, dim = db_client
        count = 1000
        embeddings = np.random.default_rng(42).random((count, dim)).tolist()

        with client.init():
            inserted, err = client.insert_embeddings(embeddings=embeddings, metadata=list(range(count)))
            assert err is None
            assert inserted == count

            client.optimize(data_size=count)

            test_id = 42
            results = client.search_embedding(query=embeddings[test_id], k=10)
            assert len(results) == 10
            assert int(results[0]) == test_id
