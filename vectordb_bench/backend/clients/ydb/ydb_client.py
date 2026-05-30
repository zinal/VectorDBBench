import logging
import os
import struct
from contextlib import contextmanager
from typing import Any

from ..api import VectorDB
from .config import YDBIndexConfig

log = logging.getLogger(__name__)

YDB_USER_ENV = "YDB_USER"
YDB_PASSWORD_ENV = "YDB_PASSWORD"
YDB_CREDENTIAL_ENV_KEYS = (
    "YDB_SERVICE_ACCOUNT_KEY_FILE_CREDENTIALS",
    YDB_USER_ENV,
    "YDB_ACCESS_TOKEN_CREDENTIALS",
    "YDB_OAUTH2_KEY_FILE",
)


def convert_vector_to_bytes(vector: list[float]) -> bytes:
    values = [float(v) for v in vector]
    packed = struct.pack(f"<{len(values)}f", *values)
    return packed + b"\x01"


class YDB(VectorDB):
    """YDB vector search client using vector_kmeans_tree indexes."""

    thread_safe = True

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: YDBIndexConfig,
        collection_name: str = "vdbbench_ydb",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "YDB"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.index_name = f"{collection_name}_vector_idx"
        self.dim = dim

        self.driver = None
        self.pool = None

        if drop_old:
            with self._session_pool() as pool:
                self._drop_table(pool)
                self._create_table(pool)

    @staticmethod
    def _resolve_login(db_config: dict) -> tuple[str, str]:
        user = db_config.get("user") or os.environ.get(YDB_USER_ENV, "")
        password = db_config.get("password") or os.environ.get(YDB_PASSWORD_ENV, "")
        return user, password

    @staticmethod
    def _has_sdk_credentials_env() -> bool:
        if any(os.environ.get(key) for key in YDB_CREDENTIAL_ENV_KEYS):
            return True
        if os.environ.get("YDB_ANONYMOUS_CREDENTIALS", "0") == "1":
            return True
        return os.environ.get("YDB_METADATA_CREDENTIALS", "0") == "1"

    @staticmethod
    def _build_credentials(db_config: dict):
        import ydb

        auth_mode = db_config.get("auth_mode", "env")
        if auth_mode == "anonymous":
            return ydb.AnonymousCredentials()
        if auth_mode == "token":
            token = db_config.get("token") or os.environ.get("YDB_ACCESS_TOKEN_CREDENTIALS", "")
            if not token:
                msg = "auth_mode=token requires a non-empty token"
                raise ValueError(msg)
            return ydb.AccessTokenCredentials(token)

        user, password = YDB._resolve_login(db_config)
        if auth_mode == "login" or user:
            if not user:
                msg = f"auth_mode=login requires --user or ${YDB_USER_ENV}"
                raise ValueError(msg)
            return ydb.StaticCredentials(user, password)

        if YDB._has_sdk_credentials_env():
            return ydb.credentials_from_env_variables()

        log.debug("No YDB credentials in env; using anonymous auth for local server")
        return ydb.AnonymousCredentials()

    @contextmanager
    def _session_pool(self):
        import ydb

        credentials = self._build_credentials(self.db_config)
        driver = ydb.Driver(
            endpoint=self.db_config["endpoint"],
            database=self.db_config["database"],
            credentials=credentials,
        )
        pool = None
        try:
            driver.wait(timeout=5, fail_fast=True)
            pool = ydb.QuerySessionPool(driver)
            yield pool
        finally:
            if pool is not None:
                pool.stop()
            driver.stop()

    @contextmanager
    def init(self):
        import ydb

        credentials = self._build_credentials(self.db_config)
        self.driver = ydb.Driver(
            endpoint=self.db_config["endpoint"],
            database=self.db_config["database"],
            credentials=credentials,
        )
        self.driver.wait(timeout=5, fail_fast=True)
        self.pool = ydb.QuerySessionPool(self.driver)
        try:
            yield
        finally:
            if self.pool is not None:
                self.pool.stop()
                self.pool = None
            if self.driver is not None:
                self.driver.stop()
                self.driver = None

    def _drop_table(self, pool) -> None:
        pool.execute_with_retries(f"DROP TABLE IF EXISTS `{self.table_name}`")
        log.info("Dropped table %s", self.table_name)

    def _create_table(self, pool) -> None:
        pool.execute_with_retries(
            f"""
            CREATE TABLE IF NOT EXISTS `{self.table_name}` (
                id Uint64 NOT NULL,
                embedding String NOT NULL,
                PRIMARY KEY (id)
            );
            """
        )
        log.info("Created table %s", self.table_name)

    def _add_vector_index(self, pool, levels: int, clusters: int) -> None:
        import ydb

        index_param = self.case_config.index_param()
        strategy = index_param["strategy"]
        temp_index_name = f"{self.index_name}__temp"
        overlap_clusters = index_param.get("overlap_clusters", 3)

        pool.execute_with_retries(
            f"""
            ALTER TABLE `{self.table_name}`
            ADD INDEX {temp_index_name}
            GLOBAL USING vector_kmeans_tree
            ON (embedding)
            WITH (
                {strategy},
                vector_type="Float",
                vector_dimension={self.dim},
                levels={levels},
                clusters={clusters},
                overlap_clusters={overlap_clusters}
            );
            """
        )

        table_path = f"{self.driver._driver_config.database}/{self.table_name}"
        self.driver.table_client.alter_table(
            table_path,
            rename_indexes=[
                ydb.RenameIndexItem(
                    source_name=temp_index_name,
                    destination_name=self.index_name,
                    replace_destination=True,
                ),
            ],
        )
        log.info(
            "Created vector index %s on %s (levels=%d, clusters=%d)",
            self.index_name,
            self.table_name,
            levels,
            clusters,
        )

    def optimize(self, data_size: int | None = None) -> None:
        if not self.case_config.create_index_after_load:
            log.info("Skipping vector index build (create_index_after_load=False)")
            return

        levels, clusters = self.case_config.resolved_index_params(data_size)
        log.info(
            "Building YDB vector index for %d rows: levels=%d, clusters=%d",
            data_size or 0,
            levels,
            clusters,
        )
        self._add_vector_index(self.pool, levels=levels, clusters=clusters)

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception]:
        import ydb

        if not embeddings:
            return 0, None

        batch_size = 1000
        items_struct_type = ydb.StructType()
        items_struct_type.add_member("id", ydb.PrimitiveType.Uint64)
        items_struct_type.add_member("embedding", ydb.PrimitiveType.String)

        query = f"""
        DECLARE $items AS List<Struct<
            id: Uint64,
            embedding: String
        >>;

        UPSERT INTO `{self.table_name}` (id, embedding)
        SELECT id, embedding
        FROM AS_TABLE($items);
        """

        inserted = 0
        for offset in range(0, len(embeddings), batch_size):
            end = min(offset + batch_size, len(embeddings))
            items = [
                {
                    "id": metadata[i],
                    "embedding": convert_vector_to_bytes(embeddings[i]),
                }
                for i in range(offset, end)
            ]
            self.pool.execute_with_retries(
                query,
                {"$items": (items, ydb.ListType(items_struct_type))},
            )
            inserted += len(items)

        return inserted, None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        import ydb

        search_param = self.case_config.search_param()
        knn_function = search_param["knn_function"]
        sort_order = search_param["sort_order"]
        top_clusters = search_param["kmeans_tree_search_top_size"]

        use_index = self.case_config.create_index_after_load
        view_clause = f"VIEW {self.index_name}" if use_index else ""

        yql = f"""
        PRAGMA ydb.KMeansTreeSearchTopSize = "{top_clusters}";
        DECLARE $embedding AS String;

        SELECT id
        FROM `{self.table_name}` {view_clause}
        ORDER BY Knn::{knn_function}(embedding, $embedding) {sort_order}
        LIMIT {k};
        """

        result_sets = self.pool.execute_with_retries(
            yql,
            {
                "$embedding": (
                    convert_vector_to_bytes(query),
                    ydb.PrimitiveType.String,
                ),
            },
        )

        ids: list[int] = []
        for result_set in result_sets:
            for row in result_set.rows:
                ids.append(int(row["id"]))
        return ids
