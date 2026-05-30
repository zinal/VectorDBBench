import os
from typing import ClassVar, TypedDict

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from vectordb_bench.backend.filter import Filter, FilterOp, non_filter

from ..api import DBCaseConfig, DBConfig, MetricType


class YDBConfigDict(TypedDict, total=False):
    endpoint: str
    database: str
    auth_mode: str
    token: str
    user: str
    password: str
    table_name: str


class YDBConfig(DBConfig):
    _extra_empty_skip: ClassVar[frozenset[str]] = frozenset({"password", "token", "user"})

    endpoint: str = "grpc://localhost:2136"
    database: str = "/local"
    auth_mode: str = "env"
    token: SecretStr | None = None
    user: str = ""
    password: SecretStr | None = None
    table_name: str = ""

    @model_validator(mode="before")
    @classmethod
    def apply_env_defaults(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if not data.get("endpoint") and os.environ.get("YDB_ENDPOINT"):
            data["endpoint"] = os.environ["YDB_ENDPOINT"]
        if not data.get("database") and os.environ.get("YDB_DATABASE"):
            data["database"] = os.environ["YDB_DATABASE"]
        return data

    def to_dict(self) -> YDBConfigDict:
        token_str = self.token.get_secret_value() if self.token else ""
        password_str = self.password.get_secret_value() if self.password else ""
        result: YDBConfigDict = {
            "endpoint": self.endpoint,
            "database": self.database,
            "auth_mode": self.auth_mode,
            "token": token_str,
            "user": self.user,
            "password": password_str,
        }
        if self.table_name:
            result["table_name"] = self.table_name
        return result


def compute_kmeans_tree_params(
    data_size: int,
    levels: int | None = None,
    clusters: int | None = None,
) -> tuple[int, int]:
    """Pick index shape so leaf clusters stay small (see YDB kmeans-tree docs)."""
    if levels is None:
        if data_size < 100_000:
            levels = 1
        elif data_size < 1_000_000:
            levels = 2
        else:
            levels = 3

    if clusters is None:
        target_leaf_size = 512
        clusters = int(round((max(data_size, 1) / target_leaf_size) ** (1.0 / levels)))
        clusters = max(20, min(512, clusters))

    return levels, clusters


def index_on_columns(filters: Filter) -> tuple[str, ...]:
    if filters.type == FilterOp.NumGE:
        return ("id", "embedding")
    if filters.type == FilterOp.StrEqual:
        return ("labels", "embedding")
    return ("embedding",)


class YDBIndexConfig(BaseModel, DBCaseConfig):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    metric_type: MetricType | None = None
    create_index_after_load: bool = True
    level: int | None = Field(default=None, alias="levels")
    nlist: int | None = Field(default=None, alias="clusters")
    num_leaves_to_search: int = Field(default=10, alias="kmeans_tree_search_top_size")
    overlap_clusters: int = 3
    cover_embedding: bool = True

    def index_strategy(self) -> str:
        if self.metric_type == MetricType.L2:
            return "distance=euclidean"
        if self.metric_type == MetricType.IP:
            return "similarity=inner_product"
        return "similarity=cosine"

    def knn_function(self) -> str:
        if self.metric_type == MetricType.L2:
            return "EuclideanDistance"
        if self.metric_type == MetricType.IP:
            return "InnerProductSimilarity"
        return "CosineSimilarity"

    def sort_order(self) -> str:
        if self.metric_type in (MetricType.L2,):
            return "ASC"
        if self.metric_type == MetricType.IP:
            return "DESC"
        return "DESC"

    def resolved_index_params(self, data_size: int | None) -> tuple[int, int]:
        size = data_size or 1
        if self.level is not None and self.nlist is not None:
            return self.level, self.nlist
        return compute_kmeans_tree_params(size, self.level, self.nlist)

    def index_on_columns(self, filters: Filter = non_filter) -> tuple[str, ...]:
        return index_on_columns(filters)

    def cover_clause(self) -> str:
        if not self.cover_embedding:
            return ""
        return "COVER (embedding)"

    def index_param(self, filters: Filter = non_filter) -> dict:
        levels, clusters = self.resolved_index_params(None)
        on_columns = self.index_on_columns(filters)
        return {
            "strategy": self.index_strategy(),
            "levels": levels,
            "clusters": clusters,
            "overlap_clusters": self.overlap_clusters,
            "on_columns": on_columns,
            "cover_clause": self.cover_clause(),
        }

    def search_param(self) -> dict:
        return {
            "knn_function": self.knn_function(),
            "sort_order": self.sort_order(),
            "kmeans_tree_search_top_size": self.num_leaves_to_search,
        }
