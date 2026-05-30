from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import CommonTypedDict, cli, click_parameter_decorators_from_typed_dict, run


class YDBTypedDict(CommonTypedDict):
    endpoint: Annotated[
        str,
        click.option(
            "--endpoint",
            type=str,
            default="grpc://localhost:2136",
            show_default=True,
            envvar="YDB_ENDPOINT",
            help="YDB gRPC endpoint",
        ),
    ]
    database: Annotated[
        str,
        click.option(
            "--database",
            type=str,
            default="/local",
            show_default=True,
            envvar="YDB_DATABASE",
            help="YDB database path",
        ),
    ]
    auth_mode: Annotated[
        str,
        click.option(
            "--auth-mode",
            type=click.Choice(["env", "anonymous", "token", "login"], case_sensitive=False),
            default="env",
            show_default=True,
            help="Authentication mode: env vars (YDB_USER/YDB_PASSWORD or SDK creds), anonymous, token, or login",
        ),
    ]
    token: Annotated[
        str,
        click.option(
            "--token",
            type=str,
            default="",
            show_default=False,
            help="Access token when auth-mode=token",
        ),
    ]
    user: Annotated[
        str,
        click.option(
            "--user",
            type=str,
            default="",
            show_default=True,
            envvar="YDB_USER",
            help="Username for login auth (or set YDB_USER)",
        ),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            default="",
            show_default=False,
            envvar="YDB_PASSWORD",
            help="Password for login auth (or set YDB_PASSWORD)",
        ),
    ]
    levels: Annotated[
        int | None,
        click.option(
            "--levels",
            type=int,
            default=None,
            help="vector_kmeans_tree levels (auto if omitted)",
        ),
    ]
    clusters: Annotated[
        int | None,
        click.option(
            "--clusters",
            type=int,
            default=None,
            help="vector_kmeans_tree clusters per level (auto if omitted)",
        ),
    ]
    kmeans_tree_search_top_size: Annotated[
        int,
        click.option(
            "--kmeans-tree-search-top-size",
            type=int,
            default=10,
            show_default=True,
            help="PRAGMA ydb.KMeansTreeSearchTopSize for search completeness",
        ),
    ]
    overlap_clusters: Annotated[
        int,
        click.option(
            "--overlap-clusters",
            type=int,
            default=3,
            show_default=True,
            help="vector_kmeans_tree overlap_clusters (higher = better recall, larger index)",
        ),
    ]
    cover_embedding: Annotated[
        bool,
        click.option(
            "--cover-embedding/--no-cover-embedding",
            default=True,
            show_default=True,
            help="Store vectors in the index posting table (COVER embedding)",
        ),
    ]
    table_name: Annotated[
        str,
        click.option(
            "--table-name",
            type=str,
            default="",
            show_default=False,
            help="YDB table name (auto-generated per case if omitted)",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(YDBTypedDict)
def YDB(**parameters: Unpack[YDBTypedDict]):
    from .config import YDBConfig, YDBIndexConfig

    token = parameters["token"] or None
    password = parameters["password"] or None

    run(
        db=DB.YDB,
        db_config=YDBConfig(
            db_label=parameters["db_label"],
            endpoint=parameters["endpoint"],
            database=parameters["database"],
            auth_mode=parameters["auth_mode"],
            token=SecretStr(token) if token else None,
            user=parameters["user"],
            password=SecretStr(password) if password else None,
            table_name=parameters["table_name"],
        ),
        db_case_config=YDBIndexConfig(
            levels=parameters["levels"],
            clusters=parameters["clusters"],
            kmeans_tree_search_top_size=parameters["kmeans_tree_search_top_size"],
            overlap_clusters=parameters["overlap_clusters"],
            cover_embedding=parameters["cover_embedding"],
        ),
        **parameters,
    )
