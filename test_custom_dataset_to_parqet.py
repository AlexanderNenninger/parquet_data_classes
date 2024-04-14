import io
import shutil
from math import inf, nan
from pathlib import Path

import polars as pl
import polars.testing
import polars.testing.parametric
import pytest
from hypothesis import given

from custom_dataset_to_parquet import ConversionWarning, DataSet, MetaData

pl.enable_string_cache()


@pytest.fixture()
def dataframe():
    return pl.DataFrame(
        {
            "partition": ["1", "1", "2", "2"],
            "lexical_partition": ["1", "1", "2", "2"],
            "string_partition": ["1", "2", "1", "2"],
            "int_partition": [1, 2, 1, 2],
            "integer": [1, 2, 3, None],
            "float": [4.0, None, nan, inf],
            "string": ["d", "e", "f", None],
            "categorical": ["a", "b", "a", None],
            "list": [[1, 2], [3, None, 5], None, [7, 8, 9]],
            "datetime": ["2024-01-01", "2024-02-29T16:32", None, None],
        },
        schema_overrides={
            "partition": pl.Categorical(ordering="physical"),
            "lexical_partition": pl.Categorical(ordering="lexical"),
            "categorical": pl.Categorical(ordering="physical"),
            "list": pl.List(pl.Int32),
        },
    ).with_columns(
        pl.col("datetime").str.strptime(
            dtype=pl.Datetime(time_unit="ms", time_zone="Europe/Berlin"),
        )
    )


@pytest.fixture
def metadata():
    return MetaData()


@pytest.fixture
def dataset(metadata, dataframe):
    return DataSet(metadata, dataframe)


def test_write_read_without_partitioning(dataset, tmp_path):
    with pytest.warns(ConversionWarning):
        dataset.write_parquet(tmp_path / "no-partition.parquet")
        deserialized = DataSet.read_parquet(tmp_path / "no-partition.parquet")
        dataset.assert_eq(deserialized, check_dtype=False)
        # Changed dtype.
        assert deserialized.dataframe["lexical_partition"].dtype.is_(
            pl.Categorical("physical")
        ), f"{deserialized.dataframe["lexical_partition"].dtype}"
        pl.testing.assert_frame_equal(
            dataset.dataframe.select(pl.exclude("lexical_partition")),
            deserialized.dataframe.select(pl.exclude("lexical_partition")),
        )


def test_write_read_with_partitioning(dataset, tmp_path):
    # No warnings
    with pytest.warns(ConversionWarning):
        dataset.write_parquet(
            tmp_path / "partition_no_warning", partition_cols=["partition"]
        )
        deserialized = DataSet.read_parquet(tmp_path / "partition_no_warning")
        dataset.assert_eq(deserialized, check_column_order=False, check_dtype=False)
        # Changed dtype
        assert deserialized.dataframe["lexical_partition"].dtype.is_(
            pl.Categorical("physical")
        ), f"{deserialized.dataframe["lexical_partition"].dtype}"


def test_lexical_partition(dataset, tmp_path):
    with pytest.warns(ConversionWarning):
        partion_col = "lexical_partition"
        dataset.write_parquet(tmp_path / partion_col, partition_cols=[partion_col])
        deserialized = DataSet.read_parquet(tmp_path / partion_col)
        assert dataset.dataframe[partion_col].dtype.is_(pl.Categorical("lexical"))
        assert deserialized.dataframe[partion_col].dtype.is_(pl.Categorical("physical"))
        polars.testing.assert_frame_equal(
            dataset.dataframe[[partion_col, "integer"]],
            deserialized.dataframe[[partion_col, "integer"]],
            check_row_order=False,
            check_column_order=False,
            check_dtype=False,
        )


def test_string_partition(dataset, tmp_path):
    with pytest.warns(ConversionWarning):
        partion_col = "string_partition"
        dataset.write_parquet(tmp_path / partion_col, partition_cols=[partion_col])
        deserialized = DataSet.read_parquet(tmp_path / partion_col)
        polars.testing.assert_frame_equal(
            dataset.dataframe[[partion_col, "integer"]],
            deserialized.dataframe[[partion_col, "integer"]],
            check_row_order=False,
            check_column_order=False,
            check_dtype=False,
        )


def test_int_partition(dataset, tmp_path):
    with pytest.warns(ConversionWarning):
        partion_col = "int_partition"
        dataset.write_parquet(tmp_path / partion_col, partition_cols=[partion_col])
        deserialized = DataSet.read_parquet(tmp_path / partion_col)
        polars.testing.assert_frame_equal(
            dataset.dataframe[[partion_col, "integer"]],
            deserialized.dataframe.select(
                pl.col(partion_col).cast(pl.Int64), "integer"
            ),
            check_row_order=False,
            check_column_order=False,
            check_dtype=False,
        )


@given(polars.testing.parametric.dataframes(max_size=100, null_probability=0.1))
def test_property(dataframe: pl.DataFrame):
    dataset = DataSet(MetaData(), dataframe)
    with io.BytesIO() as bio:
        dataset.write_parquet(bio)
        deserialized = DataSet.read_parquet(bio)
    deserialized.assert_eq(
        dataset,
        check_dtype=False,
    )


@pytest.fixture
def datafile():
    return "jobs.parquet"


@pytest.fixture
def benchmark_dataframe(datafile):
    return pl.read_parquet(datafile).drop_nulls(["WindowID", "Country", "State"])


@pytest.fixture
def benchmark_metadata(datafile):
    schema = pl.read_parquet_schema(datafile)
    descriptions = {
        k: "Lorem ipsum dolor sit amet, consectetuer adipiscing elit. "
        "Aenean commodo ligula eget dolor. Aenean massa. Cum sociis natoque penatibus et "
        for k in schema
    }
    return MetaData(description=descriptions)


@pytest.fixture
def benchmark_dataset(benchmark_dataframe, benchmark_metadata):
    return DataSet(dataframe=benchmark_dataframe, metadata=benchmark_metadata)


@pytest.fixture
def benchmark_path():
    p = Path("./data/benchmark")
    p.mkdir(exist_ok=True, parents=True)
    yield p
    shutil.rmtree(p, ignore_errors=True)


@pytest.fixture
def deser_dataset_path(benchmark_dataset, benchmark_path):
    p = benchmark_path / "deser.parquet"
    benchmark_dataset.write_parquet(
        p,
    )
    return str(p)


@pytest.fixture
def deser_partitioned_dataset_path(benchmark_dataset, benchmark_path):
    p = benchmark_path / "partitioned/deser"
    benchmark_dataset.write_parquet(p, partition_cols=["WindowID", "Country"])
    return str(p)


def test_benchmark_serialization(benchmark_dataset, benchmark, benchmark_path):
    _ = benchmark(
        benchmark_dataset.write_parquet,
        str(benchmark_path / "benchmark_serialization.parquet"),
    )


def test_benchmark_deserialization(benchmark, deser_dataset_path):
    _ = benchmark(DataSet.read_parquet, deser_dataset_path)


def test_benchmark_deserialization_partitioned(
    benchmark, deser_partitioned_dataset_path
):
    _ = benchmark(DataSet.read_parquet, deser_partitioned_dataset_path)
