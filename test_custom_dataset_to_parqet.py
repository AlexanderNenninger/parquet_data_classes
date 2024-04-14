from math import inf, nan

import polars as pl
import polars.testing
import polars.testing.parametric
import pytest
from hypothesis import given

from custom_dataset_to_parquet import ConversionWarning, DataSet, MetaData


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


@given(
    polars.testing.parametric.dataframes(
        cols=100, size=100, allowed_dtypes=pl.FLOAT_DTYPES | pl.DATETIME_DTYPES
    ),
    include_cols=[pl.Categorical("physical")],
)
def large_dataset(df: pl.DataFrame):
    print(df)
