from __future__ import annotations

import json
from dataclasses import dataclass, field, is_dataclass
from math import inf, nan
from typing import Any, ClassVar, Dict, Optional
from warnings import warn

import polars as pl
import pyarrow.parquet as pq
from polars.testing import assert_frame_equal


class ConversionWarning(Warning):
    """Warns if Serialization and Deserialization will change data types."""

    pass


class MyJSONEncoder(json.JSONEncoder):
    """JSONEncoder that can handle data classes.

    Usage:
    ```python
    @dataclass(frozen=True)
    class Foo:
        bar: str

    foo = Foo(bar="baz")
    encoded = json.dumps(foo, cls=MyJSONEncoder)
    ```

    You can extend it to handle other custom types.
    """

    def default(self, o):
        if is_dataclass(o):
            return {f"__python__/dataclasses/{o.__class__.__name__}": vars(o)}
        return super.default(o)


class MyJSONDecoder(json.JSONDecoder):
    """JSONDecoder that can handle data classes.

    Usage:
    ```python
    @dataclass(frozen=True)
    class Foo:
        bar: str

    foo = Foo(bar="baz")
    encoded = json.dumps(foo, cls=MyJSONEncoder)
    foo_copy = json.dumps(foo, cls=MyJSONDecoder, classes=[Foo])
    assert foo==foo_copy
    ```

    You can extend it to handle other custom types.
    """

    def __init__(self, *args, classes=[], **kwargs):
        self.dataclass_name_mapping = {cls.__name__: cls for cls in classes}
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        for cls_name in self.dataclass_name_mapping:
            identifier = f"__python__/dataclasses/{cls_name}"
            if identifier in dct:
                return self.dataclass_name_mapping[cls_name](**dct[identifier])
        return dct


@dataclass
class MetaData:
    """Some dataclass for testing."""

    foo: str = "Foo"
    bar: float = inf
    baz: Optional[int] = None
    units: Dict[str, str] = field(default_factory=dict)

    def assert_eq(self, other):
        assert self == other, f"{self=}, {other=}"


@dataclass
class DataSet:
    """Custom Dataset that saves metadata within a parquet file as utf-8 encoded json."""

    metadata: Optional[MetaData]
    dataframe: pl.DataFrame
    _class_id: ClassVar[str] = "__python__/my_namespace/DataSet"

    def __post_init__(self):
        self.class_id = f"__python__/datasets/{self.__class__.__name__}"

    def assert_eq(self, other: DataSet, **kwargs):
        """Assert that self==other. Required since `dataframe` is a `polars.DataFrame`.

        Args:
            other (DataSet): Object to compare to.
        """
        assert isinstance(other, self.__class__)
        self.metadata.assert_eq(other.metadata)
        self.metadata.assert_eq(other.metadata)
        assert_frame_equal(self.dataframe, other.dataframe, **kwargs)

    def write_parquet(self, location: Any, **kwargs):
        """Serialize DataSet to parquet.

        Args:
            file (Any): Any file location that can be handled by `pyarrow.parquet.write_table`.
            **kwargs  : Will be passed on to `pyarrow.parquet.write_table`.
        """
        # dump metadata to a utf-8 encoded json string using custom encoder.
        metadata_bytes = json.dumps(self.metadata, cls=MyJSONEncoder).encode("utf-8")
        # convert `dataframe` to Pyarrow table.
        table = self.dataframe.to_arrow()
        # Add own metadata to the table schema.
        existing_metadata = table.schema.metadata or {}
        new_metadata = {
            self._class_id.encode("utf-8"): metadata_bytes,
            **existing_metadata,
        }
        table = table.replace_schema_metadata(new_metadata)

        # Categorical columns with lexical ordering will be mapped to Categorical(ordering="physical")
        for col in self.dataframe:
            if col.dtype.is_(pl.Categorical("lexical")):
                warn(
                    f"Column {col.name} with dtype {col.dtype} will be converted to {pl.Categorical("physical")}",
                    ConversionWarning,
                )

        # Write table to parquet.
        if partition_cols := kwargs.get("partition_cols"):
            # All partition columns will be cast to `Categorical(ordering="physical")`.
            for partion_col in partition_cols:
                if not self.dataframe[partion_col].dtype.is_(
                    pl.Categorical("physical")
                ):
                    warn(
                        f"Column {partion_col} of dtype {self.dataframe[partion_col].dtype}"
                        f"will be converted to {pl.Categorical("physical")}.",
                        ConversionWarning,
                    )
            pq.write_to_dataset(table=table, root_path=location, **kwargs)
        else:
            pq.write_table(table=table, where=location, **kwargs)

    @classmethod
    def read_parquet(cls, location: Any, **kwargs) -> DataSet:
        """Deserialize `DataSet` from parquet.

        Args:
            file (Any): where to store the data. Anything supported by Pyarrow
            works.

        Returns:
            DataSet: Deserialized `DataSet`.
        """
        table = pq.read_table(location, **kwargs)
        # get metadata from table
        table_metadata = table.schema.metadata
        try:  # Try to parse `MetaData` from `table_metadata`.
            dataset_metadata_str = table_metadata.pop(
                cls._class_id.encode("utf-8")
            ).decode("utf-8")
            dataset_metadata = json.loads(
                dataset_metadata_str, cls=MyJSONDecoder, classes=[MetaData]
            )
        except (KeyError, json.JSONDecodeError):
            dataset_metadata = None
        # Replace metadata from original table.
        table = table.replace_schema_metadata(table_metadata)
        dataframe = pl.from_arrow(table)
        return DataSet(metadata=dataset_metadata, dataframe=dataframe)


if __name__ == "__main__":
    import shutil
    from io import BytesIO
    from pathlib import Path

    try:
        # MetaData test
        metadata = MetaData()
        serialized = json.dumps(metadata, cls=MyJSONEncoder)
        deserialized = json.loads(serialized, cls=MyJSONDecoder, classes=[MetaData])
        metadata.assert_eq(deserialized)

        print(serialized)

        # DataSet test
        # Partitioning casts column to categorical.
        dataframe = pl.DataFrame(
            {
                "partition": ["1", "1", "2", "2"],
                "integer": [4, 5, 6, None],
                "float": [4.0, None, nan, inf],
                "string": ["d", "e", "f", None],
            },
        ).with_columns(pl.col("partition").cast(pl.Categorical(ordering="physical")))

        dataset = DataSet(metadata=metadata, dataframe=dataframe)
        with BytesIO() as f:
            dataset.write_parquet(f)
            deserialized_dataset = DataSet.read_parquet(f)

        dataset.assert_eq(deserialized_dataset)
        print(deserialized_dataset.metadata)
        print(deserialized_dataset.dataframe)

        # With partitions
        Path("./data").mkdir(exist_ok=True)
        dataset.write_parquet(
            "data/dataset",
            partition_cols=["partition"],
            existing_data_behavior="delete_matching",
        )
        partitioned_dataset = DataSet.read_parquet("data/dataset")

        dataset.assert_eq(partitioned_dataset, check_column_order=False)
        print(partitioned_dataset)
    except Exception as e:
        raise e
    finally:
        shutil.rmtree("./data", ignore_errors=True)
        shutil.rmtree("./data", ignore_errors=True)
