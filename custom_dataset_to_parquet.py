from math import inf, nan
from typing import Any, Optional, ClassVar
import polars as pl
import pyarrow.parquet as pq
from dataclasses import dataclass, is_dataclass
import json
from polars.testing import assert_frame_equal
from __future__ import annotations


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


@dataclass(frozen=True)
class MetaData:
    """Some dataclass for testing.
    """
    foo: str = "Foo"
    bar: float = inf
    baz: Optional[int] = None

    def assert_eq(self, other):
        assert self == other, f"{self=}, {other=}"


@dataclass
class DataSet:
    """Custom Dataset that saves metadata within a parquet file as utf-8 encoded json.
    """
    metadata: MetaData
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

    def write_parquet(self, file: Any, **kwargs):
        """Dump self to parquet.

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
        # Write table to parquet.
        pq.write_table(table=table, where=file, **kwargs)

    @classmethod
    def read_parquet(cls, file: Any, **kwargs) -> DataSet:
        """Deserialize `DataSet` from parquet.

        Args:
            file (Any): where to store the data. Anything supported by Pyarrow
            works.

        Returns:
            DataSet: Deserialized `DataSet`.
        """        
        table = pq.read_table(file, **kwargs)
        # get metadata from table
        table_metadata = table.schema.metadata
        try: # Try to parse `MetaData` from `table_metadata`.
            dataset_metadata_str = table_metadata.pop(cls._class_id.encode("utf-8")).decode(
                "utf-8"
            )
            dataset_metadata = json.loads(
                dataset_metadata_str, cls=MyJSONDecoder, classes=[MetaData]
            )
        except (KeyError, json.JSONDecodeError) as e:
            dataset_metadata=None
        # Replace metadata from original table.
        table = table.replace_schema_metadata(table_metadata)
        dataframe = pl.from_arrow(table)
        return DataSet(metadata=dataset_metadata, dataframe=dataframe)
      
     
if __name__ == "__main__":
    # MetaData test
    metadata = MetaData()
    serialized = json.dumps(metadata, cls=MyJSONEncoder)
    deserialized = json.loads(serialized, cls=MyJSONDecoder, classes=[MetaData])
    metadata.assert_eq(deserialized)

    print(serialized)

    # DataSet test
    from io import BytesIO

    dataframe = pl.DataFrame(
        {
            "integer": [4, 5, 6, None],
            "float": [4.0, None, nan, inf],
            "string": ["d", "e", "f", None],
        }
    )

    dataset = DataSet(metadata=metadata, dataframe=dataframe)
    with BytesIO() as f:
        dataset.write_parquet(f)
        deserialized_dataset = DataSet.read_parquet(f)

    dataset.assert_eq(deserialized_dataset)
    print(deserialized_dataset.metadata)
    print(deserialized_dataset.dataframe)

    # {"__python__/dataclasses/MetaData": {"foo": "Foo", "bar": Infinity, "baz": null}}
    # MetaData(foo='Foo', bar=inf, baz=None)
    # shape: (4, 3)
    # ┌─────────┬───────┬────────┐
    # │ integer ┆ float ┆ string │
    # │ ---     ┆ ---   ┆ ---    │
    # │ i64     ┆ f64   ┆ str    │
    # ╞═════════╪═══════╪════════╡
    # │ 4       ┆ 4.0   ┆ d      │
    # │ 5       ┆ null  ┆ e      │
    # │ 6       ┆ NaN   ┆ f      │
    # │ null    ┆ inf   ┆ null   │
    # └─────────┴───────┴────────┘