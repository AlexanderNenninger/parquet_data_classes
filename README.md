# Map Custom Datasets to Parquet and Back

Currently there seems to be no elegant solution for serializing custom metadata in dataframe serialization formats that play nicely with the Python type system, data warehouse engines like Apache Spark, Hive, etc. and dataframe libraries like Pandas, polars, and apache Arrow.

A concrete use case for this could be storing unit information, frequency data, orientation, etc. along with timeseries. This can of course be stored in aditional metdadata files, but I found this to be error-prone and messy.

Thankfully the metadata of Apache Parquet can contain arbitrary key-value pairs ([source](https://parquet.apache.org/docs/file-format/metadata/)). The technique demonstrated in this repo is as as follows:

 1. Store all metadata along with the dataframe in a container object. This can be a dataclass, or we inherit from a metaclass defining an interface for metadata and tabular data, and that implements the serialization functionality.

 2. Serialize metadata as JSON and encode the resulting string using UTF-8. Using JSON here instead of e.g. pickle enables easy deserialization within other environments, since both JSON and UTF-8 are widely supported across programming languages. There are pitfalls w.r.t. numeric datatypes, so I might iterare in this. 

 3. Write the resulting bitstring into the key-value metadata under a user-defined key.

 4. During deserialization, read the value stored in the user-defined key. Construct Python objects (here demonstrated using dataclasses) from the binary string.

 5. Instantiate the container object, dataframe, and metadata.

## Usage

This repo is only intended to be a demonstrator, not as a library. Nontheless, to run a basic example, create a new environment from `environment.yml`. Then

```python
from io import BytesIO
from pathlib import Path

import polars as pl

from custom_dataset_to_parquet import DataSet, MetaData

# Implement your custom type for this
metadata = MetaData()

# Partitioning casts column to categorical.
# Notes:
#  - All categorical columns will be cast to `Categorical(ordering="physical")`
#  - Column order is not maintained if the dataset is partitoned. 
#  - Row order is not maintained if dataset is partioned. 
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

dataset.assert_eq(deserialized_dataset) # Check for equality. For additional arguments see `polars.testing.assert_dataframe_equals`
print("Dataset\n", dataset)
print("Deserialized dataset\n", deserialized_dataset)
```

This also works with partitions à la Apache Hive:

```python
import shutil

# Use try here to ensure cleanup. There's probably a library for this.
try:
    # Init output directory
    Path("./tmp").mkdir(exist_ok=True)
    
    # Write using "partion" as partion column
    dataset.write_parquet(
        "tmp/dataset",
        partition_cols=["partition"],
        existing_data_behavior="delete_matching",
    )

    # Read like normal
    partitioned_dataset = DataSet.read_parquet("tmp/dataset")

    # Column order has now changed. Also partitioned columns will be cast to `polars.Categorical(ordering="physical")`.
    dataset.assert_eq(partitioned_dataset, check_column_order=False)
    print(partitioned_dataset)

except Exception as e:
    raise e
finally:
    # Remove this if you want to inspect the data on disk.
    shutil.rmtree("./tmp", ignore_errors=True)
```

You can find addtional usage examples, tests of edge-cases and performance benchmark code in `./test_custom_dataset_to_parqet.py`.
