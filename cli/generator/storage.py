# Standard
import json

# Third Party
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def json_to_parquet(json_data, parquet_name):
    """
    Convert JSON data to Parquet format and save it to a Parquet file.

    Parameters:
    - json_data (list of dict): The JSON data to be converted.
    - parquet_name (str): The name for the output Parquet file. The '.parquet' extension will be added.

    Returns:
    - None
    """
    df = pd.DataFrame(json_data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_name + ".parquet")


def file_json_to_parquet(filename, parquet_name):
    """
    Convert JSON data from a local file to Parquet format and save it to a Parquet file.

    Parameters:
    - filename (str): Path to the JSON data.
    - parquet_name (str): The name for the output Parquet file. The '.parquet' extension will be added.

    Returns:
    - None
    """
    with open(filename, "r", encoding="utf-8") as data_file:
        file_data = data_file.read()

    json_data = json.loads(file_data)
    json_to_parquet(json_data, parquet_name)
