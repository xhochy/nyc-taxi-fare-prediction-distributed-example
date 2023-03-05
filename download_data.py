from pathlib import Path

import requests
import polars as pl


# Download a single month of data
r = requests.get(
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-01.parquet",
    allow_redirects=True,
)
Path("data").mkdir(exist_ok=True)
Path("data/yellow_tripdata_2021-01.parquet").write_bytes(r.content)

# Partiton by day of month
df = pl.read_parquet("data/yellow_tripdata_2021-01.parquet")
grouping = df.with_columns(day=pl.col("tpep_pickup_datetime").dt.day()).groupby("day")
for day, group in grouping:
    group.drop("day").write_parquet(f"data/yellow_tripdata_2021-01-{day}.parquet")
