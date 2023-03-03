from distributed import Client, LocalCluster

import dask.dataframe as dd
import lightgbm
import numpy as np

if __name__ == '__main__':
    # Start a Dask Distributed cluster on the local machine
    cluster = LocalCluster(threads_per_worker=1)
    client = Client(cluster)
    print(cluster)


    # Load the data (lazily)
    ddf = dd.read_parquet("data/yellow_tripdata_2021-01-*.parquet")


    # Remove basic outliers from the training data (some but minimal communication)
    cap_fare = ddf["fare_amount"].mean() + 3 * ddf["fare_amount"].std()
    cap_distance = ddf["trip_distance"].mean() + 3 * ddf["trip_distance"].std()
    # The above were only expression definitions. Compute the values.
    cap_fare, cap_distance = client.compute([cap_fare, cap_distance], sync=True)
    ddf = ddf.query(
        f"trip_distance > 0 and trip_distance < {cap_distance} and fare_amount > 0 and fare_amount < {cap_fare}"
    )


    # Feature Engineering (embarrassingly parallel)
    def split_pickuptime(df):
        return df.assign(
            pickup_dayofweek=df["tpep_pickup_datetime"].dt.dayofweek,
            pickup_hour=df["tpep_pickup_datetime"].dt.hour,
            pickup_minute=df["tpep_pickup_datetime"].dt.minute,
        )


    X = ddf.map_partitions(split_pickuptime).drop(
        columns=["tpep_pickup_datetime", "tpep_dropoff_datetime", "store_and_fwd_flag"]
    )
    y = X.pop("fare_amount")


    # Train an ML model by passing the data from Dask to LightGBM
    regressor = lightgbm.DaskLGBMRegressor(objective="regression_l1")
    regressor.fit(X, y)
