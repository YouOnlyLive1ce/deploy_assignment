from feast import FeatureStore
import pandas as pd
import os

def get_data_paths():
    data_dir = './data/processed/'
    files = [f for f in os.listdir(data_dir) if f.endswith('_custom_features.parquet')]
    return [os.path.join(data_dir, f) for f in files]

def load_event_timestamps(data_path):
    parquet_df = pd.read_parquet(data_path)
    return parquet_df['event_timestamp'].astype('datetime64[ns]')

def load_ids(data_path):
    parquet_df = pd.read_parquet(data_path)
    return parquet_df['trade_id']

def main():
    repo_path = './services/feast/feast_fs/feature_repo'
    store = FeatureStore(repo_path=repo_path)

    data_paths = get_data_paths()
    
    for data_path in data_paths:
        event_timestamps = load_event_timestamps(data_path)
        trade_ids = load_ids(data_path)

        entity_df = pd.DataFrame({
            "trade_id": trade_ids,
            "event_timestamp": event_timestamps
        })

        # Extract the sanitized file name to match the feature view names
        sanitized_file_name = os.path.basename(data_path).replace('.parquet', '').replace('_custom_features','').replace('/', '_').replace('-', '_')

        # Define the features to retrieve, including the custom ones
        features = [
            f"binanceAggTradesCustomFeatures_{sanitized_file_name}:price",
            f"binanceAggTradesCustomFeatures_{sanitized_file_name}:qty",
            f"binanceAggTradesCustomFeatures_{sanitized_file_name}:percent_to_1000",
            f"binanceAggTradesCustomFeatures_{sanitized_file_name}:aggregated_trades",
            f"binanceAggTradesCustomFeatures_{sanitized_file_name}:price_seen_before",
            f"binanceAggTradesCustomFeatures_{sanitized_file_name}:event_timestamp",
            f"binanceAggTradesCustomFeatures_{sanitized_file_name}:isBuyerMaker",
            f"binanceAggTradesCustomFeatures_{sanitized_file_name}:isBestMatch",
        ]
        
        training_df = store.get_historical_features(
            entity_df=entity_df,
            features=features
        ).to_df()

        print(f"----- Feature schema for {data_path} -----\n")
        print(training_df.info())

        print()

        print(f"----- Example features for {data_path} -----\n")
        print(training_df.head())

if __name__ == "__main__":
    main()
