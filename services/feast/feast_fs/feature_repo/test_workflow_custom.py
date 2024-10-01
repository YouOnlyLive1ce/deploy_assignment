from feast import FeatureStore
import hydra
from omegaconf import DictConfig
import pandas as pd
import re

# Function to get the registry path
@hydra.main(version_base=None, config_path='configs', config_name='config')
def get_registry_path(cfg: DictConfig):
    return './' + cfg['feast_registry_path']

# Function to get the feature repository path
def get_repo_path():
    return './services/feast/feast_fs/feature_repo'

# Function to get the data path from the config
@hydra.main(version_base=None, config_path='configs', config_name='config')
def get_data_path(cfg: DictConfig):
    data_path = './data/processed/' + cfg.parquet_file_name+'_custom_features.parquet'
    return data_path

# Function to load the 'event_timestamp' column from the parquet file
def load_event_timestamps(data_path):
    parquet_df = pd.read_parquet(data_path)
    return parquet_df['event_timestamp']

# Function to load the 'trade_id' column from the parquet file
def load_ids(data_path):
    parquet_df = pd.read_parquet(data_path)
    return parquet_df['trade_id']

# Main function to retrieve historical features
@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg: DictConfig):
    # Get the feature store repository path
    repo_path = get_repo_path()

    # Initialize the feature store
    store = FeatureStore(repo_path=repo_path)

    # Get the data path from the config
    data_path = get_data_path(cfg)

    # Load event timestamps and trade IDs from the parquet file
    event_timestamps = load_event_timestamps(data_path)
    trade_ids = load_ids(data_path)

    # Create a dataframe containing entity data (trade_id and event_timestamp)
    entity_df = pd.DataFrame({
        "trade_id": trade_ids,
        "event_timestamp": event_timestamps
    })

    # Define the features to retrieve, including the custom ones
    features = [
        "processed_binanceAggTrades:price",
        "processed_binanceAggTrades:percent_to_1000",
        "processed_binanceAggTrades:aggregated_trades",
        "processed_binanceAggTrades:price_seen_before",
        "processed_binanceAggTrades:event_timestamp"
    ]

    # Retrieve historical features
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=features
    ).to_df()

    # Print the feature schema
    print("----- Feature schema -----\n")
    print(training_df.info())

    print()

    # Print example features
    print("----- Example features -----\n")
    print(training_df.head())

if __name__ == "__main__":
    main()