from feast import FeatureStore
import hydra
from omegaconf import DictConfig
import pandas as pd
import re

# Function to get the registry path
@hydra.main(version_base=None, config_path='conf', config_name='config')
def get_registry_path(cfg: DictConfig):
    return './' + cfg['feast_registry_path']

# Function to get the data path
def get_repo_path():
    return './services/feast/feast_fs/feature_repo'

@hydra.main(version_base=None, config_path='conf', config_name='config')
def get_data_path(cfg: DictConfig):
    data_path = './data/processed/' + cfg.parquet_file
    
    # Define the regex pattern to match the filename
    file_pattern = r'test_BTCUSDT-aggTrades-(\d{4})-(\d{2})-(\d{2})\.parquet'
    
    match = re.search(file_pattern, cfg.parquet_file)
    if match:
        year = match.group(1)
        month = match.group(2)
        day = match.group(3)

    return data_path

def load_event_timestamps(data_path):
    parquet_df = pd.read_parquet(data_path)
    return parquet_df['event_timestamp']

def load_ids(data_path):
    parquet_df = pd.read_parquet(data_path)
    return parquet_df['trade_id']

# Main function to get historical features
@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig):
    # Get the registry path
    repo_path = get_repo_path()

    # Initialize the feature store
    store = FeatureStore(repo_path=repo_path)

    # Get the data path
    data_path = get_data_path(cfg)

    # Load event timestamps from the parquet file
    event_timestamps = load_event_timestamps(data_path)

    trade_ids = load_ids(data_path)

    # Define the entity
    entity_df = pd.DataFrame({
        "trade_id": trade_ids,
        "event_timestamp": event_timestamps
    })

    features = [            
        "binanceAggTrades:price",
        "binanceAggTrades:qty",
        "binanceAggTrades:first_trade_id",
        "binanceAggTrades:last_trade_id",
        "binanceAggTrades:event_timestamp",
        "binanceAggTrades:isBuyerMaker",
        "binanceAggTrades:isBestMatch",
    ]

    # Get historical features
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