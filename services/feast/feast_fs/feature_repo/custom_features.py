from feast import (
    Entity, 
    FeatureView, 
    FileSource, 
    ValueType,
    FeatureStore,
    Field
)
from feast.types import Int32, Float32, Bool, UnixTimestamp
import hydra
from omegaconf import DictConfig
import pandas as pd

@hydra.main(version_base=None, config_path='conf', config_name='config')
def get_data_path(cfg: DictConfig):
    data_path = './data/processed/' + cfg.parquet_file_name+'.parquet'
    return data_path

@hydra.main(version_base=None, config_path='conf', config_name='config')
def get_processed_data_path(cfg: DictConfig):
    data_path = './data/processed/' + cfg.parquet_file_name+'_custom_features.parquet'
    return data_path

def preprocess_data(table):
    table['price'] = table['price'].round(-1)
    table['percent_to_1000'] = (table['price'] / 1000).round(2)
    table['aggregated_trades'] = table['last_trade_id'] - table['first_trade_id'] + 1
    table['price_seen_before'] = table['price'].duplicated(keep='first')
    table = table.drop(columns=['first_trade_id', 'last_trade_id'])
    table['qty'] = table['qty'].round(3)
    return table

# Define the entity
entity_df = Entity(
    name="trade_id",
    value_type=ValueType.INT32,
    description="Identifier for trades"
)

# Function to define the file source for Feast after preprocessing
@hydra.main(version_base=None, config_path='conf', config_name='config')
def define_source(cfg: DictConfig):
    # Get the data path from the config
    data_path = get_data_path(cfg)
    
    # Read the data and preprocess it
    df = pd.read_parquet(data_path)
    processed_df = preprocess_data(df)
    
    # Save the processed data to a new parquet file
    processed_data_path = get_processed_data_path(cfg)
    processed_df.to_parquet(processed_data_path)
    
    # Define the source from the processed data
    source = FileSource(
        name="processed_binanceAggTrades",
        path=processed_data_path,
        timestamp_field="event_timestamp",
        event_timestamp_column="event_timestamp",
    )
    return source

# Define the feature view with custom features
@hydra.main(version_base=None, config_path='conf', config_name='config')
def define_feature_view(cfg: DictConfig):
    source = define_source(cfg)
    view = FeatureView(
        name="processed_binanceAggTrades",
        entities=[entity_df],
        source=source,
        online=False,
        schema=[
            Field(name="price", dtype=Float32),
            Field(name="percent_to_1000", dtype=Float32),
            Field(name="aggregated_trades", dtype=Int32),
            Field(name="price_seen_before", dtype=Bool),
            Field(name="event_timestamp", dtype=UnixTimestamp),
            Field(name="isBuyerMaker", dtype=Bool),  
            Field(name="isBestMatch", dtype=Bool),
        ],
        # Add any other properties here
    )
    return view

# Apply the custom feature view to the feature store
@hydra.main(version_base=None, config_path='conf', config_name='config')
def apply_custom_feature_view(cfg: DictConfig):
    repo_path = './services/feast/feast_fs/feature_repo'
    store = FeatureStore(repo_path=repo_path)
    
    # Apply the entity definition
    store.apply([entity_df])
    
    # Apply the custom feature view
    view = define_feature_view(cfg)
    store.apply([view])
    
    # Retrieve and print the registered feature view
    registered_view = store.get_feature_view("processed_binanceAggTrades")
    print(registered_view)

# Example usage
if __name__ == "__main__":
    apply_custom_feature_view()
