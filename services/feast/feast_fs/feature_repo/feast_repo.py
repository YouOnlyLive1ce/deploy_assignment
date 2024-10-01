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

@hydra.main(version_base=None, config_path='configs', config_name='config')
def get_data_path(cfg: DictConfig):
    data_path = './data/processed/' + cfg.parquet_file_name+'.parquet'
    return data_path

# Define the entity
entity_df = Entity(
    name="trade_id",
    value_type=ValueType.INT32,
    description="Identifier"
)

# Define the file source
@hydra.main(version_base=None, config_path='configs', config_name='config')
def define_source(cfg: DictConfig):
    data_path = get_data_path(cfg)

    source = FileSource(
        name="binanceAggTrades",
        path=data_path,
        timestamp_field="event_timestamp",  # Map the 'timestamp' field to 'event_timestamp'
        event_timestamp_column="event_timestamp",  # Alias for event_timestamp
    )
    return source

# Define the feature view
@hydra.main(version_base=None, config_path='configs', config_name='config')
def define_feature_view(cfg: DictConfig):
    source = define_source(cfg)
    view = FeatureView(
        name="binanceAggTrades",
        entities=[entity_df],
        source=source,
        online=False,
        schema=[
            Field(name="price", dtype=Float32),
            Field(name="qty", dtype=Float32),
            Field(name="first_trade_id", dtype=Int32),
            Field(name="last_trade_id", dtype=Int32),
            Field(name="event_timestamp", dtype=UnixTimestamp),
            Field(name="isBuyerMaker", dtype=Bool),  
            Field(name="isBestMatch", dtype=Bool),
        ],
        # TODO: BigQuery
    )
    return view

# Apply the feature view to the feature store
@hydra.main(version_base=None, config_path='configs', config_name='config')
def apply_feature_view(cfg: DictConfig):
    repo_path = './services/feast/feast_fs/feature_repo'
    store = FeatureStore(repo_path=repo_path)
    store.apply([entity_df])
    view = define_feature_view(cfg)
    store.apply([view])
    registered_view = store.get_feature_view("binanceAggTrades")
    print(registered_view)

# Example usage
if __name__ == "__main__":
    apply_feature_view()