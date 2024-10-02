import os
import glob
from feast import (
    Entity, 
    FeatureView, 
    FileSource, 
    ValueType,
    FeatureStore,
    Field
)
from feast.types import Int32, Float32, Bool, UnixTimestamp
import pandas as pd

def preprocess_data(table):
    table['price'] = table['price'].round(-1)
    table['percent_to_1000'] = 1000 / ((table['price'] % 1000) + 1).round(2)
    table['aggregated_trades'] = table['last_trade_id'] - table['first_trade_id'] + 1
    table['price_seen_before'] = table['price'].duplicated(keep='first')
    table = table.drop(columns=['first_trade_id', 'last_trade_id'])
    table['qty'] = table['qty'].round(3)
    return table

entity_df = Entity(
    name="trade_id",
    value_type=ValueType.INT32,
    description="Identifier for trades"
)

def process_all_files():
    input_folder = './data/processed/'
    input_files = glob.glob(os.path.join(input_folder, '*.parquet'))
    
    repo_path = './services/feast/feast_fs/feature_repo'
    store = FeatureStore(repo_path=repo_path)
    
    for file_path in input_files:
        df = pd.read_parquet(file_path)
        processed_df = preprocess_data(df)

        # Save the processed data with '_custom_features' appended to the filename
        processed_data_path = file_path.replace('.parquet', '_custom_features.parquet')
        processed_df.to_parquet(processed_data_path)
        print(processed_df.columns, processed_df.dtypes)

        sanitized_file_name = os.path.basename(file_path).replace('.parquet', '').replace('/', '_').replace('-', '_')
        source = FileSource(
            name=f"binanceAggTrades_{sanitized_file_name}",
            path=processed_data_path,
            timestamp_field="event_timestamp",
        )
        
        view = FeatureView(
            name=f"binanceAggTradesCustomFeatures_{sanitized_file_name}",
            entities=[entity_df],
            source=source,
            online=False,
            schema=[
                Field(name="price", dtype=Float32),
                Field(name="qty", dtype=Float32),
                Field(name="percent_to_1000", dtype=Float32),
                Field(name="aggregated_trades", dtype=Int32),
                Field(name="price_seen_before", dtype=Bool),
                Field(name="event_timestamp", dtype=UnixTimestamp),
                Field(name="isBuyerMaker", dtype=Bool),  
                Field(name="isBestMatch", dtype=Bool),
            ],
        )
        
        # Apply the entity and feature view to the feature store
        store.apply([entity_df])
        store.apply([view])
    
    print("All files processed and feature views applied.")

if __name__ == "__main__":
    process_all_files()
