import os
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

def process_trades_in_chunks(input_file, output_file_train, output_file_test, chunksize=100000):
    for chunk in pd.read_csv(input_file, chunksize=chunksize, header=None):
        # Remove anti-outliers, since lots of repeated rows and noise
        # Delete 90% of rows where the quantity (column 2) is <0.0005
        zero_quantity_rows = chunk[chunk[2] < 0.001]
        chunk = chunk.drop(zero_quantity_rows.sample(frac=0.95).index)
        middle_quantity_rows = chunk[chunk[2] < 0.1]
        chunk = chunk.drop(middle_quantity_rows.sample(frac=0.50).index)
        
        # Quantization
        chunk.columns = chunk.columns.astype(str)
        
        chunk = chunk.rename(columns={
            '0': 'trade_id',
            '1': 'price',
            '2': 'qty',
            '3': 'first_trade_id',
            '4': 'last_trade_id',
            '5': 'event_timestamp',
            '6': 'isBuyerMaker',
            '7': 'isBestMatch'
        })
        train, test = train_test_split(chunk, test_size=0.2, shuffle=True) # Dataset is highly inbalanced
        
        # Write train and test chunks to respective output files
        train_df = pd.DataFrame(train)
        test_df = pd.DataFrame(test)
        train_df['event_timestamp'] = pd.to_datetime(train_df['event_timestamp'], unit='ms')
        test_df['event_timestamp'] = pd.to_datetime(test_df['event_timestamp'], unit='ms')
        # Save to .parquet
        # # # Dataframes do not contain class currently
        pq.write_table(pa.Table.from_pandas(train_df), f"{output_file_train}.parquet")
        pq.write_table(pa.Table.from_pandas(test_df), f"{output_file_test}.parquet")

def main():
    # TODO: hydra
    input_folder = 'data/raw/'
    input_files = glob.glob(os.path.join(input_folder, '*aggTrades*'))

    output_folder = 'data/processed'
    os.makedirs(output_folder, exist_ok=True)

    for input_file in input_files:
        filename = os.path.basename(input_file)
        
        # Remove the '.csv' extension from the filename
        filename_without_csv = filename.replace('.csv', '')
        
        output_file_train = os.path.join(output_folder, f"train_{filename_without_csv}")
        output_file_test = os.path.join(output_folder, f"test_{filename_without_csv}")
        
        print(f"Processing file: {input_file}")
        process_trades_in_chunks(input_file, output_file_train, output_file_test)

    print("csv files converted to parquet with custom features")

if __name__ == "__main__":
    main()