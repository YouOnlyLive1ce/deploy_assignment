import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def process_trades_in_chunks(input_file, output_file_train, output_file_test, chunksize=100000):
    # Open the output files
    with open(output_file_train, 'w', newline='') as outfile_train, open(output_file_test, 'w', newline='') as outfile_test:
        for chunk in pd.read_csv(input_file, chunksize=chunksize, header=None):
            # Assuming the following index mapping based on Binance aggTrades CSV:
            # Column indices:
            # 0 - trade_id
            # 1 - price
            # 2 - qty
            # 3 - first_trade_id
            # 4 - last_trade_id
            # 5 - timestamp
            # 6 - isBuyerMaker
            # 7 - isBestMatch

            # Round 'price' to the nearest 100 using .mod() for better precision
            chunk[1] = chunk[1].round(-2)

            # Calculate percent till nearest 1000
            chunk['percent_to_1000'] = (chunk[1]/1000).round(-4)

            # Add a column for the number of aggregated trades
            chunk['aggregated_trades'] = chunk[4] - chunk[3] + 1

            # Add a column to check if the price has appeared before (sequential)
            chunk['price_seen_before'] = chunk[1].duplicated(keep='first')

            # Drop unwanted columns: 0 (trade_id), 3 (first_trade_id), 4 (last_trade_id), 5 (timestamp), 6 (isBuyerMaker), 7 (isBestMatch)
            chunk = chunk.drop(columns=[0, 3, 4, 5])

            chunk[2]=chunk[2].round(3)

            # Delete 90% of rows where the quantity (column 2) is <0.0005
            zero_quantity_rows = chunk[chunk[2] < 0.001]
            chunk = chunk.drop(zero_quantity_rows.sample(frac=0.99).index)

            middle_quantity_rows = chunk[chunk[2] < 0.5]
            chunk = chunk.drop(middle_quantity_rows.sample(frac=0.99).index)
            
            # Normalize the data
            chunk.columns = chunk.columns.astype(str)
            scaler = StandardScaler()
            chunk[['1','2']] = scaler.fit_transform(chunk[['1','2']]).astype('float32')
            chunk['1']=chunk['1'].round(3).astype('float32')
            chunk['2']=chunk['2'].round(3).astype('float32')
            chunk['6']=chunk['6'].astype('int32')
            chunk['7']=chunk['7'].astype('int32')
            chunk['percent_to_1000']=chunk['percent_to_1000'].astype('int32')
            chunk['price_seen_before'] = chunk['price_seen_before'].astype('int32')

            train, test = train_test_split(chunk, test_size=0.2)
            
            # Write train and test chunks to respective output files
            train_df = pd.DataFrame(train)  # Convert numpy array back to DataFrame
            test_df = pd.DataFrame(test)

            train_df.to_csv(outfile_train, header=outfile_train.tell() == 0, index=False)
            test_df.to_csv(outfile_test, header=outfile_test.tell() == 0, index=False)

def main():
    input_folder = 'data/raw/'
    input_files = glob.glob(os.path.join(input_folder, '*aggTrades*'))

    output_folder = 'data/processed'
    os.makedirs(output_folder, exist_ok=True)

    for input_file in input_files:
        filename = os.path.basename(input_file)
        output_file_train = os.path.join(output_folder, f"train_{filename}")
        output_file_test = os.path.join(output_folder, f"test_{filename}")

        print(f"Processing file: {input_file}")
        process_trades_in_chunks(input_file, output_file_train, output_file_test)

    print("All files processed.")

if __name__ == "__main__":
    main()

# TODO: transformer scaler to fix curve ditribution