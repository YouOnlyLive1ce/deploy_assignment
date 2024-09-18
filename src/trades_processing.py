import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

def process_trades_in_chunks(input_file, output_file_train, output_file_test, chunksize=100000):
    # Open the output files
    with open(output_file_train, 'w', newline='') as outfile_train, open(output_file_test, 'w', newline='') as outfile_test:
        for chunk in pd.read_csv(input_file, chunksize=chunksize, header=None):
            # Assuming the following index mapping based on the CSV structure:
            # Column indices:
            # 0 - trade_id (if exists)
            # 1 - price
            # 2 - qty
            # 3 - quote_qty
            # 4 - timestamp
            # 5 - isBuyerMaker
            # 6 - isBestMatch
                        
            if chunk.shape[1] > 0:  # Check if the chunk has any columns
                chunk = chunk.drop(columns=[0])  # Drop the first column (trade_id)

                chunk[2] = chunk[2].round(3)  # Round quantity
                chunk[3] = chunk[3].round(0)  # Round quote_qty

            # Round the price to the nearest 10 and convert it to an integer
            chunk[1] = chunk[1].round(-1).astype(int)

            # Create bins for quote_qty
            bins = [0, 50, 250, 500, 1000, 3000, 200000]
            labels = ['0-50', '50-250', '250-500', '500-1000', '1000-3000', '3000-200000']
            chunk['quote_qty_bins'] = pd.cut(chunk[3], bins=bins, labels=labels, include_lowest=True)
            chunk=chunk.drop(columns=[3,4,6])

            # Split into train/test (90/10 split)
            train, test = train_test_split(chunk, test_size=0.1)

            # Write train and test chunks to respective output files
            train.to_csv(outfile_train, header=outfile_train.tell() == 0, index=False)
            test.to_csv(outfile_test, header=outfile_test.tell() == 0, index=False)

def main():
    input_folder = 'data/raw/'
    input_files = glob.glob(os.path.join(input_folder, '*trades*'))

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
