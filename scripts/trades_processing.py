import os
import glob
import pandas as pd

def process_trades_in_chunks(input_file, output_file, chunksize=100000):
    # Open the output file in write mode
    with open(output_file, 'w', newline='') as outfile:
        # Process the CSV in chunks
        for chunk in pd.read_csv(input_file, chunksize=chunksize, header=None):
            # Assuming the following index mapping based on the CSV structure:
            # Column indices: 
            # 0 - trade_id (if exists)
            # 1 - price
            # 2 - qty
            # 3 - quote_qty
            # 4 - timestamp
            # 5 - bool_1
            # 6 - bool_2

            # Check if 'trade_id' is the first column (index 0) and drop it
            if chunk.shape[1] > 0:  # Check if the chunk has any columns
                chunk = chunk.drop(columns=[0])  # Drop the first column (trade_id)

            # Filter where 'quote_qty' is in the column at index 3 (4th column)
            chunk = chunk[chunk[3] > 500]

            # Round 'price' to nearest 10 (1st column)
            chunk[1] = chunk[1].round(-1)

            # Write chunk to the output file
            chunk.to_csv(outfile, header=outfile.tell() == 0, index=False)

def main():
    # Define the input folder path
    input_folder = 'data/raw/'
    # Use glob to find all files in the input folder that have 'trades' in their name
    input_files = glob.glob(os.path.join(input_folder, '*trades*'))

    # Output folder
    output_folder = 'outputs/processed/'

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through each file and process it
    for input_file in input_files:
        # Extract the filename without the path
        filename = os.path.basename(input_file)

        # Construct the output file path (same name but in the processed folder)
        output_file = os.path.join(output_folder, filename)

        # Process each file
        print(f"Processing file: {input_file}")
        process_trades_in_chunks(input_file, output_file)

    print("All files processed.")

# Call the main function if this script is executed
if __name__ == "__main__":
    main()
