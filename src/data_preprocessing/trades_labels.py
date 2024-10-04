from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
import glob

def convert_timestamp(df):
    # Feature engineering on the timestamp
    df['year'] = df['event_timestamp'].dt.year
    df['month'] = df['event_timestamp'].dt.month
    df['day'] = df['event_timestamp'].dt.day
    df['hour'] = df['event_timestamp'].dt.hour
    df['minute'] = df['event_timestamp'].dt.minute
    df['second'] = df['event_timestamp'].dt.second
    return df

def train_and_save_hc(df, distance_threshold=100, n_clusters=None, method='ward'):
    plt.figure(figsize=(10, 7))
    Z = sch.linkage(df, method=method)
    dendrogram = sch.dendrogram(Z)
    
    # Plot the distance threshold line
    # plt.axhline(y=distance_threshold, color='r', linestyle='--')
    plt.title('Dendrogram for Hierarchical Clustering')
    plt.xlabel('Data Points')
    plt.ylabel('Euclidean Distance')
    plt.show()

    hc = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=n_clusters, linkage=method)
    hc.fit(df)

    # Save the hierarchical clustering model
    pickle.dump(hc, open(f"./models/hc_model_{distance_threshold}_{n_clusters}_{method}.pkl", "wb"))
    return hc

def create_labels(df, hc):
    # Assign the predicted cluster labels to the DataFrame
    data_train_labels = hc.fit_predict(df)
    df['class'] = data_train_labels.tolist()
    return df

if __name__ == "__main__":
    input_folder = 'data/processed/'
    input_files = glob.glob(os.path.join(input_folder, '*aggTrades*_custom_features.parquet'))
    print("Processing datetime and concatenating data from files", input_files)

    # List to store DataFrames
    dfs = []

    # Load and preprocess each file
    for file_path in input_files:
        df = pd.read_parquet(file_path)
        # df = convert_timestamp(df)
        df = df.drop(['event_timestamp','trade_id'], axis=1)
        dfs.append(df)
    
    # Concatenate all dataframes into one
    full_df = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"Concatenated DataFrame shape: {full_df.shape}")

    # Perform clustering on the concatenated dataframe
    hc_model = train_and_save_hc(full_df, distance_threshold=None, n_clusters=7)

    # Predict and assign labels to the concatenated DataFrame
    full_df_with_labels = create_labels(full_df, hc_model)

    # Split the labeled concatenated DataFrame back to individual files
    start_idx = 0
    for file_path, df in zip(input_files, dfs):
        end_idx = start_idx + len(df)
        df_with_labels = full_df_with_labels.iloc[start_idx:end_idx]
        print(df_with_labels['class'].value_counts())
        
        # Save the labeled dataframe back to a parquet file
        output_path = file_path.replace('.', '_labeled.')
        df_with_labels.to_parquet(output_path)
        print(f"Processed and saved: {output_path}")

        # Update start index for the next split
        start_idx = end_idx
