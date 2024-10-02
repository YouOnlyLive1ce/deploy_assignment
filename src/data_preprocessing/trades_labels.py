from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
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

def create_labels(df,distance_threshold=400, n_clusters=None, method='single'):
    if distance_threshold:
        hc=pickle.load(open(f"./models/hc_model{distance_threshold}-{n_clusters}-{method}.pkl", "rb"))
    elif n_clusters:
        hc=pickle.load(open(f"./models/hc_model{distance_threshold}-{n_clusters}-{method}.pkl", "rb"))

    data_train_labels = hc.fit_predict(df)
    df['class'] = data_train_labels.tolist()
    return df

def train_and_save_hc(df, distance_threshold=400, n_clusters=None, method='single'):
    plt.figure(figsize=(10, 7))
    Z = sch.linkage(df, method=method)
    dendrogram = sch.dendrogram(Z)
    
    # Plot the distance threshold line
    # plt.axhline(y=distance_threshold, color='r', linestyle='--')
    # plt.title('Dendrogram for Hierarchical Clustering')
    # plt.xlabel('Data Points')
    # plt.ylabel('Euclidean Distance')
    # plt.show()

    hc = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=n_clusters, linkage=method)
    hc.fit(df)

    if distance_threshold:
        pickle.dump(hc, open(f"./models/hc_model{distance_threshold}-{n_clusters}-{method}.pkl", "wb"))
    elif n_clusters:
        pickle.dump(hc, open(f"./models/hc_model{distance_threshold}-{n_clusters}-{method}.pkl", "wb"))
    return hc

if __name__ == "__main__":
    input_folder = 'data/processed/'
    input_files = glob.glob(os.path.join(input_folder, '*aggTrades*'))
    print("processing datetime and adding labels to files", input_files)

    for file_path in input_files:
        df = pd.read_parquet(file_path)
        df = convert_timestamp(df)
        
        hc_model = train_and_save_hc(df, distance_threshold=400,n_clusters=None)
        df_with_labels = create_labels(df, 400, None, 'single')
        
        # Save the labeled dataframe back to a parquet file
        output_path = file_path.replace('.', '_labeled.')
        df_with_labels.to_parquet(file_path)
        print(f"Processed and saved: {file_path}")