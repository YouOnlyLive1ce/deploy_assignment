import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

def get_processed_data_path():
    data_path = './data/processed/train_BTCUSDT-aggTrades-2024-09-16_custom_features.parquet'
    return data_path

def read_processed_parquet():
    train_file_path = get_processed_data_path()
    table = pq.ParquetDataset(train_file_path).read()
    df = table.to_pandas()

    # Feature engineering on the timestamp
    df['year'] = df['event_timestamp'].dt.year
    df['month'] = df['event_timestamp'].dt.month
    df['day'] = df['event_timestamp'].dt.day
    df['hour'] = df['event_timestamp'].dt.hour
    df['minute'] = df['event_timestamp'].dt.minute
    df['second'] = df['event_timestamp'].dt.second

    df = df.drop(columns=['event_timestamp'])

    print(df.head)
    return df

df=read_processed_parquet()

# Perform Hierarchical Clustering
plt.figure(figsize=(10, 7))
method='single'
Z = sch.linkage(df, method=method)


dendrogram = sch.dendrogram(Z)
distance_threshold=300
plt.axhline(y=distance_threshold, color='r', linestyle='--')

plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()

hc = AgglomerativeClustering(distance_threshold=None, n_clusters=9, linkage=method) #distance_threshold=distance_threshold,
data_train_labels=hc.fit_predict(df)
df['class']=data_train_labels.tolist()
print(df['class'].value_counts())
df.to_parquet(get_processed_data_path())
