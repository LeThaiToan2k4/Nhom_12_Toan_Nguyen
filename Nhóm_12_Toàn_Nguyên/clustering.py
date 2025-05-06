import pandas as pd
from sklearn.cluster import Birch
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def perform_clustering(file_path, n_clusters=None):
    # Load data from CSV
    data = pd.read_csv(file_path)

    # Lọc các cột cần thiết cho phân cụm
    X = data[['Age', 'AnnualIncome', 'SpendingScore']].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Áp dụng thuật toán BIRCH
    birch = Birch(n_clusters=n_clusters, threshold=0.5)
    birch.fit(X)

    # Thêm nhãn phân cụm vào dữ liệu
    data['Cluster'] = birch.labels_

    # Tính số lượng khách hàng trong mỗi nhóm
    cluster_results = data.groupby('Cluster').size().to_dict()

    # Tạo đồ thị phân cụm
    plot_path = 'static/images/clustering_result.png' 
    if not os.path.exists(os.path.dirname(plot_path)):
        os.makedirs(os.path.dirname(plot_path))  

    plt.figure(figsize=(8, 6))
    
    # Sử dụng các màu sắc rõ rệt cho các nhóm
    cluster_colors = ['red', 'blue', 'yellow', 'green', 'purple', 'orange', 'cyan', 'pink']

    # Đảm bảo số màu đủ cho tất cả các nhóm
    unique_clusters = np.unique(birch.labels_)
    colors = [cluster_colors[i % len(cluster_colors)] for i in range(len(unique_clusters))]

    # Vẽ đồ thị phân cụm với màu sắc rõ rệt cho từng nhóm
    for cluster, color in zip(unique_clusters, colors):
        cluster_data = data[data['Cluster'] == cluster]
        plt.scatter(cluster_data['Age'], cluster_data['SpendingScore'], label=f'Cluster {cluster}', c=color, edgecolor='black')

    plt.xlabel('Age')
    plt.ylabel('Spending Score')
    plt.title('Kết quả phân cụm khách hàng (BIRCH)')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()

    return cluster_results, plot_path
