"""
Ensemble Clustering - GIỐNG 100% VỚI NOTEBOOK
Load labels đã train sẵn từ 3 models và kết hợp
"""
import numpy as np
import os
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

def clusters_to_matrix(clusters):
    """
    Chuyển labels thành similarity matrix
    matrix[i,j] = 1 nếu sample i và j cùng cluster, 0 nếu khác
    """
    y_reshaped = np.expand_dims(clusters, axis=-1)
    matrix = (cdist(y_reshaped, y_reshaped, metric="cityblock") == 0).astype('int')
    return matrix

def normalize_matrix(matrix):
    """
    Chuẩn hóa matrix bằng cách chia cho đường chéo
    """
    return matrix / matrix.diagonal()

def combine_list_matrix(list_matrix):
    """
    Kết hợp nhiều matrix thành 1
    """
    matrix_size = len(list_matrix[0])
    result_matrix = np.zeros((matrix_size, matrix_size), dtype='float')
    
    for matrix in list_matrix:
        result_matrix += matrix
    
    return normalize_matrix(result_matrix)

def compute_matrix(list_clusters):
    """
    Tính similarity matrix từ list các labels
    """
    list_matrix = []
    for clusters in list_clusters:
        list_matrix.append(clusters_to_matrix(clusters))
    return combine_list_matrix(list_matrix)

def build_weighted_matrix(list_matrix, weights=None):
    """
    Kết hợp nhiều matrix với trọng số
    
    Parameters:
    - list_matrix: list các similarity matrix
    - weights: trọng số cho mỗi matrix (mặc định là đều nhau)
    """
    list_matrix_copy = []
    for matrix in list_matrix:
        matrix_copy = matrix.copy()
        # Thay 0 bằng epsilon nhỏ để tránh mất thông tin
        matrix_copy[matrix_copy == 0] = 1e-8
        list_matrix_copy.append(matrix_copy)
    
    # Áp dụng trọng số
    if weights is not None:
        for i in range(len(list_matrix)):
            list_matrix_copy[i] *= weights[i]
    
    return combine_list_matrix(list_matrix_copy)

def load_pretrained_labels():
    """
    Load labels đã train sẵn từ 3 models - GIỐNG NOTEBOOK
    
    Returns:
    - kmeans_list_labels: list labels từ KMeans ensemble
    - agg_list_labels: list labels từ Agglomerative
    - spectral_list_labels: list labels từ Spectral
    """
    # Paths - BẠN CẦN SỬA LẠI CHO ĐÚNG VỚI THỰC TẾ
    kmeans_path = "models/kmeans_ensemble_labels.npy"
    agg_path = "models/agg_ensemble_labels.npy"
    spectral_path = "models/spectral_ensemble_labels.npy"
    
    # Check if files exist
    if not os.path.exists(kmeans_path):
        raise FileNotFoundError(
            f"❌ Không tìm thấy file: {kmeans_path}\n"
            f"Bạn cần có 3 file labels đã train sẵn:\n"
            f"  1. {kmeans_path}\n"
            f"  2. {agg_path}\n"
            f"  3. {spectral_path}\n"
        )
    
    # Load labels - GIỐNG NOTEBOOK
    kmeans_list_labels = np.load(kmeans_path)
    agg_list_labels = [np.load(agg_path)]
    spectral_list_labels = [np.load(spectral_path)]
    
    print(f"✓ Loaded KMeans labels: {kmeans_list_labels.shape}")
    print(f"✓ Loaded Agglo labels: {len(agg_list_labels)} arrays")
    print(f"✓ Loaded Spectral labels: {len(spectral_list_labels)} arrays")
    
    return kmeans_list_labels, agg_list_labels, spectral_list_labels

def ensemble_predict(X_pca, n_clusters, weights=None):
    """
    Ensemble clustering chính - GIỐNG NOTEBOOK 100%
    
    Parameters:
    - X_pca: dữ liệu đã qua PCA (chỉ dùng để tính silhouette)
    - n_clusters: số cụm
    - weights: trọng số cho [kmeans, agglo, spectral]. Mặc định [1, 1, 1]
    
    Returns:
    - labels: nhãn cluster cuối cùng
    - ensemble_matrix: similarity matrix tổng hợp
    """
    if weights is None:
        weights = [1, 1, 1]
    
    print(f"\n🤖 Running ensemble clustering with k={n_clusters}")
    print(f"   Weights: KMeans={weights[0]}, Agglo={weights[1]}, Spectral={weights[2]}")
    
    # 1. LOAD LABELS ĐÃ TRAIN SẴN - GIỐNG NOTEBOOK
    kmeans_list_labels, agg_list_labels, spectral_list_labels = load_pretrained_labels()
    
    # 2. TẠO SIMILARITY MATRIX CHO TỪNG MODEL - GIỐNG NOTEBOOK
    print(f"   📊 Computing similarity matrices...")
    matrix_kmeans = compute_matrix(kmeans_list_labels)
    matrix_agg = compute_matrix(agg_list_labels)
    matrix_spectral = compute_matrix(spectral_list_labels)
    
    list_matrix = [matrix_kmeans, matrix_agg, matrix_spectral]
    
    print(f"   ✓ Matrix shapes: {matrix_kmeans.shape}")
    
    # 3. KẾT HỢP VỚI TRỌNG SỐ - GIỐNG NOTEBOOK
    print(f"   🔗 Building weighted ensemble matrix...")
    ensemble_matrix = build_weighted_matrix(list_matrix, weights)
    
    # 4. SPECTRAL CLUSTERING TRÊN ENSEMBLE MATRIX - GIỐNG NOTEBOOK
    print(f"   🎯 Running Spectral Clustering on ensemble matrix...")
    spectral_ensemble = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=42
    )
    labels = spectral_ensemble.fit_predict(ensemble_matrix)
    
    # 5. ĐÁNH GIÁ
    score = silhouette_score(X_pca, labels)
    print(f"   ✅ Silhouette score: {score:.4f}")
    
    return labels, ensemble_matrix

def ensemble_with_k(ensemble_matrix, k):
    """
    Chạy Spectral Clustering trên ensemble matrix với k cho trước
    GIỐNG NOTEBOOK
    
    Parameters:
    - ensemble_matrix: similarity matrix đã kết hợp
    - k: số cụm
    
    Returns:
    - labels: nhãn cluster
    """
    from sklearn.cluster import SpectralClustering
    
    spectral_ensemble = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        random_state=42
    )
    labels = spectral_ensemble.fit_predict(ensemble_matrix)
    
    return labels

def analyze_k_range(X_pca, k_range=range(2, 20), weights=None):
    """
    Phân tích với nhiều giá trị k - GIỐNG NOTEBOOK
    
    Returns:
    - results: dict chứa k, silhouette, labels
    """
    results = {
        'k': [],
        'silhouette': [],
        'labels': []
    }
    
    print(f"\n{'='*60}")
    print(f"🔍 Analyzing k from {min(k_range)} to {max(k_range)}...")
    print(f"{'='*60}")
    
    # Load labels 1 lần (không cần load lại mỗi iteration)
    kmeans_list_labels, agg_list_labels, spectral_list_labels = load_pretrained_labels()
    
    # Tạo similarity matrices
    matrix_kmeans = compute_matrix(kmeans_list_labels)
    matrix_agg = compute_matrix(agg_list_labels)
    matrix_spectral = compute_matrix(spectral_list_labels)
    
    list_matrix = [matrix_kmeans, matrix_agg, matrix_spectral]
    
    # Build ensemble matrix
    ensemble_matrix = build_weighted_matrix(list_matrix, weights)
    
    print(f"\n📊 Testing different k values...")
    for k in k_range:
        # Spectral clustering với k khác nhau
        spectral_ensemble = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            random_state=42
        )
        labels = spectral_ensemble.fit_predict(ensemble_matrix)
        score = silhouette_score(X_pca, labels)
        
        results['k'].append(k)
        results['silhouette'].append(score)
        results['labels'].append(labels)
        
        print(f"   k={k:2d} | silhouette={score:.4f}")
    
    # Tìm k tốt nhất
    best_idx = np.argmax(results['silhouette'])
    best_k = results['k'][best_idx]
    best_score = results['silhouette'][best_idx]
    
    print(f"\n{'='*60}")
    print(f"✅ Best k: {best_k} (silhouette={best_score:.4f})")
    print(f"{'='*60}")
    
    return results