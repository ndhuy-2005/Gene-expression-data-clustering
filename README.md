# Phân Cụm Dữ Liệu Biểu Hiện Gen Với Mô Hình Ensemble (Gene Expression Clustering using Ensemble Model)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Topic](https://img.shields.io/badge/Topic-Bioinformatics%20%26%20Machine%20Learning-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

---

## 📌 Giới thiệu đề tài
Bài toán phân cụm dữ liệu biểu hiện gen (*Gene Expression Data*) đóng vai trò quan trọng trong sinh học phân tử nhằm phát hiện các phân nhóm bệnh lý và khám phá chức năng gen. Dữ liệu gen thường mang các thách thức lớn như **số chiều rất cao** (*High-dimensional*), **nhiễu sinh học cao** và **cấu trúc cụm phức tạp, phi tuyến tính**.

Đồ án này nghiên cứu và đề xuất phương pháp **Ensemble Clustering** (Phân cụm tổ hợp) kết hợp 3 thuật toán nền tảng:
1. **K-Means Clustering** (Đại diện cho phương pháp phân hoạch)
2. **Agglomerative Hierarchical Clustering** (Đại diện cho phương pháp phân cấp)
3. **Spectral Clustering** (Đại diện cho phương pháp dựa trên đồ thị)

Mục tiêu chính là kết hợp kết quả từ các mô hình thành phần thông qua **Ma trận tương đồng (Consensus/Similarity Matrix)** kèm trọng số linh hoạt, sau đó tái phân cụm bằng Spectral Clustering để thu được kết quả ổn định, chính xác và có độ tin cậy cao nhất.

## 🏗️ Architecture & Workflow

Mô hình được triển khai qua 5 bước chính:

```mermaid
flowchart TD
    %% Dataset input
    A["Raw Gene Expression Data<br/>(111 samples × 22,277 genes)"] --> B["Pre-processing<br/>StandardScaler + PCA (10 Components)"]
    
    %% Base models feature split
    B --> C1["K-Means Clustering<br/>(128 Runs, Random Init)"]
    B --> C2["Agglomerative Clustering<br/>(k = 5, Linkage = Average)"]
    B --> C3["Spectral Clustering<br/>(k = 5, Affinity = Nearest Neighbors)"]
    
    %% Similarity matrices
    C1 --> D1["128 Similarity Matrices<br/>(K-Means)"]
    C2 --> D2["1 Similarity Matrix<br/>(Agglomerative)"]
    C3 --> D3["1 Similarity Matrix<br/>(Spectral)"]
    
    %% Fusion with weights
    D1 --> E["Weighted Matrix Fusion<br/>Applying Weight Vector W = [w1, w2, w3]"]
    D2 --> E
    D3 --> E
    
    %% Final ensemble steps
    E --> F["Combined Consensus Matrix<br/>(Normalized 111 × 111)"]
    F --> G["Spectral Re-clustering<br/>(Affinity = Precomputed)"]
    G --> H["Final Cluster Labels<br/>(Evaluated via Silhouette & ARI)"]

    %% Styling
    style A fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#fff
    style B fill:#1e3a8a,stroke:#60a5fa,stroke-width:2px,color:#fff
    style C1 fill:#0f766e,stroke:#14b8a6,stroke-width:1px,color:#fff
    style C2 fill:#0f766e,stroke:#14b8a6,stroke-width:1px,color:#fff
    style C3 fill:#0f766e,stroke:#14b8a6,stroke-width:1px,color:#fff
    style E fill:#6b21a8,stroke:#a855f7,stroke-width:2px,color:#fff
    style F fill:#374151,stroke:#9ca3af,stroke-width:1px,color:#fff
    style G fill:#1d4ed8,stroke:#3b82f6,stroke-width:2px,color:#fff
    style H fill:#15803d,stroke:#22c55e,stroke-width:2px,color:#fff
