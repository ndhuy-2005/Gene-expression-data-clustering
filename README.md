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
## ⚙️ Quy trình chi tiết



1. **Tiền xử lý & Giảm chiều (Preprocessing & Dimensionality Reduction):**

   * Sử dụng `StandardScaler` đưa các thuộc tính về cùng thang đo.

   * Áp dụng **PCA** để giảm chiều không gian đặc trưng từ $22277$ gen xuống $10$ thành phần chính (PC), loại bỏ nhiễu và bảo toàn thông tin quan trọng.



2. **Huấn luyện Mô hình Thành phần (Base Models):**

   * **K-Means:** Chạy $128$ lần ngẫu nhiên để tạo tính đa dạng cục bộ ("nhiều phiếu bầu nhỏ").

   * **Agglomerative Clustering:** Chọn $k=5$ dựa trên phân tích Dendrogram tại khoảng cách phân tách rõ ràng (~200). Đóng vai trò là "phiếu bầu lớn" ổn định.

   * **Spectral Clustering:** Thiết lập $k=5$, sử dụng đồ thị $k$-NN ($n\_neighbors=10$), đóng vai trò là "phiếu bầu lớn" với khả năng phân tách cấu trúc phi tuyến.



3. **Xây dựng Ma trận Tương đồng (Similarity Matrix Formulation):**

   * Chuyển đổi nhãn phân cụm thành ma trận tương đồng $A_{ij} = 1$ nếu $x_i, x_j$ thuộc cùng cụm và $0$ nếu ngược lại.

   * Tổng hợp các ma trận với bộ trọng số tùy chỉnh $W = [w_{kmeans}, w_{agg}, w_{spectral}]$.



4. **Tái phân cụm (Re-clustering):**

   * Áp dụng **Spectral Clustering** với `affinity='precomputed'` trên Ma trận Tương đồng tổng hợp đã chuẩn hóa.



5. **Đánh giá (Evaluation):**

   * Sử dụng **Silhouette Score**, **Inertia**, và **Adjusted Rand Index (ARI)** để đánh giá chất lượng phân cụm và độ ổn định.



---



## 📊 Kết quả thực nghiệm chính



* **Hiệu năng tổng thể:**

  * Mô hình Ensemble với cấu hình mặc định $W = [1, 1, 1]$ cho ma trận tương đồng dạng khối chéo (*block-diagonal*) rất sắc nét, triệt tiêu nhiễu cục bộ tốt hơn từng thuật toán đơn lẻ.

  * Chỉ số Silhouette khảo sát từ $k=2 \rightarrow 19$ cho thấy mức độ tách biệt cụm ổn định.



* **Phân tích Trọng số (Weighted Analysis):**

  * $W = [1, 3, 5]$ (Ưu tiên Spectral): Bắt tốt cấu trúc phi tuyến phức tạp.

  * $W = [5, 1, 3]$ (Ưu tiên K-Means): Các cụm phân bố cân bằng hơn quanh tâm mật độ.

  * $W = [3, 5, 1]$ (Ưu tiên Agglomerative): Thay đổi vai trò cụm chính-phụ dựa trên cấu trúc cây phân cấp.



* **Thử nghiệm Loại bỏ Thành phần (Ablation Study):**

  * Thử nghiệm cho thấy ngay cả khi loại bỏ 1 thuật toán nền ($w_i = 0$), hệ thống Ensemble vẫn duy trì tính ổn định cao và tự điều chỉnh linh hoạt.



---



## 🛠️ Công nghệ & Thư viện sử dụng

* **Ngôn ngữ:** Python 3.x

* **Thư viện chính:**

  * `numpy`, `pandas`: Xử lý ma trận & dữ liệu

  * `scikit-learn`: Tiền xử lý (StandardScaler, PCA), các thuật toán phân cụm (KMeans, AgglomerativeClustering, SpectralClustering) & các chỉ số đánh giá (silhouette_score, adjusted_rand_score)

  * `matplotlib`, `seaborn`: Trực quan hóa dữ liệu (Scatter plots, Similarity Heatmaps, Dendrograms)



---



## 📁 Cấu trúc Repository đề xuất



```text

├── data/

│   └── GSE44861.csv              # Dữ liệu gen biểu hiện (GEO Accession: GSE44861)

├── notebooks/

│   ├── 01_Preprocessing_PCA.ipynb

│   ├── 02_Base_Models_Evaluation.ipynb

│   └── 03_Ensemble_Clustering.ipynb

├── src/

│   ├── preprocessing.py          # Tiền xử lý & PCA

│   ├── base_models.py            # Huấn luyện KMeans, Agglomerative, Spectral

│   └── ensemble.py               # Xây dựng ma trận tương đồng & Re-clustering

├── report/

│   └── Machine_Learning_Report.pdf # Báo cáo chi tiết dạng PDF

├── README.md

└── requirements.txt

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





