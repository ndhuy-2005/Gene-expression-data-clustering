import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess.preprocessor_pipeline import PreprocessPipeline
from ensemble.ensemble_clustering import load_pretrained_labels, compute_matrix, build_weighted_matrix, ensemble_with_k
from utils.geo_loader import load_geo_series_matrix


import numpy as np

def find_best_k():
    """Tự động tìm k tối ưu - GIỐNG NOTEBOOK"""
    try:
        if not file_path.get():
            messagebox.showinfo("Thông báo", "Vui lòng chọn file CSV trước!")
            return
        
        # Get weights
        try:
            w1 = float(w1_entry.get())
            w2 = float(w2_entry.get())
            w3 = float(w3_entry.get())
            weights = [w1, w2, w3]
        except:
            weights = [3, 5, 1]
        
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, "🔍 Đang tìm k tối ưu (k=2 đến 19)...\n")
        result_text.insert(tk.END, "⏱️  Có thể mất 1-2 phút...\n\n")
        root.update()
        
        # Load data
        path = file_path.get()

        if path.endswith(".txt"):
            X = load_geo_series_matrix(path)
        else:
            data = pd.read_csv(path)
            X = data.values

        print("X shape before preprocess:", X.shape)

        
        # Preprocess
        preprocessor_path = "models/preprocessor.pkl"
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            X_pca = preprocessor.transform(X)
        else:
            preprocessor = PreprocessPipeline(use_pca=True, n_components=10)
            X_pca = preprocessor.fit(X)
        
        # Load labels
        kmeans_list_labels, agg_list_labels, spectral_list_labels = load_pretrained_labels()
        
        # Compute matrices
        matrix_kmeans = compute_matrix(kmeans_list_labels)
        matrix_agg = compute_matrix(agg_list_labels)
        matrix_spectral = compute_matrix(spectral_list_labels)
        
        list_matrix = [matrix_kmeans, matrix_agg, matrix_spectral]
        
        # Build ensemble
        ensemble_matrix = build_weighted_matrix(list_matrix, weights)
        
        # Test k từ 2 đến 19 - GIỐNG NOTEBOOK
        from sklearn.metrics import silhouette_score
        
        k_range = range(2, 20)
        results = {
            'k': [],
            'silhouette': []
        }
        
        result_text.insert(tk.END, "k  | Silhouette\n")
        result_text.insert(tk.END, "---|------------\n")
        
        for k in k_range:
            labels = ensemble_with_k(ensemble_matrix, k)
            score = silhouette_score(X_pca, labels)
            
            results['k'].append(k)
            results['silhouette'].append(score)
            
            result_text.insert(tk.END, f"{k:2d} | {score:10.4f}\n")
            root.update()
        
        # Tìm k tốt nhất
        best_idx = np.argmax(results['silhouette'])
        best_k = results['k'][best_idx]
        best_score = results['silhouette'][best_idx]
        
        result_text.insert(tk.END, f"\n{'='*30}\n")
        result_text.insert(tk.END, f"✅ K TỐI ƯU: {best_k}\n")
        result_text.insert(tk.END, f"📏 Silhouette: {best_score:.4f}\n")
        result_text.insert(tk.END, f"{'='*30}\n")
        
        # Tự động set k
        k_entry.delete(0, tk.END)
        k_entry.insert(0, str(best_k))
        
        messagebox.showinfo("Hoàn thành", f"K tối ưu: {best_k}\n(Silhouette: {best_score:.4f})")
        
    except Exception as e:
        messagebox.showerror("Lỗi", f"Lỗi: {str(e)}")

def run_clustering():
    try:
        if not file_path.get():
            messagebox.showerror("Lỗi", "Chưa chọn file CSV")
            return
        
        # Get k và weights
        try:
            k = int(k_entry.get())
            w1 = float(w1_entry.get())
            w2 = float(w2_entry.get())
            w3 = float(w3_entry.get())
            weights = [w1, w2, w3]
        except ValueError:
            messagebox.showerror("Lỗi", "Vui lòng nhập số hợp lệ!")
            return
        
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, "🚀 Đang chạy ensemble clustering...\n\n")
        root.update()
        
                # Load data
        path = file_path.get()

        if path.endswith(".txt"):
            X = load_geo_series_matrix(path)
        else:
            data = pd.read_csv(path)
            X = data.values




        result_text.insert(tk.END, f"✓ Loaded: {X.shape}\n")
        root.update()
        
        # Preprocess
        preprocessor_path = "models/preprocessor.pkl"
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            X_pca = preprocessor.transform(X)
        else:
            preprocessor = PreprocessPipeline(use_pca=True, n_components=10)
            X_pca = preprocessor.fit(X)
        
        result_text.insert(tk.END, f"✓ Preprocessed: {X.shape} → {X_pca.shape}\n\n")
        root.update()
        
        # Load pretrained labels
        result_text.insert(tk.END, "📂 Loading pretrained labels...\n")
        root.update()
        
        kmeans_list_labels, agg_list_labels, spectral_list_labels = load_pretrained_labels()
        
        result_text.insert(tk.END, f"✓ KMeans labels: {kmeans_list_labels.shape}\n")
        result_text.insert(tk.END, f"✓ Agglo labels loaded\n")
        result_text.insert(tk.END, f"✓ Spectral labels loaded\n\n")
        root.update()
        
        # Compute matrices
        result_text.insert(tk.END, "📊 Computing similarity matrices...\n")
        root.update()
        
        matrix_kmeans = compute_matrix(kmeans_list_labels)
        matrix_agg = compute_matrix(agg_list_labels)
        matrix_spectral = compute_matrix(spectral_list_labels)
        
        list_matrix = [matrix_kmeans, matrix_agg, matrix_spectral]
        
        result_text.insert(tk.END, f"✓ Matrices ready: {matrix_kmeans.shape}\n\n")
        root.update()
        
        # Build weighted ensemble matrix
        result_text.insert(tk.END, f"🔗 Building ensemble with weights: [{w1}, {w2}, {w3}]\n")
        root.update()
        
        ensemble_matrix = build_weighted_matrix(list_matrix, weights)
        
        result_text.insert(tk.END, f"✓ Ensemble matrix: {ensemble_matrix.shape}\n\n")
        root.update()
        
        # Clustering with k
        result_text.insert(tk.END, f"🎯 Running Spectral Clustering with k={k}...\n")
        root.update()
        
        labels = ensemble_with_k(ensemble_matrix, k)
        
        # Stats
        unique, counts = np.unique(labels, return_counts=True)
        
        result_text.insert(tk.END, f"\n{'='*50}\n")
        result_text.insert(tk.END, "✅ HOÀN THÀNH!\n")
        result_text.insert(tk.END, f"{'='*50}\n\n")
        
        result_text.insert(tk.END, f"📊 KẾT QUẢ:\n")
        result_text.insert(tk.END, f"   • Số cụm: {k}\n")
        result_text.insert(tk.END, f"   • Weights: [{w1}, {w2}, {w3}]\n\n")
        
        result_text.insert(tk.END, "📈 PHÂN BỐ:\n")
        for label, count in zip(unique, counts):
            pct = (count / len(labels)) * 100
            result_text.insert(tk.END, f"   Cluster {label}: {count} mẫu ({pct:.1f}%)\n")
        
        # Silhouette score
        from sklearn.metrics import silhouette_score
        score = silhouette_score(X_pca, labels)
        result_text.insert(tk.END, f"\n📏 Silhouette Score: {score:.4f}\n")
        
        # Save for visualization
        global global_X_pca, global_labels, global_ensemble_matrix, global_k
        global_X_pca = X_pca
        global_labels = labels
        global_ensemble_matrix = ensemble_matrix
        global_k = k
        
    except Exception as e:
        messagebox.showerror("Lỗi", f"Có lỗi xảy ra:\n{str(e)}")
        import traceback
        result_text.insert(tk.END, f"\n❌ LỖI:\n{traceback.format_exc()}\n")

def visualize():
    """Hiển thị Heatmap + Scatter plot"""
    try:
        if global_X_pca is None:
            messagebox.showinfo("Thông báo", "Vui lòng chạy clustering trước!")
            return
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Heatmap ensemble matrix
        sns.heatmap(global_ensemble_matrix, cmap='viridis', ax=axes[0])
        axes[0].set_title("Heatmap Similarity Matrix", fontsize=14, fontweight='bold')
        
        # 2. Scatter cluster
        scatter = axes[1].scatter(global_X_pca[:, 0], global_X_pca[:, 1], 
                                 c=global_labels, cmap='viridis', s=50, alpha=0.7)
        axes[1].set_xlabel("PCA First Component", fontsize=12)
        axes[1].set_ylabel("PCA Second Component", fontsize=12)
        axes[1].set_title(f"Clustering Result k={global_k}", fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1], label='Cluster')
        
        plt.tight_layout()
        
        # Save
        output_file = f"ensemble_result_k{global_k}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        result_text.insert(tk.END, f"\n💾 Saved: {output_file}\n")
        
        plt.show()
        
    except Exception as e:
        messagebox.showerror("Lỗi", f"Lỗi visualization:\n{str(e)}")

def browse_file():
    path = filedialog.askopenfilename(
        filetypes=[
            ("CSV files", "*.csv"),
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
    )
    if path:
        file_path.set(path)

def main():
    global file_path, result_text, root
    global k_entry, w1_entry, w2_entry, w3_entry
    global global_X_pca, global_labels, global_ensemble_matrix, global_k
    
    # Global vars for visualization
    global_X_pca = None
    global_labels = None
    global_ensemble_matrix = None
    global_k = None
    
    root = tk.Tk()
    root.title("Ensemble Clustering - Simple")
    root.geometry("650x600")
    root.resizable(False, False)
    
    file_path = tk.StringVar()
    
    # Header
    header = tk.Frame(root, bg="#34495e", height=50)
    header.pack(fill=tk.X)
    header.pack_propagate(False)
    tk.Label(header, text="🧬 ENSEMBLE CLUSTERING", 
             font=("Arial", 16, "bold"), fg="white", bg="#34495e").pack(pady=10)
    
    # File selection
    file_frame = tk.Frame(root)
    file_frame.pack(pady=10)
    
    tk.Label(file_frame, text="📂 File CSV:", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5)
    tk.Entry(file_frame, textvariable=file_path, width=40).grid(row=0, column=1, padx=5)
    tk.Button(file_frame, text="Browse", command=browse_file, 
              bg="#3498db", fg="white", font=("Arial", 9, "bold")).grid(row=0, column=2, padx=5)
    
    # Parameters
    param_frame = tk.LabelFrame(root, text="⚙️ Tham số", font=("Arial", 10, "bold"), padx=10, pady=10)
    param_frame.pack(pady=10, padx=20, fill=tk.X)
    
    # k
    tk.Label(param_frame, text="Số cụm (k):", font=("Arial", 10)).grid(row=0, column=0, sticky='w', pady=5)
    k_entry = tk.Entry(param_frame, width=10, font=("Arial", 11))
    k_entry.insert(0, "2")
    k_entry.grid(row=0, column=1, padx=10, pady=5)
    
    # Weights
    tk.Label(param_frame, text="Weights:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky='w', pady=5)
    
    weights_frame = tk.Frame(param_frame)
    weights_frame.grid(row=1, column=1, padx=10, pady=5)
    
    tk.Label(weights_frame, text="KMeans:", font=("Arial", 9)).grid(row=0, column=0)
    w1_entry = tk.Entry(weights_frame, width=5, font=("Arial", 10))
    w1_entry.insert(0, "3")
    w1_entry.grid(row=0, column=1, padx=3)
    
    tk.Label(weights_frame, text="Agglo:", font=("Arial", 9)).grid(row=0, column=2, padx=(10,0))
    w2_entry = tk.Entry(weights_frame, width=5, font=("Arial", 10))
    w2_entry.insert(0, "5")
    w2_entry.grid(row=0, column=3, padx=3)
    
    tk.Label(weights_frame, text="Spectral:", font=("Arial", 9)).grid(row=0, column=4, padx=(10,0))
    w3_entry = tk.Entry(weights_frame, width=5, font=("Arial", 10))
    w3_entry.insert(0, "1")
    w3_entry.grid(row=0, column=5, padx=3)
    
    # Buttons
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)
    
    tk.Button(btn_frame, text="🔍 TÌM K TỐI ƯU", command=find_best_k,
              bg="#9b59b6", fg="white", font=("Arial", 10, "bold"),
              height=2, width=18).pack(side=tk.LEFT, padx=3)
    
    tk.Button(btn_frame, text="▶ CHẠY", command=run_clustering,
              bg="#27ae60", fg="white", font=("Arial", 10, "bold"),
              height=2, width=18).pack(side=tk.LEFT, padx=3)
    
    tk.Button(btn_frame, text="📊 XEM", command=visualize,
              bg="#e74c3c", fg="white", font=("Arial", 10, "bold"),
              height=2, width=18).pack(side=tk.LEFT, padx=3)
    
    # Result text
    result_frame = tk.Frame(root)
    result_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
    
    scrollbar = tk.Scrollbar(result_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    result_text = tk.Text(result_frame, height=15, font=("Consolas", 9),
                         yscrollcommand=scrollbar.set, bg="#f8f9fa")
    result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=result_text.yview)
    
    # Footer
    footer = tk.Frame(root, bg="#34495e", height=30)
    footer.pack(fill=tk.X, side=tk.BOTTOM)
    footer.pack_propagate(False)
    tk.Label(footer, text="Gene Expression Ensemble Clustering", 
             fg="white", bg="#34495e", font=("Arial", 8)).pack(pady=5)
    
    root.mainloop()

if __name__ == "__main__":
    main()