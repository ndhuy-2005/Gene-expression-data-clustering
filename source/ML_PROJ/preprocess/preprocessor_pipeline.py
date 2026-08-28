from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

class PreprocessPipeline:
    def __init__(self, use_pca=True, n_components=50):
        self.use_pca = use_pca
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = None

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)

        if self.use_pca:
            n_samples, n_features = X_scaled.shape
            n_comp = min(self.n_components, n_samples, n_features)

            self.pca = PCA(n_components=n_comp)
            return self.pca.fit_transform(X_scaled)

        return X_scaled

    def transform(self, X):
        X_scaled = self.scaler.transform(X)

        if self.use_pca and self.pca is not None:
            n_samples, n_features = X_scaled.shape
            n_comp = min(self.pca.n_components, n_samples, n_features)

            # Nếu PCA cũ quá lớn → tạo PCA mới cho file nhỏ
            if n_comp != self.pca.n_components:
                pca_tmp = PCA(n_components=n_comp)
                return pca_tmp.fit_transform(X_scaled)

            return self.pca.transform(X_scaled)

        return X_scaled
