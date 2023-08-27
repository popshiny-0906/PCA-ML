import math
import matplotlib.pyplot as plt
from sklearn import datasets

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        num_samples, num_features = len(X), len(X[0])
        self.mean = [sum(X[i][j] for i in range(num_samples)) / num_samples for j in range(num_features)]
        
        for i in range(num_samples):
            for j in range(num_features):
                X[i][j] -= self.mean[j]
        
        cov_matrix = [[0] * num_features for _ in range(num_features)]
        for i in range(num_features):
            for j in range(num_features):
                cov_matrix[i][j] = sum(X[k][i] * X[k][j] for k in range(num_samples)) / (num_samples - 1)
        
        eigenvalues, eigenvectors = self.eigenvectors_cov(cov_matrix)
        sorted_eigenvalues, sorted_eigenvectors = self.sort_eigen(eigenvalues, eigenvectors)
        
        self.components = sorted_eigenvectors[:self.n_components]

    def transform(self, X):
        num_samples = len(X)
        for i in range(num_samples):
            for j in range(len(X[0])):
                X[i][j] -= self.mean[j]
        
        transformed_data = []
        for sample in X:
            transformed_sample = [sum(sample[i] * self.components[j][i] for i in range(len(sample))) for j in range(self.n_components)]
            transformed_data.append(transformed_sample)
        return transformed_data
    
    def eigenvectors_cov(self, cov_matrix):
        num_features = len(cov_matrix)
        
        eigenvalues, eigenvectors = [], []
        for i in range(num_features):
            eigenvalue, eigenvector = 0, [0] * num_features
            for j in range(num_features):
                eigenvalue += cov_matrix[i][j] * cov_matrix[j][i]
            eigenvalues.append(eigenvalue)
            eigenvectors.append(eigenvector)
        
        return eigenvalues, eigenvectors
    
    
    if__name__=="main":
        X=data.data_module
        Y=data.target_names
    
    def sort_eigen(self, eigenvalues, eigenvectors):
        sorted_indices = sorted(range(len(eigenvalues)), key=lambda k: eigenvalues[k], reverse=True)
        sorted_eigenvalues = [eigenvalues[i] for i in sorted_indices]
        sorted_eigenvectors = [eigenvectors[i] for i in sorted_indices]
        return sorted_eigenvalues, sorted_eigenvectors


# Testing
if __name__ == "__main__":
    data = datasets.load_iris()
    X = data.data
    y = data.target

    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print("Shape of X:", len(X))
    print("Shape of transformed X:", len(X_projected))

    x1 = [sample[0] for sample in X_projected]
    x2 = [sample[1] for sample in X_projected]

    plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()
