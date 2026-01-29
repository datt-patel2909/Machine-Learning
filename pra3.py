import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = {
    'Feature1': [2.5, 0.5, 2.2, 1.9, 3.1],
    'Feature2': [2.4, 0.7, 2.9, 2.2, 3.0]
}

df = pd.DataFrame(data)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

cov_matrix = np.cov(scaled_data.T)

eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

sorted_index = np.argsort(eigen_values)[::-1]
sorted_eigenvectors = eigen_vectors[:, sorted_index]

k = 1
principal_components = sorted_eigenvectors[:, :k]

pca_data = np.dot(scaled_data, principal_components)

print(pca_data)