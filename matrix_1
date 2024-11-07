import numpy as np
import pandas as pd

# Load the data from the CSV file
knn_v1 = pd.read_csv("D:\TS2VG\knn_v4.csv")

# Define the mapping of version to 'ab' and nodes to 'cd'
version_ab_mapping = {2: '00', 5: '01', 3: '10', 4: '11'}
nodes_cd_mapping = {512: '00', 1024: '01', 2048: '10', 4096: '11'}

# Initialize the Karnaugh-like matrix
matrix_size = (len(version_ab_mapping), len(nodes_cd_mapping))
karnaugh_matrix = np.zeros(matrix_size)

# Fill the Karnaugh-like matrix with the corresponding values from the CSV file
for index, row in knn_v1.iterrows():
    version = row['Version']
    nodes = row['Nodes']
    score = row['Cross-validation mean score']
    ab = version_ab_mapping[version]
    cd = nodes_cd_mapping[nodes]
    i = list(version_ab_mapping.keys()).index(version)
    j = list(nodes_cd_mapping.keys()).index(nodes)
    karnaugh_matrix[i, j] = score

# Display the Karnaugh-like matrix
print("Karnaugh-Like Matrix:")
print(karnaugh_matrix)
