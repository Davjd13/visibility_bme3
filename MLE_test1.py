import pickle
import matplotlib.pyplot as plt
from ts2vg import NaturalVG
import networkx as nx
import pandas as pd
import powerlaw
import MLE_functions
import os
import MLE_functions_v2
import numpy as np

# def create_asoiaf():
#     G1 = pickle.load(open('ASoIaF', 'rb'))
#     return G1
# G1 = create_asoiaf()

data1 = pd.read_csv('D:\TS2VG\Audio\s03_ex01_s01.csv')
data2 = pd.read_csv('D:\TS2VG\Audio\s03_ex05.csv')
data3 = pd.read_csv('D:\TS2VG\Audio\s03_ex06.csv')
data4 = pd.read_csv('D:\TS2VG\Audio\s03_ex07.csv')
dat1 = data3['P4'][:2048] #P4 / Cz / F8 / T7
nodes1 = range(len(dat1))
vg1 = NaturalVG()
vg1.build(dat1)
edges1 = vg1.edges
G1 = nx.Graph()
G1.add_nodes_from(nodes1)
G1.add_edges_from(edges1)
list = MLE_functions.degree_list(G1)
print(list)
k_min = list.min()
# result = MLE_functions_v2.fit('Graph', G1, k_min=k_min, plot_type='both', save=False)
fit1 = MLE_functions.fit('Graph', G1, k_min=k_min, plot_type='both', save=False)
# params = MLE_functions.bootstrap(G1_list, result)
# print(MLE_functions.summary_stats(G1_list, result, params))

# fit2 = MLE_functions_v2.fit('Graph', G1, plot_type='both', save=False)
# largest_eigenvalue = max(nx.linalg.spectrum.adjacency_spectrum(G1))
# smallest_eigenvalue = min(nx.linalg.spectrum.adjacency_spectrum(G1))
#
# print(f"Largest Eigenvalue: {largest_eigenvalue}")
# print(f"Smallest Eigenvalue: {smallest_eigenvalue}")
#
# plt.figure(figsize=(8, 8))
# plt.imshow(A.todense(), cmap='gray', vmin=0, vmax=1)
#
# plt.colorbar()
#
# plt.title('Adjacency Matrix of G1')
# plt.xlabel('Nodes')
# plt.ylabel('Nodes')
# plt.show()

# x_values = ['F8']
# number_of_nodes = 1000
# audio_folder = 'D:\\TS2VG\\Audio'
# audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.csv')]
#
# for x in x_values:
#     x_results = []
#
#     for audio_file in audio_files:
#         file_path = os.path.join(audio_folder, audio_file)
#         data = pd.read_csv(file_path)
#
#         channel_results = []
#         dat = data[x][:number_of_nodes]
#         nodes = range(len(dat))
#         vg = NaturalVG()
#         vg.build(dat)
#         edges = vg.edges
#         G = nx.Graph()
#         G.add_nodes_from(nodes)
#         G.add_edges_from(edges)
#
#         fit_result = MLE_functions.fit('Graph', G, plot_type='both', save=False)
