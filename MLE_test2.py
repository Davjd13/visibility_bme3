from ts2vg import NaturalVG
from ts2vg import HorizontalVG
import networkx as nx
import pandas as pd
import MLE_functions
import csv
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def statistical(X):
    n = nx.number_of_nodes(X)
    m = nx.number_of_edges(X)
    AD = round(2 * m / n, 3)  # average degree
    Den = round(AD / (n - 1), 3)  # density
    ASPL = round(nx.average_shortest_path_length(X), 3)  # average shortest path length
    GE = round(nx.global_efficiency(X), 3)  # global efficiency
    D = nx.diameter(X)  # diameter
    CC = round(sum(nx.closeness_centrality(X).values()), 3)  # closeness centrality
    T = round(nx.transitivity(X), 3)  # transitivity
    ACC = round(nx.average_clustering(X), 3)  # average clustering coefficient
    B = nx.betweenness_centrality(X)
    Bmax = max(B.values())
    CPD = round((1 / (n-1)) * sum(Bmax - Bi for Bi in B.values()), 3)  # central point dominance
    AC = round(nx.degree_assortativity_coefficient(X), 3)  # degree assortativity coefficient
    results = [AD, Den, ASPL, GE, D, CC, T, ACC, CPD, AC]
    return results

x_value = 'P4'
y_values = [512, 1024, 2048, 4096]
audio_folder = 'D:\\TS2VG\\Audio'
audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.csv')]

csv_file_template = 'fea_ex_v4_nodes_{}.csv'

for y_length in y_values:
    all_results = []

    for audio_file in audio_files:
        file_path = os.path.join(audio_folder, audio_file)
        data = pd.read_csv(file_path)

        y_first = 0
        while y_first + y_length <= 4096:
            dat = data[x_value][y_first:y_first + y_length]
            nodes = range(len(dat))
            vg = HorizontalVG()
            vg.build(dat)
            edges = vg.edges
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)

            k_min = MLE_functions.degree_list(G).min()
            fit_result = MLE_functions.fit('Graph', G, k_min=k_min, plot_type='ccdf', save=False)
            stats_result = statistical(G)

            row_data = [audio_file, x_value, y_length, fit_result[0], fit_result[1], np.round(fit_result[2][0][0], 2),
                        np.round(fit_result[2][0][1], 2) if fit_result[2][0].any() and len(fit_result[2][0]) > 1 else 0,
                        *stats_result]

            all_results.append(row_data)

            y_first += y_length

    csv_file_path = csv_file_template.format(y_length)

    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        top_level_header = ['File', 'Channel', 'Nodes', 'Kmin', 'Fit', 'Param1', 'Param2'] + \
                           ['AD', 'Den', 'ASPL', 'GE', 'D', 'CC', 'T', 'ACC', 'CPD', 'AC']
        writer.writerow(top_level_header)

        for row_data in all_results:
            writer.writerow(row_data)

    print(f"Results for {y_length} nodes saved to {csv_file_path}")
