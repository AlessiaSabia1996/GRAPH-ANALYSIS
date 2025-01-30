#!/usr/bin/env python
# coding: utf-8

# In[1]:


import community as community_louvain
import matplotlib.pyplot as plt
import networkx as nx
from nilearn import datasets
from nilearn import plotting, image
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import numpy as np
import os
import pandas as pd
import re
import pickle
from collections import defaultdict


# In[2]:


def load_csv_file(file_path):
    csv = pd.read_csv(file_path)    
    return csv


# In[3]:


def compute_conn_matrix(path, m1,m2, c, df, file_extension):
    total_files = 0
    subject_names = df['ID_centre']
    correlation_matrices_cortical = []
    correlation_matrices_subcortical = []

    for subdirs, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(file_extension):
                filepath = os.path.join(subdirs,file)
                match = re.search(r"(sub-\d+)", filepath) # extract subname from the filepath
                if match:
                    sub_name = match.group(1)
                    if sub_name in subject_names.values:
                        print(f'Elaborating subject: {sub_name}')
                        sub_row = df.loc[df['ID_centre'] == sub_name] #select the row of the corresponding subject
                        patient = sub_row['ID_centre'].values[0]
                        sex = sub_row['SEX'].values[0]
                        age = sub_row['AGE'].values[0]
                        group = sub_row['GROUP'].values[0]
                        ms_type = sub_row['MS_type'].values[0]
                        edss = sub_row['EDSS'].values[0]
                        sdmt = sub_row['SDMT'].values[0]
                        #education_years = sub_row['EDUCATION_years'].values[0]
                        
                        img_confounds, _ = load_confounds(filepath, strategy=('motion', 'wm_csf', 'high_pass', 'global_signal'))
                        time_series_cort = m1.fit_transform(filepath, confounds=img_confounds) #extract signals from regions defined by the atlas
                        time_series_subcort = m2.fit_transform(filepath, confounds=img_confounds)
                        
                        correlation_matrix_cort = c.fit_transform([time_series_cort])[0]
                        correlation_matrix_subcort = c.fit_transform([time_series_subcort])[0]
                        
                        correlation_matrices_cortical.append((patient, sex, age, group, ms_type,
                                                              edss, sdmt, correlation_matrix_cort))
                        correlation_matrices_subcortical.append((patient, sex, age, group, ms_type, edss,
                                                                 sdmt, correlation_matrix_subcort))

                        save_path = os.path.join(subdirs, f'{sub_name}_cortical_correlation_matrix_fmriprep.npy')
                        np.save(save_path, correlation_matrix_cort)

                        save_path_subcort = os.path.join(subdirs, f'{sub_name}_subcortical_correlation_matrix_fmriprep.npy')
                        np.save(save_path_subcort, correlation_matrix_subcort)
                
                
                
                total_files += 1
                print(f'filepath {filepath}')
                #print(f'subname {sub_name}')
                #print(f'patient {patient}, sex {sex}, age {age}, group {group}, ms_type {ms_type}, edss {edss}, sdmt{sdmt}, education years {education_years}')
    #print(subject_names)
    print(f'Files elaborated: {total_files}')
    return correlation_matrices_cortical, correlation_matrices_subcortical


# In[4]:


def create_graph_with_threshold(adjacency_matrix, threshold, labels):

    G = nx.Graph()

    edges = []
    for i in range(adjacency_matrix.shape[0]):
        for j in range(i + 1, adjacency_matrix.shape[1]):
            weight = adjacency_matrix[i][j]
            if weight > 0:  
                edges.append((labels[i], labels[j], weight))

    total_edges = len(edges)
    
    if total_edges == 0:
        raise ValueError("Not enough edges in adjacency matrix")
    
    edges_to_keep = int(total_edges * threshold)

    selected_edges = np.random.choice(len(edges), edges_to_keep, replace=False)
    
    for idx in selected_edges:
        G.add_edge(edges[idx][0], edges[idx][1], weight=edges[idx][2])

    return G


# In[5]:


def upload_matrices(path, df):
    subject_names = df['ID_centre']
    patients_cortical_matrices = []
    patients_subcortical_matrices = []
    controls_cortical_matrices = []
    controls_subcortical_matrices = []
    for subdirs, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.npy'):
                filename = os.path.join(subdirs,file)
                
                match = re.search(r"(cortical|subcortical)", file)  
                match_sub = re.search(r"(sub-\d+)", filename) 
                
                if match and match_sub:
                    sub_name = match_sub.group(1)
                    if sub_name in subject_names.values:                        
                        region = match.group(1)
                        matrix = np.load(filename)
                        sub_row = df.loc[df['ID_centre'] == sub_name] #select the row of the corresponding subject    
                        patient = sub_row['ID_centre'].values[0]
                        sex = sub_row['SEX'].values[0]
                        age = sub_row['AGE'].values[0]
                        group = sub_row['GROUP'].values[0]
                        ms_type = sub_row['MS_type'].values[0]
                        edss = sub_row['EDSS'].values[0]
                        sdmt = sub_row['SDMT'].values[0]
                        if group == 'HC':
                            if region == "cortical":
                                controls_cortical_matrices.append((patient, matrix, group, sex, age, ms_type, edss, sdmt))
                            if region == "subcortical":
                                controls_subcortical_matrices.append((patient, matrix, group, sex, age, ms_type, edss, sdmt))
                        if group == 'MS':
                            if region == "cortical":
                                patients_cortical_matrices.append((patient, matrix, group, sex, age, ms_type, edss, sdmt))
                            if region == "subcortical":
                                patients_subcortical_matrices.append((patient, matrix, group, sex, age, ms_type, edss, sdmt))
                    
    return patients_cortical_matrices, patients_subcortical_matrices, controls_cortical_matrices, controls_subcortical_matrices


# In[6]:


def weighted_graph_density(G):
    # Numero di nodi nel grafo
    num_nodes = len(G.nodes)
    
    if num_nodes < 2:
        return 0

    total_weight = sum(data['weight'] for u, v, data in G.edges(data=True))
    
    density = (2 * total_weight) / (num_nodes * (num_nodes - 1))
    
    return density


# In[39]:


def compute_metrics(list_of_martices, cortical= False, cl=None ,sl=None, graph_p):
    nodes_degree = []
    betweenness = []
    nodes_degree_centality= []
    clustering = []
    globalEfficiency = []
    graph_density = []
    vulnerability = []
    modularities = []
    transitivities = []
    eigens = []
    graphs = []
    spl = []
    clust_avg = []
    partitions = []
    
    #0: patient, 1: matrix, 2: group, 3: sex, 4: age, 5: ms_type, 6: edss, 7: sdmt
    for matrix in list_of_martices:
        print(f'Elaborating patient {matrix[0]}')
        np.fill_diagonal(matrix[1], 0)

        if cortical:
            graph = create_graph_with_threshold(matrix[1], threshold=0.8, labels=cl)
        else:
            graph = create_graph_with_threshold(matrix[1], threshold=0.8, labels=sl)
        graph_path = graph_p
        with open(graph_path, "wb") as f:
            pickle.dump(graph, f)
        
        graphs.append((matrix[0], graph))
        #node degree centrality
        nodes_degree_centality.append((matrix[0],nx.degree_centrality(graph), matrix[3], matrix[4], matrix[5], matrix[6], matrix[7]))
        
        #eigenvector centrality
        eigens.append((matrix[0], nx.eigenvector_centrality(graph, weight='weight'), matrix[3], matrix[4], matrix[5], matrix[6], matrix[7]))
        
        #nodes degree
        nodes_degree.append((matrix[0],nx.degree(graph), matrix[3], matrix[4], matrix[5], matrix[6], matrix[7]))
        
        #betweenness centrality
        betweenness.append((matrix[0], nx.betweenness_centrality(graph, normalized=True, weight="weight"), matrix[3], matrix[4], matrix[5], matrix[6], matrix[7]))
        
        #clustering coefficient for single node
        clustering.append((matrix[0], nx.clustering(graph, weight='weight'), matrix[3], matrix[4], matrix[5], matrix[6], matrix[7]))
        
        #average clustering coefficient
        clust_avg.append((matrix[0], nx.average_clustering(graph, weight='weight'), matrix[3], matrix[4], matrix[5], matrix[6], matrix[7]))
        
        #Global efficiency
        globalEfficiency.append((matrix[0], nx.global_efficiency(graph),  matrix[3], matrix[4], matrix[5], matrix[6], matrix[7]))
        
        #graph density
        graph_density.append((matrix[0], weighted_graph_density(graph), matrix[3], matrix[4], matrix[5], matrix[6], matrix[7]))
        
        #communities_extraction 
        #A list of sets (partition of G). 
        #Each set represents one community and contains all the nodes that constitute it.
        partitions_of_graph = nx.community.louvain_communities(graph)
        partitions.append((matrix[0], partitions_of_graph, matrix[3], matrix[4], matrix[5], matrix[6], matrix[7]))
        
        #modularity
        modularity = nx.community.modularity(graph,partitions_of_graph)

        modularities.append((matrix[0], partitions_of_graph, modularity, matrix[3], matrix[4], matrix[5], matrix[6], matrix[7]))
        
        #node vulnerability
        vulnerabilities = {}
    
        
        for node in graph.nodes:
            
            G_copy = graph.copy()
            G_copy.remove_node(node)

            efficiency_without_node = nx.global_efficiency(G_copy)

            vulnerabilities[node] = (nx.global_efficiency(graph) - efficiency_without_node) / nx.global_efficiency(graph)
        vulnerability.append((matrix[0], vulnerabilities, matrix[3], matrix[4], matrix[5], matrix[6], matrix[7]))
        
        #shortest path lenght
        spl.append((matrix[0], nx.average_shortest_path_length(graph), matrix[3], matrix[4], matrix[5], matrix[6], matrix[7]))
        
        #transitivity
        transitivities.append((matrix[0], nx.transitivity(graph), matrix[3], matrix[4], matrix[5], matrix[6], matrix[7]))
                             
                             
                             
                             
    return graphs, nodes_degree_centality, eigens, nodes_degree, betweenness, clustering, clust_avg, globalEfficiency, graph_density, partitions, modularities, vulnerability, spl, transitivities


# In[65]:


def save_metrics_to_csv(nodes_degree_centality, eigens, nodes_degree, betweenness, clustering, clust_avg, globalEfficiency, graph_density, partitions, modularities, vulnerability, spl, transitivities, output_dir):
    
    nodes_degree_ = []
    for patient, degree_dict, _,_,_,_,_ in nodes_degree:
        for node, degree_value in dict(degree_dict).items():
            nodes_degree_.append({"Patient": patient, "Node": node, "Degree": degree_value})
    pd.DataFrame(nodes_degree_).to_csv(f"{output_dir}/nodes_degree.csv", index=False)
    
    nodes_degree_rows = []
    for patient, degree_dict, _,_,_,_,_ in nodes_degree_centality:
        for node, degree_value in dict(degree_dict).items():
            nodes_degree_rows.append({"Patient": patient, "Node": node, "DegreeCentrality": degree_value})
    pd.DataFrame(nodes_degree_rows).to_csv(f"{output_dir}/nodes_degree_centrality.csv", index=False)
    
    eigen_rows = []
    for patient, eigen_dict, _,_,_,_, _ in eigens:
        for node, eig in dict(eigen_dict).items():
            eigen_rows.append({"Patient": patient, "Node": node, "Eigenvector": eig})
    pd.DataFrame(eigen_rows).to_csv(f"{output_dir}/eigenvector_centrality.csv", index=False)
    
    betweenness_rows = []
    for patient, bet_dict, _,_,_,_,_ in betweenness:
        for node, bet_value in bet_dict.items():
            betweenness_rows.append({"Patient": patient, "Node": node, "Betweenness": bet_value})
    pd.DataFrame(betweenness_rows).to_csv(f"{output_dir}/betweenness.csv", index=False)
    
    clustering_rows = []
    for patient, clust_dict, _,_,_,_,_ in clustering:
        for node, clust_value in clust_dict.items():
            clustering_rows.append({"Patient": patient, "Node": node, "Clustering": clust_value})
    pd.DataFrame(clustering_rows).to_csv(f"{output_dir}/clustering.csv", index=False)

    vulnerability_rows = []
    for patient, vuln_dict, _,_,_,_,_ in vulnerability:
        for node, vuln_value in vuln_dict.items():
            vulnerability_rows.append({"Patient": patient, "Node": node, "Vulnerability": vuln_value})
    pd.DataFrame(vulnerability_rows).to_csv(f"{output_dir}/vulnerability.csv", index=False)
    
    
    pd.DataFrame(clust_avg,columns=["Patient", "ClusteringAvg", "Sex", "Age", "MS_type", "EDSS", "SDMT"]).to_csv(f"{output_dir}/clustering_avg.csv", index=False)
    pd.DataFrame(globalEfficiency, columns=["Patient", "Efficiency", "Sex", "Age", "MS_type", "EDSS", "SDMT"]).to_csv(f"{output_dir}/global_efficiency.csv", index=False)
    pd.DataFrame(graph_density, columns=["Patient", "Density", "Sex", "Age", "MS_type", "EDSS", "SDMT"]).to_csv(f"{output_dir}/graph_density.csv", index=False)
    pd.DataFrame(partitions,  columns=["Patient", "Partitions", "Sex", "Age", "MS_type", "EDSS", "SDMT"]).to_csv(f"{output_dir}/partitions_in_graph.csv", index=False)
    pd.DataFrame(transitivities, columns=["Patient", "Transitivity", "Sex", "Age", "MS_type", "EDSS", "SDMT"]).to_csv(f"{output_dir}/transitivity.csv", index=False)
    pd.DataFrame(modularities, columns=["Patient", "Modularity", "Partition","Sex", "Age", "MS_type", "EDSS", "SDMT"]).to_csv(f"{output_dir}/modularities.csv", index=False)
    pd.DataFrame(spl, columns=["Patient", "SPL", "Sex", "Age", "MS_type", "EDSS", "SDMT"]).to_csv(f"{output_dir}/spl.csv", index=False)
    
    
   


# In[ ]:


dataset_schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7) #cortical atlas
cortical_labels = dataset_schaefer.labels
#cortical_labels = np.insert(dataset_schaefer.labels, 0, "Background")
cortical_maps = dataset_schaefer.maps

dataset_harvard_oxford = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm", symmetric_split=True)
subcortical_labels = dataset_harvard_oxford.labels
subcortical_filename = dataset_harvard_oxford.filename
subcortical_maps = dataset_harvard_oxford.maps

plotting.plot_roi(cortical_maps)
plotting.plot_roi(subcortical_filename, view_type="contours")
plotting.show()


masker_cortical = NiftiLabelsMasker(labels_img=cortical_maps,
                           standardize=True)
masker_subcortical = NiftiLabelsMasker(labels_img=subcortical_maps,
    standardize=True
)

correlation = ConnectivityMeasure(
    kind="correlation"
)


# In[ ]:


yeo_networks = [str(label).split('_')[2] for label in cortical_labels[1:]]  

yeo_regions_dict = {}
for label, network in zip(cortical_labels[1:], yeo_networks):  
    if network not in yeo_regions_dict:
        yeo_regions_dict[network] = []
    yeo_regions_dict[network].append(label)

for network, regions in yeo_regions_dict.items():
    print(f"Rete Yeo {network}")


# In[14]:


mainz_data = load_csv_file(csv_path)


# In[ ]:


cortical_matrix, subcortical_matrix = compute_conn_matrix(path = derivatives_path,
                                                          m1=masker_cortical,m2=masker_subcortical,c=correlation,df=mainz_data, file_extension = '.nii.gz')


# In[18]:


correlation_matrix = cortical_matrix[0][7]
print(f' patient {cortical_matrix[0][0]} of group {cortical_matrix[0][3]}')
np.fill_diagonal(correlation_matrix, 0)

plotting.plot_matrix(
    correlation_matrix,
    figure=(10, 8),
    labels=cortical_labels,
    vmax=0.8,
    vmin=-0.8,
    title="Motion, WM, CSF",
    reorder=True,
)


# In[21]:


import seaborn as sns
sns.histplot(patients_cortical[0][1], legend=False)
positive_values = (patients_cortical[0][1] > 0).sum()
negative_values = (patients_cortical[0][1] < 0).sum()

print(f"Numero di valori positivi: {positive_values}")
print(f"Numero di valori negativi: {negative_values}")

coords = plotting.find_parcellation_cut_coords(cortical_maps)
view = plotting.view_connectome(
    correlation_matrix, coords, edge_threshold="80%"
)
view

# In[20]:


patients_cortical, patients_subcortical, controls_cortical, controls_subcortical = upload_matrices(path_matrix, mainz_data)


# In[ ]:


def calculate_mean_adjacency_matrix(adjacency_matrices, method='mean'):
    stacked_matrices = np.stack(adjacency_matrices, axis=0)
    if method == 'mean':
        return np.mean(stacked_matrices, axis=0)
    elif method == 'median':
        return np.median(stacked_matrices, axis=0)

def create_graph_from_adjacency(adjacency_matrix):
    G = nx.from_numpy_array(adjacency_matrix)
    return G

def calculate_global_metrics(G):
    clustering_coefficient = nx.average_clustering(G)
    try:
        path_length = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        path_length = np.nan  
    return clustering_coefficient, path_length

def calculate_random_metrics(G, num_randomizations=100):
    clustering_coeffs = []
    path_lengths = []
    for _ in range(num_randomizations):
        G_random = nx.expected_degree_graph([d for _, d in G.degree()], selfloops=False)
        
        clustering_coeffs.append(nx.average_clustering(G_random))
        try:
            path_lengths.append(nx.average_shortest_path_length(G_random))
        except nx.NetworkXError:
            path_lengths.append(np.nan)  

    C_rand = np.nanmean(clustering_coeffs)
    L_rand = np.nanmean(path_lengths)
    return C_rand, L_rand

def calculate_small_worldness(C, L, C_rand, L_rand):
    return (C / C_rand) / (L / L_rand)

def compute_group_small_worldness(adjacency_matrices):
    mean_adjacency_matrix = calculate_mean_adjacency_matrix(adjacency_matrices, method='mean')
    
    G_mean = create_graph_from_adjacency(mean_adjacency_matrix)
    
    C, L = calculate_global_metrics(G_mean)
    
    C_rand, L_rand = calculate_random_metrics(G_mean)
    
    S = calculate_small_worldness(C, L, C_rand, L_rand)
    return S

patients_matrixes = []
for patient, matrix, group, sex, age, ms_type, edss, sdmt in patients_cortical:
    patients_matrixes.append(matrix)
controls_matrixes = []
for patient, matrix, group, sex, age, ms_type, edss, sdmt in controls_cortical:
    controls_matrixes.append(matrix)
    
small_worldness_patients = compute_group_small_worldness(patients_matrixes)
small_worldness_controls = compute_group_small_worldness(controls_matrixes)

print(f"Small-worldness (Patients): {small_worldness_patients:.10f}")
print(f"Small-worldness (Controls): {small_worldness_controls:.10f}")



METRICS COMPUTATION FOR PATIENTS
# In[66]:


pc_graphs, nodes_degree_centality, eigens, nodes_degree, betweenness, clustering, clust_avg, globalEfficiency, graph_density, partitions, modularities, vulnerability, spl, transitivities = compute_metrics(patients_cortical, cortical=True, cl= cortical_label, graph_path)
save_metrics_to_csv(nodes_degree_centality, eigens, nodes_degree, betweenness, clustering, clust_avg, globalEfficiency, graph_density, partitions, modularities, vulnerability, spl, transitivities, output_dir=outdir)


# In[ ]:


cc_graphs,nodes_degree_centality, eigens, nodes_degree, betweenness, clustering, clust_avg, globalEfficiency, graph_density, partitions, modularities, vulnerability, spl, transitivities = compute_metrics(controls_cortical, cortical=True, cl= cortical_labels)
save_metrics_to_csv(nodes_degree_centality, eigens, nodes_degree, betweenness, clustering, clust_avg, globalEfficiency, graph_density, partitions, modularities, vulnerability, spl, transitivities, output_dir=oudir)


# In[ ]:


patient_subcortical_graphs, ps_node_degree, ps_betweenness, ps_clustering, ps_globalEfficiency, ps_graph_density, ps_vulnerability, ps_modularities = compute_metrics(patients_subcortical, cortical=False, sl= subcortical_labels)
save_metrics_to_csv(ps_node_degree, ps_betweenness, ps_clustering, ps_globalEfficiency, ps_graph_density, ps_vulnerability, ps_modularities, output_dir=outdir)


# In[ ]:


control_subcortical_graphs,cs_node_degree, cs_betweenness, cs_clustering, cs_globalEfficiency, cs_graph_density, cs_vulnerability, cs_modularities = compute_metrics(controls_subcortical, cortical=False, sl= subcortical_labels)
save_metrics_to_csv(cs_node_degree, cs_betweenness, cs_clustering, cs_globalEfficiency, cs_graph_density, cs_vulnerability, cs_modularities, output_dir=outdir)


# In[ ]:


G = patient_cortical_graphs[0][1] 

pos = nx.spring_layout(G, seed=42, k=0.15)  

plt.figure(figsize=(12, 12))
nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", 
        font_size=8, font_weight="bold", node_size=500)

plt.show()


# In[ ]:




