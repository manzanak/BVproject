# Import necessary libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
import matplotlib.cm as cm # for colormaps
import operator

# Load the datasets
diagnosis_data = pd.read_csv('diagnosis_data.csv')
classification = pd.read_csv('classification.csv')

def plot_fraction_of_reads(subject_ids, otus):
  """Plots the fraction of reads for each patient separately.

  Args:
    subject_ids: A list of subject IDs.
    otus: A list of OTU names.
  """
  for i, patient in enumerate(subject_ids):
    patient_data = diagnosis_data[(diagnosis_data['Subject_ID'] == patient)]
    display(patient_data)

    plt.figure(figsize=(10, 6))
    for j, species in enumerate(otus):
        total_reads = patient_data['total_reads'].astype(float)
        species_read = patient_data[species].astype(float)
        fraction = species_read / total_reads
        species_name = classification.loc[classification['#OTU'] == species.replace('otu', ''), 'Blast hit'].values[0]

        x_labels = patient_data['Follow-up'].astype(str) + "/" + patient_data['BV_status'].astype(str)
        plt.plot(x_labels, fraction, label=species_name)


    plt.title(f'Patient {patient}')
    plt.xlabel('Follow-up / BV Status')
    plt.ylabel('Mean Relative Abundance')
    plt.legend()
    plt.xticks(x_labels) # Set x-axis ticks to follow-up values
    plt.show()

def create_correlation_matrix_network(patient_data, otu_names, threshold=0.4, p_value_threshold=0.10):
    """
    Creates a correlation matrix and co-occurrence network for a given patient's data.

    Args:
        patient_data: DataFrame containing OTU data for a single patient.
        otu_names: DataFrame mapping OTU IDs to taxonomic classifications.
        threshold: The correlation coefficient threshold for creating edges in the network.
        p_value_threshold: The p-value threshold for significance.

    Returns:
        A tuple containing the correlation matrix and the networkx graph.
    """

    # Select OTU columns
    total_reads = patient_data['total_reads'].astype(float)
    OTUs = patient_data.drop(columns=['Follow-up', 'Subject_ID', 'Sample_ID',
                                      'Amsel_discharge','Amsel_odor',
                                      'Amsel_clue','pH','Amsel_score',
                                      'Nugent_Score', 'total_reads',
                                      'BV_status', 'Recurrence_Type'])
    OTUs = OTUs.apply(pd.to_numeric, errors='coerce') #This line converts all columns in OTUs to numeric, coercing any errors to NaN.

    # Normalize OTU reads
    OTUs = OTUs.div(total_reads, axis=0)

    # Using mean abundance
    abundance_threshold = 0.001
    otu_mean_abundance = OTUs.mean(axis=0)

    # Filter OTUs
    filtered_otus = otu_mean_abundance[otu_mean_abundance >= abundance_threshold].index

    # Keep only the filtered OTUs in your normalized data
    filtered_data = OTUs[filtered_otus]

    # Filter out OTUs
    filtered_OTU_names = otu_names.loc[otu_names['#OTU'].isin(filtered_data.columns)]

    # Convert the filtered data to numeric type before normalization
    filtered_data = filtered_data.apply(pd.to_numeric, errors='coerce')

    # Normalize OTU reads
    filtered_data = filtered_data.div(filtered_data.sum(axis=1), axis=0)
    # display(filtered_data)


    # Calculate Spearman's rank correlations
    correlation_matrix = filtered_data.corr(method='spearman')
    otus = correlation_matrix.index

    # Create the network graph
    graph = nx.Graph()
    graph.add_nodes_from(otus)

    for i in range(len(otus)):
        for j in range(i + 1, len(otus)):
            rho = correlation_matrix.iloc[i, j]
            p_value = spearmanr(filtered_data[otus[i]], filtered_data[otus[j]])[1]
            if rho > threshold and p_value < p_value_threshold:
                graph.add_edge(otus[i], otus[j], weight=rho)
    # Calculate node degrees and sizes
    node_degrees = dict(graph.degree())
    node_sizes = [v * 50 for v in node_degrees.values()]

    # Create color map and node colors
    otu_to_taxonomy = dict(zip(filtered_OTU_names["#OTU"], filtered_OTU_names["Blast hit"]))
    unique_taxonomies = filtered_OTU_names["Blast hit"].unique()
    taxonomy_colors = {taxonomy: cm.tab20c(i / len(unique_taxonomies)) for i, taxonomy in enumerate(unique_taxonomies)}
    node_colors = [taxonomy_colors[otu_to_taxonomy.get(node, "Unknown")] for node in graph.nodes()]

    # Visualize the network
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(graph, k=0.5, seed = 42)
    nx.draw(graph, pos, with_labels=True, node_size=node_sizes, node_color=node_colors, font_size=8, width=[d['weight'] * 2 for (u, v, d) in graph.edges(data=True)])
    plt.title(f"Microbial Co-occurrence Network")


    #Create Legend
    legend_patches = []
    for taxonomy in unique_taxonomies:
        otu_numbers = [otu for otu, tax in otu_to_taxonomy.items() if tax == taxonomy]
        label = f"{taxonomy} ({', '.join(otu_numbers)})" #Include all OTU numbers for that taxonomy
        legend_patches.append(mpatches.Patch(color=taxonomy_colors[taxonomy], label=label))

    plt.legend(handles=legend_patches, title="Blast hit (OTU)", loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    plt.show()


    return correlation_matrix, graph

def create_correlation_heatmap(correlation_matrix, follow_up):
    """
    Creates a heatmap of the correlation matrix.

    Args:
        correlation_matrix (pd.DataFrame): The correlation matrix to visualize.
        patient_id (str): The ID of the patient.
    """
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
    plt.title(f"Correlation Matrix Heatmap for Patient {follow_up}")
    plt.show()

def calculate_graph_features(graph):
  """Calculates various features of a graph and returns them as a DataFrame.

  Args:
    graph: A NetworkX graph object.

  Returns:
    A pandas DataFrame containing the calculated graph features.
  """

  features = {}

  # Basic graph properties
  features['number_of_nodes'] = graph.number_of_nodes()
  features['number_of_edges'] = graph.number_of_edges()
  features['density'] = nx.density(graph)

  # Centrality measures
  degree_centrality = nx.degree_centrality(graph)
  features['average_degree_centrality'] = sum(degree_centrality.values()) / len(degree_centrality) if len(degree_centrality) > 0 else 0
  betweenness_centrality = nx.betweenness_centrality(graph)
  features['average_betweenness_centrality'] = sum(betweenness_centrality.values()) / len(betweenness_centrality) if len(betweenness_centrality) > 0 else 0
  closeness_centrality = nx.closeness_centrality(graph)
  features['average_closeness_centrality'] = sum(closeness_centrality.values()) / len(closeness_centrality) if len(closeness_centrality) > 0 else 0
  try:
      eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=500, tol=1e-6)  # Increased max_iter, decreased tol
      features['average_eigenvector_centrality'] = sum(eigenvector_centrality.values()) / len(eigenvector_centrality) if len(eigenvector_centrality) > 0 else 0
  except nx.PowerIterationFailedConvergence:
      print("Eigenvector centrality calculation did not converge. Setting to 0.")
      features['average_eigenvector_centrality'] = 0  # Handle non-convergence

  # Other features
  features['average_clustering_coefficient'] = nx.average_clustering(graph) if graph.number_of_nodes() > 1 else 0
  features['diameter'] = nx.diameter(graph) if nx.is_connected(graph) else float('inf')
  features['average_shortest_path_length'] = nx.average_shortest_path_length(graph) if nx.is_connected(graph) else float('inf')
  features['transitivity'] = nx.transitivity(graph) if graph.number_of_nodes() > 1 else 0
  features['average_node_connectivity'] = nx.average_node_connectivity(graph) if graph.number_of_nodes() > 1 else 0
  features['average_path_length'] = features['average_shortest_path_length'] #Both essentially represent the same feature

  # Convert the dictionary to a DataFrame
  features_df = pd.DataFrame([features])

  return features_df

def get_top_5_nodes(graph):
    """
    Gets the top 5 nodes based on degree centrality.
    """
    degree_centrality = nx.degree_centrality(graph)
    sorted_nodes = dict(sorted(degree_centrality.items(), key=operator.itemgetter(1), reverse=True))
    top_5_nodes = list(sorted_nodes.keys())[:5]
    return top_5_nodes

def measure_node_features(graph):
    """
    Measures various features for each node in a graph.

    Args:
        graph: A NetworkX graph object.

    Returns:
        A pandas DataFrame where each row represents a node and columns are features.
    """

    node_features = {}
    for node in graph.nodes():
        node_features[node] = {}

        # Degree centrality
        node_features[node]['degree_centrality'] = nx.degree_centrality(graph)[node]

        # Betweenness centrality
        node_features[node]['betweenness_centrality'] = nx.betweenness_centrality(graph)[node]

        # Closeness centrality
        node_features[node]['closeness_centrality'] = nx.closeness_centrality(graph)[node]

        # Eigenvector centrality
        try:
            node_features[node]['eigenvector_centrality'] = nx.eigenvector_centrality(graph, max_iter=500, tol=1e-6)[node]
        except nx.PowerIterationFailedConvergence:
            print(f"Eigenvector centrality calculation did not converge for node {node}. Setting to 0.")
            node_features[node]['eigenvector_centrality'] = 0

        # Clustering coefficient
        node_features[node]['clustering_coefficient'] = nx.clustering(graph, nodes=node)

        # Degree
        node_features[node]['degree'] = graph.degree(node)

    return pd.DataFrame.from_dict(node_features, orient='index')

def calculate_influence_score(node_features_df):
    """
    Calculates an influence score for each node based on a weighted combination
    of degree centrality, betweenness centrality, and closeness_centrality.

    Args:
        node_features_df (pd.DataFrame): DataFrame containing node features
                                        with columns 'degree_centrality',
                                        'betweenness_centrality', and
                                        'closeness_centrality'.

    Returns:
        pd.Series: A Series containing the influence score for each node.
    """
    # Define weights for each centrality measure (can be adjusted based on domain knowledge)
    weight_degree = 0.4
    weight_betweenness = 0.4
    weight_closeness = 0.2

    # Ensure the necessary columns exist
    required_columns = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']
    if not all(col in node_features_df.columns for col in required_columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    # Normalize the centrality measures if they are on different scales (optional but recommended)
    # Simple min-max normalization: (x - min) / (max - min)
    normalized_df = node_features_df[required_columns].copy()
    for col in required_columns:
        max_val = normalized_df[col].max()
        min_val = normalized_df[col].min()
        if max_val - min_val != 0:
            normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        else:
            # Handle cases where all values are the same (e.g., for isolated nodes)
            normalized_df[col] = 0

    # Calculate the influence score
    influence_score = (normalized_df['degree_centrality'] * weight_degree +
                       normalized_df['betweenness_centrality'] * weight_betweenness +
                       normalized_df['closeness_centrality'] * weight_closeness)

    return influence_score
# Example Run
imm_recur_subject_IDs = ['3', '5', '9', '10', '11', '16', '27', '35', '38', '39', '43', '44', '45', '47', '55']
imm_recur_subject_data = diagnosis_data[diagnosis_data['Subject_ID'].isin(imm_recur_subject_IDs)]
correlation_matrix_imm_recur, graph_imm_recur = create_correlation_matrix_network(imm_recur_subject_data, OTU_names)
features = calculate_graph_features(graph_imm_recur)
print("Features for immediate recurrence subjects:")
display(features)
create_correlation_heatmap(correlation_matrix_imm_recur, 'Immediate Recurrence')
top_5_no_imm_recur = get_top_5_nodes(graph_imm_recur)
print("Top 5 nodes in graph_no_initial_res (based on degree centrality):")
for node in top_5_no_imm_recur:
    print(node)
imm_recur_node_measures = measure_node_features(graph_imm_recur)
calculate_influence_score(imm_recur_node_measures)
