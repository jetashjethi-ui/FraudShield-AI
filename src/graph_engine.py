"""
FraudShield AI — Graph Analysis Engine (Layer 16)
Temporal graph-based fraud ring detection using NetworkX.
Builds a transaction relationship graph and extracts graph-level features
per user: centrality, PageRank, community membership, fraud neighbor scores.
"""

import pandas as pd
import numpy as np
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

try:
    from community import community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False


def build_graph_features(df):
    """Build transaction graph and extract features per user (card1)."""
    print("\n  [Layer 16] Temporal Graph Analysis (Fraud Ring Detection)...")

    users = df['card1'].unique()
    n_users = len(users)
    print(f"    -> {n_users:,} unique users (card1 values)")

    # For very large datasets, sample to keep graph manageable
    MAX_USERS = 50000
    if n_users > MAX_USERS:
        print(f"    -> Sampling {MAX_USERS:,} users for graph construction...")
        # Keep all fraud users + random sample of legitimate
        fraud_users = set(df[df['isFraud'] == 1]['card1'].unique())
        legit_users = set(df[df['isFraud'] == 0]['card1'].unique()) - fraud_users
        rng = np.random.RandomState(42)
        sampled_legit = set(rng.choice(list(legit_users),
                           min(MAX_USERS - len(fraud_users), len(legit_users)),
                           replace=False))
        graph_users = fraud_users | sampled_legit
        df_graph = df[df['card1'].isin(graph_users)].copy()
    else:
        df_graph = df.copy()
        graph_users = set(users)

    # ── Build the graph ──────────────────────────────────────────
    G = nx.Graph()
    G.add_nodes_from(df_graph['card1'].unique())

    # Edge type 1: Shared device
    if 'DeviceInfo' in df_graph.columns:
        device_groups = df_graph.dropna(subset=['DeviceInfo']).groupby('DeviceInfo')['card1'].apply(set)
        n_device_edges = 0
        for device, card_set in device_groups.items():
            card_list = list(card_set)
            if 2 <= len(card_list) <= 20:  # Skip very common devices (noisy)
                for i in range(len(card_list)):
                    for j in range(i + 1, min(len(card_list), i + 5)):  # Limit edges per device
                        G.add_edge(card_list[i], card_list[j], relation='shared_device')
                        n_device_edges += 1
        print(f"    -> {n_device_edges:,} edges from shared devices")

    # Edge type 2: Shared address
    if 'addr1' in df_graph.columns:
        addr_groups = df_graph.dropna(subset=['addr1']).groupby('addr1')['card1'].apply(set)
        n_addr_edges = 0
        for addr, card_set in addr_groups.items():
            card_list = list(card_set)
            if 2 <= len(card_list) <= 15:
                for i in range(len(card_list)):
                    for j in range(i + 1, min(len(card_list), i + 5)):
                        if not G.has_edge(card_list[i], card_list[j]):
                            G.add_edge(card_list[i], card_list[j], relation='shared_address')
                            n_addr_edges += 1
        print(f"    -> {n_addr_edges:,} edges from shared addresses")

    # Edge type 3: Shared email domain (only uncommon domains)
    if 'P_emaildomain' in df_graph.columns:
        email_counts = df_graph['P_emaildomain'].value_counts()
        # Only use uncommon email domains (not gmail, yahoo, etc.)
        rare_domains = email_counts[(email_counts >= 2) & (email_counts <= 100)].index
        rare_email_df = df_graph[df_graph['P_emaildomain'].isin(rare_domains)]
        email_groups = rare_email_df.groupby('P_emaildomain')['card1'].apply(set)
        n_email_edges = 0
        for domain, card_set in email_groups.items():
            card_list = list(card_set)
            if 2 <= len(card_list) <= 10:
                for i in range(len(card_list)):
                    for j in range(i + 1, min(len(card_list), i + 3)):
                        if not G.has_edge(card_list[i], card_list[j]):
                            G.add_edge(card_list[i], card_list[j], relation='shared_email')
                            n_email_edges += 1
        print(f"    -> {n_email_edges:,} edges from shared email domains")

    print(f"    -> Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # ── Extract graph features ───────────────────────────────────
    print("    -> Computing graph metrics...")

    # Degree centrality
    degree_cent = nx.degree_centrality(G)

    # PageRank
    try:
        pagerank = nx.pagerank(G, max_iter=50, tol=1e-4)
    except Exception:
        pagerank = {n: 1.0 / G.number_of_nodes() for n in G.nodes()}

    # Clustering coefficient
    clustering = nx.clustering(G)

    # Community detection (Louvain)
    if HAS_LOUVAIN and G.number_of_edges() > 0:
        print("    -> Running Louvain community detection...")
        partition = community_louvain.best_partition(G, random_state=42)
    else:
        # Fallback: connected components as communities
        partition = {}
        for i, comp in enumerate(nx.connected_components(G)):
            for node in comp:
                partition[node] = i

    # Community sizes
    from collections import Counter
    comm_sizes = Counter(partition.values())

    # Fraud labels per user (for computing community fraud rate)
    user_fraud = df_graph.groupby('card1')['isFraud'].max().to_dict()

    # Community fraud rates
    comm_fraud = {}
    comm_members = {}
    for node, comm_id in partition.items():
        if comm_id not in comm_members:
            comm_members[comm_id] = []
        comm_members[comm_id].append(node)

    for comm_id, members in comm_members.items():
        fraud_count = sum(1 for m in members if user_fraud.get(m, 0) == 1)
        comm_fraud[comm_id] = fraud_count / max(len(members), 1)

    # Fraud neighbor features
    fraud_neighbor_count = {}
    fraud_neighbor_ratio = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) == 0:
            fraud_neighbor_count[node] = 0
            fraud_neighbor_ratio[node] = 0
        else:
            fn_count = sum(1 for n in neighbors if user_fraud.get(n, 0) == 1)
            fraud_neighbor_count[node] = fn_count
            fraud_neighbor_ratio[node] = fn_count / len(neighbors)

    # Bridge detection (betweenness on sampled graph for speed)
    if G.number_of_nodes() < 10000:
        betweenness = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
    else:
        betweenness = nx.betweenness_centrality(G, k=100)

    # ── Build feature dataframe ──────────────────────────────────
    graph_df = pd.DataFrame({
        'card1': list(G.nodes()),
        'graph_degree_centrality': [degree_cent.get(n, 0) for n in G.nodes()],
        'graph_pagerank': [pagerank.get(n, 0) for n in G.nodes()],
        'graph_clustering': [clustering.get(n, 0) for n in G.nodes()],
        'graph_community_id': [partition.get(n, -1) for n in G.nodes()],
        'graph_community_size': [comm_sizes.get(partition.get(n, -1), 1) for n in G.nodes()],
        'graph_community_fraud_rate': [comm_fraud.get(partition.get(n, -1), 0) for n in G.nodes()],
        'graph_fraud_neighbor_count': [fraud_neighbor_count.get(n, 0) for n in G.nodes()],
        'graph_fraud_neighbor_ratio': [fraud_neighbor_ratio.get(n, 0) for n in G.nodes()],
        'graph_betweenness': [betweenness.get(n, 0) for n in G.nodes()],
        'graph_is_bridge': [(betweenness.get(n, 0) > np.percentile(list(betweenness.values()), 95))
                            for n in G.nodes()],
    })
    graph_df['graph_is_bridge'] = graph_df['graph_is_bridge'].astype(int)

    # Merge back into main df
    df = df.merge(graph_df, on='card1', how='left')

    # Fill NaN for users not in the graph
    graph_cols = [c for c in df.columns if c.startswith('graph_')]
    for col in graph_cols:
        df[col] = df[col].fillna(0)

    n_features = len(graph_cols)
    print(f"    -> {n_features} graph features created")
    print(f"    -> Communities found: {len(set(partition.values()))}")
    high_fraud_comms = sum(1 for v in comm_fraud.values() if v > 0.3)
    print(f"    -> High-fraud communities (>30% fraud rate): {high_fraud_comms}")

    return df
