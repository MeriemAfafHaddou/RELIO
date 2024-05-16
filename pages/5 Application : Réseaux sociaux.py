import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import pandas as pd
import time
from IPython.display import display, clear_output
import streamlit as st
import plotly.graph_objs as go

st.write("""
# Application : Détection de drift dans les réseaux sociaux
""")


s0 = {'index': 0, 'snapshot': pd.read_csv('data/graphes/Digg_snapshot_0.edgelist.txt', sep=' ', header=None).iloc[:, :2]}
s1 = {'index': 1, 'snapshot': pd.read_csv('data/graphes/Digg_snapshot_1.edgelist.txt', sep=' ', header=None).iloc[:, :2]}
s2 = {'index': 2, 'snapshot': pd.read_csv('data/graphes/Digg_snapshot_2.edgelist.txt', sep=' ', header=None).iloc[:, :2]}
s3 = {'index': 3, 'snapshot': pd.read_csv('data/graphes/Digg_snapshot_3.edgelist.txt', sep=' ', header=None).iloc[:, :2]}
s4 = {'index': 4, 'snapshot': pd.read_csv('data/graphes/Digg_snapshot_4.edgelist.txt', sep=' ', header=None).iloc[:, :2]}
s5 = {'index': 5, 'snapshot': pd.read_csv('data/graphes/Digg_snapshot_5.edgelist.txt', sep=' ', header=None).iloc[:, :2]}
s6 = {'index': 6, 'snapshot': pd.read_csv('data/graphes/Digg_snapshot_6.edgelist.txt', sep=' ', header=None).iloc[:, :2]}
s7 = {'index': 7, 'snapshot': pd.read_csv('data/graphes/Digg_snapshot_7.edgelist.txt', sep=' ', header=None).iloc[:, :2]}
s8 = {'index': 8, 'snapshot': pd.read_csv('data/graphes/Digg_snapshot_8.edgelist.txt', sep=' ', header=None).iloc[:, :2]}
s9 = {'index': 9, 'snapshot': pd.read_csv('data/graphes/Digg_snapshot_9.edgelist.txt', sep=' ', header=None).iloc[:, :2]}
for snapshot in [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]:
    snapshot['snapshot'].columns = ['source', 'target']
# Load data
snapshots = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]  # Replace with your actual snapshots

df = s1['snapshot'].sample(1000)
num_rows = len(df)
num_rows_per_slider = 200

# Create a slider to select the starting row
start_row = st.slider("Select starting row", 100, num_rows, 400, num_rows_per_slider)

# Select the rows based on the slider value
selected_rows = df.iloc[:start_row]


nodes = []
edges = []

# Add nodes
for node in selected_rows["source"].unique():
    nodes.append(Node(id=str(node), label=str(node), size=10))

# Add nodes that are in target and not in source
for node in selected_rows["target"].unique():
    if node not in selected_rows["source"].unique():
        nodes.append(Node(id=str(node), label=str(node), size=10))

# Add edges
for index, row in selected_rows.iterrows():
    edges.append(Edge(source=str(row['source']), target=str(row['target'])))

config = Config(width=750,
                height=950,
                directed=True, 
                physics=True, 
                hierarchical=False)

return_value = agraph(nodes=nodes, 
                      edges=edges, 
                      config=config)

# A = list(selected_rows["source"].unique())
# B = list(selected_rows["target"].unique())
# node_list = set(A + B)
# G = nx.Graph()

# # Add nodes and edges to the graph object
# for i in node_list:
#     G.add_node(i)
# for i, j in selected_rows.iterrows():
#     G.add_edges_from([(j["source"], j["target"])])

# pos = nx.spring_layout(G, k=0.2, iterations=10)
# # Add positions of nodes to the graph
# for n, p in pos.items():
#     G._node[n]['pos'] = p

# edge_trace = go.Scatter(
#     x=[],
#     y=[],
#     line=dict(width=1, color='#888'),
#     hoverinfo='none',
#     mode='lines')

# for edge in G.edges():
#     x0, y0 = G._node[edge[0]]['pos']
#     x1, y1 = G._node[edge[1]]['pos']
#     edge_trace['x'] += tuple([x0, x1, None])
#     edge_trace['y'] += tuple([y0, y1, None])

# node_trace = go.Scatter(
#     x=[],
#     y=[],
#     text=[],
#     mode='markers',
#     hoverinfo='text',
#     marker=dict(
#         showscale=True,
#         colorscale='purp',
#         color=[],
#         size=20,
#         colorbar=dict(
#             thickness=10,
#             title='# Connections',
#             xanchor='left',
#             titleside='right',
#         ),
#         line=dict(width=0)))

# for node in G.nodes():
#     x, y = G._node[node]['pos']
#     node_trace['x'] += tuple([x])
#     node_trace['y'] += tuple([y])

# for node, adjacencies in enumerate(G.adjacency()):
#     node_trace['marker']['color'] += tuple([len(adjacencies[1])])
#     node_info = str(adjacencies[0]) + ' # of connections: ' + str(len(adjacencies[1]))
#     node_trace['text'] += tuple([node_info])

# fig = go.Figure(data=[edge_trace, node_trace],
#                 layout=go.Layout(
#                     title=f"Visualisation du Snapshot numéro {1}",
#                     titlefont=dict(size=25),
#                     showlegend=False,
#                     hovermode='closest',
#                     margin=dict(b=20, l=5, r=5, t=40),
#                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
# fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')),
#                   selector=dict(mode='markers'))

# st.plotly_chart(fig, use_container_width=True)
