import sklearn
from sklearn.preprocessing import MinMaxScaler
import kmapper as km
from sklearn import datasets
import pandas as pd
from torch_geometric.datasets import CoraFull

# Code from Dr. Akcora
'''
# Creating PyTorch tda graph out of node features

def createTDAGraph(self, data, label, htmlPath="", timeWindow=0, network=""):
    try:
        per_overlap = [0.5]
        n_cubes = [5]
        cls = 2  # We use cls=5, but this parameter can be further refined.  Its impact on results seems minimal.
        Xfilt = data
        Xfilt = Xfilt.drop(columns=['nodeID'])
        mapper = km.KeplerMapper()
        scaler = MinMaxScaler(feature_range=(0, 1))

        Xfilt = scaler.fit_transform(Xfilt)
        # TSNE lens
        # lens = mapper.fit_transform(Xfilt, projection=sklearn.manifold.TSNE())
        # PCA lens
        lens = mapper.fit_transform(Xfilt, projection=PCA(n_components=2))

        for overlap in per_overlap:
            for n_cube in n_cubes:
                graph = mapper.map(
                    lens,
                    Xfilt,
                    clusterer=sklearn.cluster.KMeans(n_clusters=cls, random_state=1618033),
                    cover=km.Cover(n_cubes=n_cube, perc_overlap=overlap))  # 0.2 0.4


                # Creat a networkX graph for TDA mapper graph, in this graph nodes will be the clusters and the node featre would be the cluster size

                # removing al the nodes without any edges (Just looking at the links)
                tdaGraph = nx.Graph()
                for key, value in graph['links'].items():
                    tdaGraph.add_nodes_from([(key, {"cluster_size": len(graph["nodes"][key])})])
                    for to_add in value:
                        tdaGraph.add_nodes_from([(to_add, {"cluster_size": len(graph["nodes"][to_add])})])
                        tdaGraph.add_edge(key, to_add)

                # we have the tda Graph here
                # convert TDA graph to pytorch data
                directory = 'PygGraphs/TimeSeries/' + network + '/PCA_TDA_Tuned/Overlap_{}_Ncube_{}/'.format(
                    overlap,
                    n_cube)
                featureNames = ["cluster_size"]
                if not os.path.exists(directory):
                    os.makedirs(directory)
                pygData = self.from_networkx(tdaGraph, label=label, group_node_attrs=featureNames)
                with open(directory + "/" + network + "_" + "TDA_graph(cube-{},overlap-{})_".format(n_cube,
                                                                                                    overlap) + str(
                    timeWindow), 'wb') as f:
                    pickle.dump(pygData, f)


    except Exception as e:
        print(str(e))


# ========================================================================

# Extracting the node features

for item in selectedNetworkInGraphDataWindow.to_dict(orient="records"):
    from_node_features = {}
    to_node_features = {}
    # calculating node features for each edge
    # feature 1 -> sum of outgoing edge weights
    from_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['from']]

    try:
        to_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['to']]
    except Exception as e:
        to_node_features["outgoing_edge_weight_sum"] = 0

    # feature 2 -> sum of incoming edge weights
    to_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['to']]
    try:
        from_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['from']]
    except Exception as e:
        from_node_features["incoming_edge_weight_sum"] = 0
    # feature 3 -> number of outgoing edges
    from_node_features["outgoing_edge_count"] = outgoing_count[item['from']]
    try:
        to_node_features["outgoing_edge_count"] = outgoing_count[item['to']]
    except Exception as e:
        to_node_features["outgoing_edge_count"] = 0

    # feature 4 -> number of incoming edges
    to_node_features["incoming_edge_count"] = incoming_count[item['to']]
    try:
        from_node_features["incoming_edge_count"] = incoming_count[item['from']]
    except Exception as e:
        from_node_features["incoming_edge_count"] = 0

    # add temporal vector to all nodes, populated with -1

    from_node_features_with_daily_temporal_vector = dict(from_node_features)
    from_node_features_with_daily_temporal_vector["dailyClusterID"] = [-1] * windowSize
    from_node_features_with_daily_temporal_vector["dailyClusterSize"] = [-1] * windowSize

    to_node_features_with_daily_temporal_vector = dict(to_node_features)
    to_node_features_with_daily_temporal_vector["dailyClusterID"] = [-1] * windowSize
    to_node_features_with_daily_temporal_vector["dailyClusterSize"] = [-1] * windowSize



    transactionGraph.add_nodes_from([(item["from"], from_node_features)])
    transactionGraph.add_nodes_from([(item["to"], to_node_features)])
    transactionGraph.add_edge(item["from"], item["to"], value=item["value"])


    new_row = pd.DataFrame(({**{"nodeID": item["from"]}, **from_node_features}), index=[0])
    node_features = pd.concat([node_features, new_row], ignore_index=True)

    new_row = pd.DataFrame(({**{"nodeID": item["to"]}, **to_node_features}), index=[0])
    node_features = pd.concat([node_features, new_row], ignore_index=True)

    node_features = node_features.drop_duplicates(subset=['nodeID'])
'''

def createGraph(self, data, labels):
    try:
        Xfilt = data
        mapper = km.KeplerMapper()
        scaler = MinMaxScaler(feature_range=(0,1))
        Xfilt = scaler.fit_transform(Xfilt)

        lens = mapper.fit_transform(Xfilt, project=sklearn.manifold.TSNE())

    except Exception as e:
        print(str(e))


def createTDAGraph(self, data, label, htmlPath: str, timeWindow: int, network: str):
    try:
        per_overlap = [0.5]
        n_cubes = [5]
        cls = 2  # We use cls=5, but this parameter can be further refined.  Its impact on results seems minimal.
        Xfilt = data
        Xfilt = Xfilt.drop(columns=['nodeID'])
        mapper = km.KeplerMapper()
        scaler = MinMaxScaler(feature_range=(0, 1))

        Xfilt = scaler.fit_transform(Xfilt)
        # TSNE lens
        # lens = mapper.fit_transform(Xfilt, projection=sklearn.manifold.TSNE())
        # PCA lens
        lens = mapper.fit_transform(Xfilt, projection=PCA(n_components=2))

        for overlap in per_overlap:
            for n_cube in n_cubes:
                graph = mapper.map(
                    lens,
                    Xfilt,
                    clusterer=sklearn.cluster.KMeans(n_clusters=cls, random_state=1618033),
                    cover=km.Cover(n_cubes=n_cube, perc_overlap=overlap))  # 0.2 0.4


                # Creat a networkX graph for TDA mapper graph, in this graph nodes will be the clusters and the node featre would be the cluster size

                # removing al the nodes without any edges (Just looking at the links)
                tdaGraph = nx.Graph()
                # Adds nodes to a
                for key, value in graph['links'].items():
                    tdaGraph.add_nodes_from([(key, {"cluster_size": len(graph["nodes"][key])})]) 
                    for to_add in value:
                        tdaGraph.add_nodes_from([(to_add, {"cluster_size": len(graph["nodes"][to_add])})])
                        tdaGraph.add_edge(key, to_add)

    except Exception as e:
        print(str(e))


# ========================================================================

# Extracting the node features
def extractFeatures(selectedNetworkInGraphDataWindow: object, outgoing_weight_sum: list, incoming_weight_sum: list, outgoing_count: list, incoming_count: list):
    for item in selectedNetworkInGraphDataWindow.to_dict(orient="records"):
        from_node_features = {}
        to_node_features = {}
        # calculating node features for each edge
        # feature 1 -> sum of outgoing edge weights
        from_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['from']]

        try:
            to_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['to']]
        except Exception as e:
            to_node_features["outgoing_edge_weight_sum"] = 0

        # feature 2 -> sum of incoming edge weights
        to_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['to']]
        try:
            from_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['from']]
        except Exception as e:
            from_node_features["incoming_edge_weight_sum"] = 0
        # feature 3 -> number of outgoing edges
        from_node_features["outgoing_edge_count"] = outgoing_count[item['from']]
        try:
            to_node_features["outgoing_edge_count"] = outgoing_count[item['to']]
        except Exception as e:
            to_node_features["outgoing_edge_count"] = 0

        # feature 4 -> number of incoming edges
        to_node_features["incoming_edge_count"] = incoming_count[item['to']]
        try:
            from_node_features["incoming_edge_count"] = incoming_count[item['from']]
        except Exception as e:
            from_node_features["incoming_edge_count"] = 0

        # add temporal vector to all nodes, populated with -1

        from_node_features_with_daily_temporal_vector = dict(from_node_features)
        from_node_features_with_daily_temporal_vector["dailyClusterID"] = [-1] * windowSize
        from_node_features_with_daily_temporal_vector["dailyClusterSize"] = [-1] * windowSize

        to_node_features_with_daily_temporal_vector = dict(to_node_features)
        to_node_features_with_daily_temporal_vector["dailyClusterID"] = [-1] * windowSize
        to_node_features_with_daily_temporal_vector["dailyClusterSize"] = [-1] * windowSize



        transactionGraph.add_nodes_from([(item["from"], from_node_features)])
        transactionGraph.add_nodes_from([(item["to"], to_node_features)])
        transactionGraph.add_edge(item["from"], item["to"], value=item["value"])


        new_row = pd.DataFrame(({**{"nodeID": item["from"]}, **from_node_features}), index=[0])
        node_features = pd.concat([node_features, new_row], ignore_index=True)

        new_row = pd.DataFrame(({**{"nodeID": item["to"]}, **to_node_features}), index=[0])
        node_features = pd.concat([node_features, new_row], ignore_index=True)

        node_features = node_features.drop_duplicates(subset=['nodeID'])
