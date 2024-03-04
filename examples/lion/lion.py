import numpy as np
import sklearn
import kmapper as km


data = np.genfromtxt("examples/data/test_embeddings.csv", delimiter=",")
labels = np.genfromtxt("examples/data/node_labels.csv", delimiter=",")

# Initialize to use t-SNE with 2 components (reduces data to 2 dimensions). Also note high overlap_percentage.
mapper = km.KeplerMapper(verbose=2)
lens = mapper.fit_transform(data)

# Fit and transform data
projected_data = mapper.fit_transform(data, projection=sklearn.manifold.TSNE())

graph = mapper.map(
    data,
    lens
    clusterer=sklearn.cluster.Birch(),
    cover=km.Cover(n_cubes=10, perc_overlap=0.2),
)

mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="examples/output/digits_custom_tooltips.html",
    color_values=labels,
    color_function_name="labels",
    # custom_tooltips=tooltip_s,
)
# Tooltips with the target y-labels for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="examples/output/digits_ylabel_tooltips.html",
    custom_tooltips=labels,
)

'''
data = np.genfromtxt("lion-reference.csv", delimiter=",")

mapper = km.KeplerMapper(verbose=1)

lens = mapper.fit_transform(data)

graph = mapper.map(
    lens,
    data,
    clusterer=sklearn.cluster.BIRCH(eps=0.1, min_samples=5),
    cover=km.Cover(n_cubes=10, perc_overlap=0.2),
)

mapper.visualize(graph, path_html="lion_keplermapper_output.html")

# You may want to visualize the original point cloud data in 3D scatter too
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0],data[:,1],data[:,2])
plt.savefig("lion-reference.csv.png")
plt.show()
"""
'''


