"""

Digits Dataset
================

This digits example shows two ways of customizing the tooltips options in the HTML visualization. It generates the visualization with tooltips set as the y-label, or number of the image. The second generated result uses the actual image in the tooltips.

`Visualization with y-label tooltip <../../_static/digits_ylabel_tooltips.html>`_

`Visualization with custom tooltips <../../_static/digits_custom_tooltips.html>`_

"""

# sphinx_gallery_thumbnail_path = '../examples/digits/digits-tsne-custom-tooltip-mnist.png'

import io
import sys
import base64

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import kmapper as km
import pandas as pd

try:
    from PIL import Image
except ImportError as e:
    print("This example requires Pillow. Run `pip install pillow` and then try again.")
    sys.exit()

# Original Code
'''
# Load digits data
data, labels = datasets.load_digits().data, datasets.load_digits().target

# Raw data is (0, 16), so scale to 8 bits (pillow can't handle 4-bit greyscale PNG depth)
scaler = MinMaxScaler(feature_range=(0, 255))
data = scaler.fit_transform(data).astype(np.uint8)

# Create images for a custom tooltip array
tooltip_s = []
for image_data in data:
    with io.BytesIO() as output:
        img = Image.fromarray(image_data.reshape((8, 8)), "L")
        img.save(output, "PNG")
        contents = output.getvalue()
        img_encoded = base64.b64encode(contents)
        img_tag = """<img src="data:image/png;base64,{}">""".format(
            img_encoded.decode("utf-8")
        )
        tooltip_s.append(img_tag)

tooltip_s = np.array(
    tooltip_s
)  # need to make sure to feed it as a NumPy array, not a list

# Initialize to use t-SNE with 2 components (reduces data to 2 dimensions). Also note high overlap_percentage.
mapper = km.KeplerMapper(verbose=2)

# Fit and transform data
projected_data = mapper.fit_transform(data, projection=sklearn.manifold.TSNE())

# Create the graph (we cluster on the projected data and suffer projection loss)
graph = mapper.map(
    projected_data,
    clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),
    cover=km.Cover(35, 0.4),
)

# Create the visualizations (increased the graph_gravity for a tighter graph-look.)
print("Output graph examples to html")
# Tooltips with image data for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="examples\output\digits_custom_tooltips.html",
    color_values=labels,
    color_function_name="labels",
    custom_tooltips=tooltip_s,
)

# Tooltips with the target y-labels for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="examples\output\digits_ylabel_tooltips.html",
    custom_tooltips=labels,
) 

# Matplotlib examples
km.draw_matplotlib(graph, layout="spring")
plt.show()

'''
# Modified Code to get really big map

df = pd.read_csv("examples/data/embeddings.csv")
columns = [c for c in df.columns]
X = np.array(df[columns].fillna(0))  # quick and dirty imputation
# y = np.array(df["diagnosis"])

# Raw data is (0, 16), so scale to 8 bits (pillow can't handle 4-bit greyscale PNG depth)
scaler = MinMaxScaler(feature_range=(0, 255))
data = scaler.fit_transform(X).astype(np.float32)

# Initialize to use t-SNE with 2 components (reduces data to 2 dimensions). Also note high overlap_percentage.
mapper = km.KeplerMapper(verbose=2)

# Fit and transform data
projected_data = mapper.fit_transform(data, projection=sklearn.manifold.TSNE())

# Create the graph (we cluster on the projected data and suffer projection loss)
graph = mapper.map(
    projected_data,
    clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),
    cover=km.Cover(35, 0.4),
)

# Create the visualizations (increased the graph_gravity for a tighter graph-look.)
print("Output graph examples to html")

# Tooltips with image data for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="examples\output\digits_custom_tooltips.html",
    # color_values=labels,
    color_function_name="labels",
    # custom_tooltips=tooltip_s,
)

# Tooltips with the target y-labels for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="examples\output\digits_ylabel_tooltips.html",
    # custom_tooltips=labels,
) 

# Matplotlib examples
km.draw_matplotlib(graph, layout="spring")
plt.show()


# Next try doing the np.array type embeddings
'''
df = pd.read_csv("examples/data/node_embeddings.csv")
df = df.transpose()
columns = [c for c in df.columns]
X = np.array(df[columns].fillna(0))  # quick and dirty imputation
# y = np.array(df["diagnosis"])

# Raw data is (0, 16), so scale to 8 bits (pillow can't handle 4-bit greyscale PNG depth)
scaler = MinMaxScaler(feature_range=(0, 255))
data = scaler.fit_transform(X).astype(np.float32)

# Initialize to use t-SNE with 2 components (reduces data to 2 dimensions). Also note high overlap_percentage.
mapper = km.KeplerMapper(verbose=2)

# Fit and transform data
projected_data = mapper.fit_transform(data, projection=sklearn.manifold.TSNE())

# Create the graph (we cluster on the projected data and suffer projection loss)
graph = mapper.map(
    projected_data,
    clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),
    cover=km.Cover(35, 0.4),
)

# Create the visualizations (increased the graph_gravity for a tighter graph-look.)
print("Output graph examples to html")

# Tooltips with image data for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="examples\output\digits_custom_tooltips.html",
    # color_values=labels,
    color_function_name="labels",
    # custom_tooltips=tooltip_s,
)

# Tooltips with the target y-labels for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="examples\output\digits_ylabel_tooltips.html",
    # custom_tooltips=labels,
) 

# Matplotlib examples
km.draw_matplotlib(graph, layout="spring")
plt.show()
'''


# Took inspiration from the lions set
'''
# data, labels = datasets.load_digits().data, datasets.load_digits().target

# Raw data is (0, 16), so scale to 8 bits (pillow can't handle 4-bit greyscale PNG depth)
scaler = MinMaxScaler(feature_range=(0, 255))
data = np.genfromtxt("examples/data/test_embeddings.csv", delimiter=",")


# Initialize to use t-SNE with 2 components (reduces data to 2 dimensions). Also note high overlap_percentage.
mapper = km.KeplerMapper(verbose=2)

# Fit and transform data
projected_data = mapper.fit_transform(data, projection=sklearn.manifold.TSNE())

# Create the graph (we cluster on the projected data and suffer projection loss)
graph = mapper.map(
    projected_data,
    clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),
    cover=km.Cover(35, 0.4),
)

# Create the visualizations (increased the graph_gravity for a tighter graph-look.)
print("Output graph examples to html")
# Tooltips with image data for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="examples\output\digits_custom_tooltips.html",
    # color_values=labels,
    color_function_name="labels",
    # custom_tooltips=tooltip_s,
)


# Tooltips with the target y-labels for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="examples\output\digits_ylabel_tooltips.html",
    # custom_tooltips=labels,
) 

# Matplotlib examples
km.draw_matplotlib(graph, layout="spring")
plt.show()


'''

'''
data = pd.read_csv("examples\data\embeddings.csv")
labels = datasets.load_digits().target

# Raw data is (0, 16), so scale to 8 bits (pillow can't handle 4-bit greyscale PNG depth)
scaler = MinMaxScaler(feature_range=(0, 255))
data = scaler.fit_transform(data).astype(np.uint8)

# Initialize to use t-SNE with 2 components (reduces data to 2 dimensions). Also note high overlap_percentage.
mapper = km.KeplerMapper(verbose=2)

# Fit and transform data
projected_data = mapper.fit_transform(data, projection=sklearn.manifold.TSNE())

# Create the graph (we cluster on the projected data and suffer projection loss)
graph = mapper.map(
    projected_data,
    clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),
    cover=km.Cover(35, 0.4),
)

# Create the visualizations (increased the graph_gravity for a tighter graph-look.)
print("Output graph examples to html")
# Tooltips with image data for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="examples\output\digits_custom_tooltips.html",
    color_values=labels,
    color_function_name="labels",
    # custom_tooltips=tooltip_s,
)

# Tooltips with the target y-labels for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="examples\output\digits_ylabel_tooltips.html",
    custom_tooltips=labels,
) 

# Matplotlib examples
km.draw_matplotlib(graph, layout="spring")
plt.show()
'''
'''


data, labels = datasets.load_digits().data, datasets.load_digits().target

print(len(data))
print(len(labels))
myset = set(labels)
print(myset)

# Raw data is (0, 16), so scale to 8 bits (pillow can't handle 4-bit greyscale PNG depth)
scaler = MinMaxScaler(feature_range=(0, 255))
data = scaler.fit_transform(data).astype(np.uint8)

# Create images for a custom tooltip array
tooltip_s = []
for image_data in data:
    with io.BytesIO() as output:
        img = Image.fromarray(image_data.reshape((8, 8)), "L")
        img.save(output, "PNG")
        contents = output.getvalue()
        img_encoded = base64.b64encode(contents)
        img_tag = """<img src="data:image/png;base64,{}">""".format(
            img_encoded.decode("utf-8")
        )
        tooltip_s.append(img_tag)

tooltip_s = np.array(
    tooltip_s
)  # need to make sure to feed it as a NumPy array, not a list

# Initialize to use t-SNE with 2 components (reduces data to 2 dimensions). Also note high overlap_percentage.
mapper = km.KeplerMapper(verbose=2)

# Fit and transform data
projected_data = mapper.fit_transform(data, projection=sklearn.manifold.TSNE())

# Create the graph (we cluster on the projected data and suffer projection loss)
graph = mapper.map(
    projected_data,
    clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),
    cover=km.Cover(35, 0.4),
)

# Create the visualizations (increased the graph_gravity for a tighter graph-look.)
print("Output graph examples to html")
# Tooltips with image data for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="examples\output\digits_custom_tooltips.html",
    color_values=labels,
    color_function_name="labels",
    custom_tooltips=tooltip_s,
)

# Tooltips with the target y-labels for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="examples\output\digits_ylabel_tooltips.html",
    custom_tooltips=labels,
) 

# Matplotlib examples
km.draw_matplotlib(graph, layout="spring")
plt.show()
'''