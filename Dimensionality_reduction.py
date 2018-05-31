import tempfile
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import MDS
test_data_home = tempfile.mkdtemp()
mnist = fetch_mldata('MNIST original', data_home=test_data_home)
X, y = mnist["data"], mnist["target"]
data=mnist.data[::30]
target=mnist.target[::30]
print(data)
model=Isomap(n_components=2)
ISO=model.fit_transform(data)
plt.scatter(ISO[:,0],ISO[:,1],c=target, cmap=plt.cm.get_cmap('jet',10))
plt.colorbar(ticks=range(10))
plt.title('ISO MAP')
plt.show()
LLE = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
x_min = LLE.fit_transform(data)
plt.scatter(x_min[:,0],x_min[:,1],c=target, cmap=plt.cm.get_cmap('jet',10))
plt.title('Locally Linear Embedding')
plt.show()
LLEM = LocallyLinearEmbedding(n_components=2, n_neighbors=10,method='modified')
x_min_m = LLEM.fit_transform(data)
plt.scatter(x_min_m[:,0],x_min_m[:,1],c=target, cmap=plt.cm.get_cmap('jet',10))
plt.title('Modified Locally Linear Embedding ')
plt.show()
T=TSNE(n_components=2)
x_T=T.fit_transform(data)
plt.scatter(x_T[:,0],x_T[:,1],c =target, cmap=plt.cm.get_cmap('jet',10))
plt.title('TSNE')
plt.show()
Spectral = SpectralEmbedding(n_components=2,affinity='nearest_neighbors')
X_se=Spectral.fit_transform(data)
plt.scatter(X_se[:,0],X_se[:,1],c =target, cmap=plt.cm.get_cmap('jet',10))
plt.title('Spectral Embedding')
plt.show()
