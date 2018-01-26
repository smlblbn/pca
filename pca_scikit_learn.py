import numpy as np
from sklearn.decomposition import PCA

def show(img):
	imgplot = plt.imshow(img.reshape(231,195), cmap='gray')
	plt.show()

dataset = np.load('FaceImages.npy')
k = 16
mean = np.mean(dataset,axis=0)
std = np.std(dataset,axis=0)
image = dataset[0,:]

pca = PCA(n_components=k)
pca.fit(dataset)
x = pca.transform(image.reshape(1,-1))
y = pca.inverse_transform(x)
recon = np.dot(pca.components_.T,x.T)

def pca(x):
	x = (x - x.mean(axis = 0))/np.std(dataset,axis=0)
	
	num_observations, num_dimensions = x.shape
	if num_dimensions > 100:
		eigenvalues, eigenvectors = np.linalg.eigh(np.dot(x, x.T))
		v = (np.dot(x.T, eigenvectors).T)[::-1]
		s = np.sqrt(eigenvalues)[::-1]
	else:
		u, s, v = np.linalg.svd(x, full_matrices = False)
		
	return v, s

kk = pca(dataset)

print(kk)

show(y)
show(np.squeeze(recon)+pca.mean_)
