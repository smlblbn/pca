import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show(img):
	imgplot = plt.imshow(img.reshape(231,195), cmap='gray')
	plt.show()

def showEigenface(ver, hor, X):
	L = []
	R = []

	for i in range(ver):
		for j in range(hor):
			L.append(X[i*hor+j,:].reshape(231,195))
		R.append(np.hstack(L))
		L = []

	A = np.vstack(R)

	imgplot = plt.imshow(A, cmap='gray')
	plt.show()

def findPrincipalComponents(dataset,k):

	mean = np.mean(dataset,axis=0)
	std = np.std(dataset,axis=0)

	dataset2 = (dataset-mean)/std

	cov = np.dot(dataset2, dataset2.T)
	e,v = np.linalg.eigh(cov)

	# sort the eigenvalue and same time eigen vector and choose them
	idx = np.argsort(e)[::-1]
	v = v[:,idx]
	e = e[idx]
	v = v[:, :k]

	u = np.dot(dataset2.T, v)

	return u

def pcaCompress(image,component):
	c = np.dot(component.T, image)
	q = np.dot(component,c)

	return q

dataset = np.load('FaceImages.npy')
image = [dataset[0,:],dataset[1,:]]
karray = [16,32,64]

for i in range(len(image)):
	for j in range(len(karray)):
		pca = findPrincipalComponents(dataset,karray[j])

		precon = pcaCompress(image[i],pca)

		show(image[i])
		show(precon)

		if karray[j] == 16:
			showEigenface(4,4,pca.T)
		elif karray[j] == 32:
			showEigenface(4,8,pca.T)
		elif karray[j] == 64:
			showEigenface(8,8,pca.T)

