from sklearn import cluster
from sklearn.cluster import MiniBatchKMeans
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io

batch_size = 1000
clusters=64

# 
def getImages(str):
    mat = io.ImageCollection(str) # read a collection of images
    print('pictures numbers: %d' % len(mat))
    return mat

# get all imgs SIFT to construct training database
def getSIFT(mat):
    ordict = []
    siftt = cv2.xfeatures2d.SIFT_create()
    for i in range(len(mat)):
        kp, des = siftt.detectAndCompute(mat[i], None)  # des.shape= (len(kp),128)
        ordict.append(des)
    return ordict

# get a img SIFT 
def img2SIFT(img):
    mat = cv2.imread(img)
    siftt = cv2.xfeatures2d.SIFT_create()
    kp, des = siftt.detectAndCompute(mat, None)
    return des

# use all extracted features to train a codebook
def trainCodeBook(features):
    database = []
    for i in range(len(features)):
        for j in range(len(features[i])):
            database.append(features[i][j])
    database = np.array(database)

    kmeans = MiniBatchKMeans(init='k-means++', n_clusters=clusters, max_iter=1000,
                             batch_size=batch_size, n_init=10, max_no_improvement=10, verbose=0).fit(database)
    label = kmeans.labels_
    centroid = kmeans.cluster_centers_
    return kmeans, label, centroid


def vlad(kmeans, locdes, centroid):
    # assign the sift vector to corrspended centroid by the nearest neighbor principle
    labels = kmeans.predict(locdes)
    des=[]
    for i in range(clusters):
        des.append(np.zeros(shape=(1,128)))
        matched = np.argwhere(labels == i)
        if len(matched) == 0:
            des[i] = np.zeros_like(centroid[0])
        else:
            # calculate the sum of subtraction and l2 normlization
            for idx in matched:
                des[i] += (locdes[idx] - centroid[i])
            des[i] /= np.linalg.norm(des[i], ord=2)
    # reshape to a single vector
    des = np.array(des).reshape(1,-1)
    return des


if __name__ == "__main__":
    folder = './src/*.jpeg'
    # train codebook
    imgs = getImages(folder)
    features = getSIFT(imgs)
    kmeans, _, centroid = trainCodeBook(features)

    # get a img VLAD feature
    des = img2SIFT('./src/image1.jpeg')
    ans = vlad(kmeans, des, centroid)

    
