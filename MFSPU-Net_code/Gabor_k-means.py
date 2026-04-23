import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img = cv2.imread("C:\\Users\\HE CONG BING\\Desktop\\1.jpg", cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

num_orientations = 4
num_scales = 2
filters = []
for theta in range(num_orientations):
    for sigma in (3, 5):
        kern = cv2.getGaborKernel((21, 21), sigma, theta * np.pi / num_orientations, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filters.append(kern)

gabor_responses = []
features = []
for kern in filters:
    fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
    gabor_responses.append(fimg)
    features.append(fimg.reshape(-1))

features = np.array(features).T

k = 3
kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
labels = kmeans.labels_.reshape(rows, cols)

plt.figure(figsize=(15, 8))

plt.subplot(2, num_orientations+1, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

for i, fimg in enumerate(gabor_responses):
    plt.subplot(2, num_orientations+1, i+2)
    plt.title(f"Gabor {i+1}")
    plt.imshow(fimg, cmap='gray')
    plt.axis('off')

plt.subplot(2, num_orientations+1, num_orientations+2)
plt.title("Segmentation")
plt.imshow(labels, cmap='nipy_spectral')
plt.axis('off')

plt.tight_layout()
plt.show()
