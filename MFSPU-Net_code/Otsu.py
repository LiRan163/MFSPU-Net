import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

img_path = "C:\\Users\\HE CONG BING\\Desktop\\1.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Incorrect image path, please check if img_path is correct.")

hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
hist = hist / hist.sum()

bins = np.arange(256)
mu_T = (bins * hist).sum()

max_sigma = 0
best_T = 0

for T in range(256):
    w0 = hist[:T + 1].sum()
    w1 = hist[T + 1:].sum()

    if w0 == 0 or w1 == 0:
        continue

    mu0 = (bins[:T + 1] * hist[:T + 1]).sum() / w0
    mu1 = (bins[T + 1:] * hist[T + 1:]).sum() / w1

    sigma_b = w0 * (mu0 - mu_T) ** 2 + w1 * (mu1 - mu_T) ** 2

    if sigma_b > max_sigma:
        max_sigma = sigma_b
        best_T = T

print("Otsu optimal threshold =", best_T)

_, seg = cv2.threshold(img, best_T, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.plot(bins, hist, color='black')
plt.axvline(x=best_T, color='red', linestyle='--', label=f'T={best_T}')
plt.title("Histogram + Otsu Threshold")
plt.legend()

plt.subplot(1, 3, 3)
plt.imshow(seg, cmap='gray')
plt.title("Segmentation Result")
plt.axis("off")

plt.tight_layout()
plt.show()
