import numpy as np
import cv2
import matplotlib.pyplot as plt


# Thresholding
def thresholdingBinary(img):
    ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, th5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

    titles = ['original', 'binary', 'binary_inv', 'trunc', 'to_zero', 'to_zero_inv']
    images = [img, th1, th2, th3, th4, th5]

    for i in range(0, len(images)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])

    plt.show()

def thresholdingAdaptive(img):
    ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    titles = ['original', 'binary', 'adaptive mean', 'adaptive gaussian']
    images = [img, th1, th2, th3]

    for i in range(0, len(images)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])

    plt.show()

def thresholdingOtsu(img):
    ret, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(ret)
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.imshow(img, 'gray')
    plt.title("Obraz wejsciowy")

    plt.subplot(2, 2, 2)
    plt.hist(img.ravel(), 128, [0, 256])
    plt.title("Wartosc progu: %f" % ret)

    plt.subplot(2, 2, 3)
    plt.imshow(th1, 'gray')
    plt.title("Progowanie metoda Otsu")

    plt.subplot(2, 2, 4)
    plt.hist(th1.ravel(), 128, [0, 256])
    plt.title("Histogram po progowaniu")
    plt.show()

# Input images
image1 = cv2.imread("L3_obraz1.jpg")
image2 = cv2.imread("L3_obraz2.png")

# cv2.imshow("obraz", image1)
# cv2.waitKey(0)
# cv2.imshow("obraz", image2)
# cv2.waitKey(0)

imgGray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
imgGray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# call thresholding function
# thresholdingBinary(imgGray1)
# thresholdingBinary(imgGray2)

# cv2.ADAPTIVE_THRESH_MEAN_C : threshold value is the mean of neighbourhood area.
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C : threshold value is the weighted sum of neighbourhood values where weights are a gaussian window.
# thresholdingAdaptive(imgGray1)
# thresholdingAdaptive(imgGray2)

# Otsu method
# thresholdingOtsu(imgGray2)


# Clustering
# define criteria, number of clusters(K) and apply kmeans()
img = cv2.imread("L3_obraz2.png")
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()