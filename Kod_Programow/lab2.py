import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("lab2.jpg")
print(image.shape)
new_image = image.copy()
# cv2.imshow("obraz", image)
# cv2.waitKey(0)

grey = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)
# cv2.imshow("obraz", grey)
# cv2.waitKey(0)

grey_contr = grey
min_grey = np.min(grey_contr)
max_grey = np.max(grey_contr)
delta = max_grey - min_grey

# rozciaganie kontrastu
for i in range(0, grey.shape[0]):
    for j in range(0, grey.shape[1]):
        grey_contr[i][j] = ((grey_contr[i][j] - min_grey)/delta)*255

grey_eq = cv2.equalizeHist(grey)

plt.figure(0)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
plt.subplot(3,2,1)
plt.imshow(grey, 'gray')

plt.subplot(3,2,2)
plt.hist(grey.ravel(),128,[0,256])
plt.title("Hist grey")

plt.subplot(3,2,3)
plt.imshow(grey_contr, 'gray')

plt.subplot(3,2,4)
plt.hist(grey_contr.ravel(),128,[0,256])
plt.title("Hist contrast")

plt.subplot(3,2,5)
plt.imshow(grey_eq, 'gray')

plt.subplot(3,2,6)
plt.hist(grey_eq.ravel(),128,[0,256])
plt.title("Hist EQ")
plt.show()


# RGB
b,g,r = cv2.split(new_image)
plt.figure(1)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
# RED
plt.subplot(3,2,1)
plt.imshow(r, 'gray')
plt.title("Red")

plt.subplot(3,2,2)
plt.hist(r.ravel(),256,[0,256])
plt.title("Red histogram")

# GREEN
plt.subplot(3,2,3)
plt.imshow(g, 'gray')
plt.title("Green")

plt.subplot(3,2,4)
plt.hist(g.ravel(),256,[0,256])
plt.title("Green histogram")

# BLUE
plt.subplot(3,2,5)
plt.imshow(b, 'gray')
plt.title("Blue")

plt.subplot(3,2,6)
plt.hist(b.ravel(),256,[0,256])
plt.title("Blue histogram")
plt.show()

# to HSV
hsv = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)
plt.figure(1)
plt.subplot(2,2,1)
plt.imshow(hsv[:,:,0], 'gray')
plt.title("Hue")

plt.subplot(2,2,2)
plt.imshow(hsv[:,:,1], 'gray')
plt.title("Saturation")

plt.subplot(2,2,3)
plt.imshow(hsv[:,:,2], 'gray')
plt.title("Value")
plt.show()

# flower = cv2.imread("flower.jpg")
# flower_hsv = cv2.cvtColor(flower, cv2.COLOR_BGR2HSV)

# only yellow
## mask o yellow (15,0,0) ~ (36, 255, 255)
mask = cv2.inRange(hsv, (15,0,0), (36, 255, 255))
target = cv2.bitwise_and(new_image,new_image,mask=mask)
cv2.imshow("yellow car", target[300:600,900:1350])
cv2.waitKey(0)





