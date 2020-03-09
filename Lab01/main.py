import cv2
import numpy as np

image = cv2.imread("L1_image.jpg")
print(image.shape)
new_image = image.copy()
cv2.imshow("obraz", image)
cv2.waitKey(0)

(B, G, R) = image[70, 10]
print(B, G, R)

count = np.unique(image.reshape(-1, image.shape[2]), axis=0)

print(len(count))

# Ekstakcja twarzy
face = image[40:70, 100:120]
cv2.imshow("Face", face)
cv2.waitKey(0)

# Zmiana rozmiaru
# obrazka o 50%
im_x = image.shape[1]
im_y = image.shape[0]
resized = cv2.resize(image, (int(im_x/2) , int(im_y/2)))
cv2.imshow("Resized photo", resized)
cv2.waitKey(0)

# Obrot obrazu o 30 stopni
middle = int(im_x/2), int(im_y/2)
rotate_matrix = cv2.getRotationMatrix2D(middle, 30, 1.0)

abs_cos = abs(rotate_matrix[0,0])
abs_sin = abs(rotate_matrix[0,1])

# find the new width and height bounds
bound_w = int(im_y * abs_sin + im_x * abs_cos)
bound_h = int(im_x * abs_cos + im_y * abs_sin)

im_x_new = int(bound_w / 2) - im_x
im_y_new = int(bound_h / 2) - im_y

rotated_image = cv2.warpAffine(image, rotate_matrix, (im_x_new, im_y_new))

cv2.imshow("Rotated image", rotated_image)
cv2.waitKey(0)

# Oznaczenie twarzy na obrazie
start_point = (100, 40)
end_point = (120, 70)
# color of line
color = (0, 0, 255)
# thickness
thickness = 2
# image with borders
image_border = cv2.rectangle(image, start_point, end_point, color, thickness)

# TEXT
org = (80, 80)
fontScale = 0.5
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# color BGR
color = (0, 0, 255)
image_border_text = cv2.putText(image_border, "Tom Smith", org, font, fontScale, color)
cv2.imshow("Photo with border on the face", image_border_text)
cv2.waitKey(0)

# Zapisywanie obrazu
cv2.imwrite("Obrazek.jpg", image_border_text)

# To GRAY
image2 = cv2.cvtColor(image_border_text, cv2.COLOR_BGR2GRAY)
# Zapisywanie obrazu
cv2.imwrite("Obrazek_szary.jpg", image2)









