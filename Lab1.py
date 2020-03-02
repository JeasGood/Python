import cv2 
import matplotlib.pyplot as plt
import numpy as np
img1=cv2.imread("cat-vector.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imwrite("cat-vector.jpg",img1)
img2=cv2.imread("@WALLPAPERS.jpg",cv2.IMREAD_COLOR)
cv2.imwrite("@WALLPAPERS.jpg", img2)
plt.figure(figsize=(10, 10))
plt.imshow(img1, cmap='gray')
plt.show();
img_up = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(img_up)
plt.show();
fimg_up_to_upper = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
plt.figure(figsize=(10, 10))
plt.imshow(fimg_up_to_upper)
plt.show();
plt.figure(figsize=(10, 10))

plt.subplot(131);
R = np.zeros(img2.shape, dtype='uint8')
R[:, :, 0] = img2[:, :, 0]
plt.imshow(R)

plt.subplot(132);
G = np.zeros(img2.shape, dtype='uint8')
G[:, :, 1] = img2[:, :, 1]
plt.imshow(G)

plt.subplot(133);
B = np.zeros(img2.shape, dtype='uint8')
B[:, :, 2] = img2[:, :, 2]
plt.imshow(B)

plt.show()

plt.figure(figsize=(10, 10))

plt.subplot(131);
H = np.zeros(fimg_up_to_upper.shape, dtype='uint8')
H[:, :, 0] = fimg_up_to_upper[:, :, 0]
plt.imshow(H)

plt.subplot(132);
S = np.zeros(fimg_up_to_upper.shape, dtype='uint8')
S[:, :, 1] = fimg_up_to_upper[:, :, 1]
plt.imshow(S)

plt.subplot(133);
V = np.zeros(fimg_up_to_upper.shape, dtype='uint8')
V[:, :, 2] = fimg_up_to_upper[:, :, 2]
plt.imshow(V)

plt.show()

cp = img1.shape
x = cp[0]/2
y = cp[1]/2

img1cp = img1[0 : int(x), 0 : int(y)]
plt.imshow(img1cp, cmap = "gray")
plt.show()

cv2.imwrite("./img1cp.png", img1cp)