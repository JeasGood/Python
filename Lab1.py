 {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n"
   ]
 } 
   {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n"
   ]
 } 
     {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n\n"
   ]
 } 
       {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "img1=cv2.imread("cat-vector.jpg",cv2.IMREAD_GRAYSCALE)\n"
"cv2.imwrite("cat-vector.jpg",img1)\n\n"
   ]
 } 
         {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "img2=cv2.imread("@WALLPAPERS.jpg",cv2.IMREAD_COLOR)\n",
"cv2.imwrite("@WALLPAPERS.jpg", img2)\n\n"
   ]
 } 
           {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 10))\n\n",
"plt.imshow(img1, cmap='gray')\n",
"plt.show();\n\n"
   ]
 } 
             {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "img_up = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)\n",
"plt.figure(figsize=(10, 10))\n\n",
"plt.imshow(img_up)\n",
"plt.show();\n\n"
   ]
 } 
      {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "fimg_up_to_upper = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)\n\n",
"plt.figure(figsize=(10, 10))\n",
"plt.imshow(fimg_up_to_upper)\n",
"plt.show();\n\n"
   ]
 } 
      {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 10))\n\n"
   ]
 } 
      {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.subplot(1,3,1);\n",
"R = np.zeros(img2.shape, dtype='uint8')\n",
"R[:, :, 0] = img2[:, :, 0]\n",
"plt.imshow(R)\n\n"
   ]
 } 
      {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.subplot(1,3,2);\n",
"G = np.zeros(img2.shape, dtype='uint8')\n",
"G[:, :, 1] = img2[:, :, 1]\n",
"plt.imshow(G)\n\n"
   ]
 } 
      {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.subplot(1,3,3);\n",
"B = np.zeros(img2.shape, dtype='uint8')\n",
"B[:, :, 2] = img2[:, :, 2]\n",
"plt.imshow(B)\n\n"
   ]
 } 
    {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.show()\n\n"
   ]
 }   
    {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 10))\n\n"
   ]
 }    
    {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.subplot(1,3,1);\n",
"H = np.zeros(fimg_up_to_upper.shape, dtype='uint8')\n",
"H[:, :, 0] = fimg_up_to_upper[:, :, 0]\n",
"plt.imshow(H)\n\n"
   ]
 }   
    {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.subplot(1,3,2);\n",
"S = np.zeros(fimg_up_to_upper.shape, dtype='uint8')\n",
"S[:, :, 1] = fimg_up_to_upper[:, :, 1]\n",
"plt.imshow(S)\n\n"
   ]
 }   
    {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.subplot(133);\n",
"V = np.zeros(fimg_up_to_upper.shape, dtype='uint8')\n",
"V[:, :, 2] = fimg_up_to_upper[:, :, 2]\n",
"plt.imshow(V)\n\n"
   ]
 }   
    {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.show()\n\n"
   ]
 }    
    {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "cp = img1.shape\n",
"x = cp[0]/2\n",
"y = cp[1]/2\n\n"
   ]
 } 
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "img1cp = img1[0 : int(x), 0 : int(y)]\n",
"plt.imshow(img1cp, cmap = "gray")\n",
"plt.show()\n\n"
   ]
 } 
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "cv2.imwrite("./img1cp.png", img1cp)\n"
   ]
 } 
