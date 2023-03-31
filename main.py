from functions import *

img = cv.imread('images/sample/sample1.webp')
img2 = cv.imread('images/sample/sample2.webp')
img3 = cv.imread('images/sample/sample3.webp')

print(tf.__version__)

# Image gray
grey_image = img_grey(img)
# cv.imshow("Image Gray", grey_image)

# Blurring the image
blurred_image = blur(grey_image)
# cv.imshow("Image Blur", blurred_image)

# Canny image
canny_image = canny(blurred_image)

# thresholding image
# thresholding_image = thresholding(canny_image)

# sobel image
# sobel_image = sobel_detection(canny_image)

# laplacian image
laplacian_image = laplacian_detection(canny_image)

# region of interest
# roi_image = region_of_interest(laplacian_image)
