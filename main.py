from functions import *

img = cv.imread('images/sample/sample1.webp')

# Image gray
grey_image = img_grey(img)

# Blurring the image
blurred_image = blur(grey_image)

# Canny image
canny_image = canny(blurred_image)

# thresholding image
# thresholding_image = thresholding(canny_image)

# sobel image
# sobel_image = sobel_detection(canny_image)

# laplacian image
# laplacian_image = laplacian_detection(sobel_image)

# region of interest
# roi_image = region_of_interest(laplacian_image)


