import cv2 as cv
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


# ------------------ Image Gray ------------------

def img_grey(image):
    image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_grey = cv.resize(image_grey, (900, 600))

    # Histogram
    # plt.hist(image_grey.ravel(), bins=256, range=(0, 230))
    # plt.title("Grayscale Histogram")
    # plt.xlabel("grayscale value")
    # plt.ylabel("pixels")
    # plt.show()

    # cv.imshow("Image Gray", image_grey)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return image_grey


# ------------------ Image Blur ------------------

def blur(image):
    image_blur = cv.GaussianBlur(image, (7, 7), 0)

    # cv.imshow("Image Blur", image_blur)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return image_blur


# ------------------ Canny ------------------

def canny(image_blur):
    image_canny = cv.Canny(image_blur, 150, 200)

    cv.imshow("Image Canny", image_canny)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return image_canny


# -------------------- Sobel ----------------------

def sobel_detection(img):
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
    cv.imshow("Image Sobel", sobelx)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return sobely


# -------------- Laplacian Edge Detection --------------
def laplacian_detection(img):
    laplacian = cv.Laplacian(img, cv.CV_64F)
    cv.imshow("Image Laplacian", laplacian)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return laplacian


# ------------------ Thresholding ------------------
def thresholding(img):
    thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    cv.imshow("Image Thresholding", thresh)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return thresh
# ------------------ Region of Interest ------------------
# def region_of_interest(img):
#     height = img.shape[0]
#     polygons = np.array([
#         [(200, height), (1100, height), (550, 250)]
#     ])
#     mask = np.zeros_like(img)
#     cv.fillPoly(mask, polygons, 255)
#
#     masked_image = cv.bitwise_and(img, mask)
#     cv.imshow("Image ROI", masked_image)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
#     return masked_image
