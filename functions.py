import cv2 as cv
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# ------------------ Test Print ------------------
def test_print():
    print("Test Print")


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
    image_canny = cv.Canny(image_blur, 90, 120)
    image_canny2 = cv.Canny(image_blur, 0, 299)
    # image_canny3 = cv.Canny(image_blur, 35, 210)
    # image_canny4 = cv.Canny(image_blur, 55, 210)
    # image_canny5 = cv.Canny(image_blur, 45, 210)

    # cv.imshow("Image Canny 90 x 120", image_canny)
    cv.imshow("Image Canny 0 x 299", image_canny2)
    # cv.imshow("Image Canny 35 x 210", image_canny3)
    # cv.imshow("Image Canny 55 x 210", image_canny4)
    # cv.imshow("Image Canny 45 x 210", image_canny5)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return image_canny


# -------------------- Sobel ----------------------

def sobel_detection(img):
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)

    combined_sobel = cv.bitwise_or(sobelx, sobely)

    cv.imshow("Image Sobel", combined_sobel)
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
    thresh = cv.adaptiveThreshold(img, 256, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, 2)

    cv.imshow("Image Thresholding", thresh)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return thresh


# ------------------ Region of Interest ------------------

# ------------------- length of a line -------------------
def length_of_line(x1, y1, x2, y2):
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    print('this should be updated')
    return length


# --------------------- Tensor Flow --------------------------
def tensor():
    # Class names
    class_names = ['Heavy Scratch', 'Light Scratch', 'No Scratch']

    # Train image
    train_image = cv.imread('images/audi/1.jpg')
    print(type(train_image))
    print(train_image.shape)
    print(type(train_image.shape))


tensor()
