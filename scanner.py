from imutils.perspective import four_point_transform
import cv2
import numpy as np

height = 800
width = 600
green = (0, 255, 0)

#image = cv2.imread("../CamScanner-Python/images/receipt.jpg")
image = cv2.imread("test-case2.jpg")
image = cv2.resize(image, (width, height))
orig_image = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert the image to gray scale
blur = cv2.GaussianBlur(gray, (5, 5), 0) # Add Gaussian blur
edged = cv2.Canny(blur, 75, 225) # Apply the Canny algorithm to find the edges

# Show the image and the edges
cv2.imshow('Original image:', image)
cv2.imshow('Edged:', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Show the image and all the contours
cv2.imshow("Image", image)
cv2.drawContours(image, contours, -1, green, 3)
cv2.imshow("All contours", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# go through each contour

for i, contour in enumerate(contours):
    # we approximate the contour
    peri = cv2.arcLength(contour, False)
    print(peri)
    print(i)
    approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
    """
    cv2.drawContours(orig_image, [approx], -1, green, 3)
    cv2.imshow("Contours of the document", orig_image)
    cv2.waitKey(0)
    """
    # if we found a countour with 4 points we break the for loop
    # (we can assume that we have found our document)    
    if len(approx) == 4:
        doc_cnts = approx
        break

"""
contour = max(contours, key = cv2.contourArea)
peri = cv2.arcLength(contour, False)
doc_cnts = cv2.approxPolyDP(contour, 0.05 * peri, True)
"""


# We draw the contours on the original image not the modified one
cv2.drawContours(orig_image, [doc_cnts], -1, green, 3)
cv2.imshow("Contours of the document", orig_image)
cv2.waitKey(0)
# apply warp perspective to get the top-down view
warped = four_point_transform(orig_image, doc_cnts.reshape(4, 2))
# convert the warped image to grayscale
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Scanned", cv2.resize(warped, (600, 800)))
#cv2.waitKey(0)
warped = cv2.resize(warped, (600, 800))
final = cv2.imwrite('result2.png',warped)
cv2.destroyAllWindows()
