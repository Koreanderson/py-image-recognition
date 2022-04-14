from multiprocessing.spawn import prepare
from PIL import Image
import pytesseract
import pkg_resources
import cv2

#pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = '/usr/local/Cellar/tesseract/5.1.0/bin/tesseract'

# image_to_ocr = cv2.imread('images/letter.jpg')
# image_to_ocr = cv2.imread('images/not-working.jpg')
image_to_ocr = cv2.imread('images/numbers.png')

# Convert image to greyscale
preprocessed_img = cv2.cvtColor(image_to_ocr, cv2.COLOR_BGR2GRAY)

# Do binary and OTSU thresholding
preprocessed_img = cv2.threshold(preprocessed_img,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Smooth the image using median blur
preprocessed_img = cv2.medianBlur(preprocessed_img, 3)

# Save the preprocessed image temproarily into the disk
cv2.imwrite('temp_img.jpg', preprocessed_img)

# Read the temp image from disc as pil image
preprocessed_pil_img = Image.open('temp_img.jpg')

#pass the pil image to tesseract to do OCR


text_extracted = pytesseract.image_to_string(preprocessed_pil_img)

print('Extracted Text:',text_extracted)

#display original image

cv2.imshow("Actual Image", image_to_ocr)
