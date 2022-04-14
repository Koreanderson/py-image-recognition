from PIL import Image
import pytesseract
import pkg_resources
import cv2

pytesseract.pytesseract.tesseract_cmd = '/usr/local/Cellar/tesseract/5.1.0/bin/tesseract'

print('Tesseract Version:')
print(pkg_resources.working_set.by_key['pytesseract'].version)

print('CV2 Version:')
print(cv2.__version__)