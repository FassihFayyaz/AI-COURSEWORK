import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
img = cv2.imread('image.jpg', cv2.IMREAD_COLOR)
print(pytesseract.image_to_string(img))