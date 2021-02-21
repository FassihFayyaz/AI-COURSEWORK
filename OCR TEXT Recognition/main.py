import cv2
import pytesseract

if __name__ == '__main__':

    '''if len(sys.argv) < 2:
        print('Usage: python ocr_simple.py image.jpg')
        sys.exit(1)'''

    # Read image path from command line
    #imPath = sys.argv[1]

    # Uncomment the line below to provide path to tesseract manually
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

    # Define config parameters.
    # '-l eng'  for using the English language
    # '--oem 1' for using LSTM OCR Engine
    # config = ('-l eng --oem 1 --psm 3')

    # Read image from disk
    im = cv2.imread('image1.jpg', cv2.IMREAD_COLOR)

    # Run tesseract OCR on image
    text = pytesseract.image_to_string(im)

    # Print recognized text
    print(text)