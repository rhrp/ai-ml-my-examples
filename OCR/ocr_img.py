"""
python -m pip install pytesseract

#base
python -m pip install opencv-python
#includes opencv-python and other contributes
python -m pip install opencv-contrib-python

#also requires tesseract-ocr binar
apt install tesseract-ocr


You also can run outside python, at shell
tesseract image.png output.txt -l eng
"""

from PIL import Image
import pytesseract
import argparse
import cv2, os

# # parse the argument
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required = True)
parser.add_argument("-p", "--preprocess", type = str, default = "thresh")
args = vars(parser.parse_args())
inputfile=args["image"]

# load the example image and convert it to grayscale
image = cv2.imread(inputfile)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# check preprocess to apply thresholding on the image
if args["preprocess"] == "thresh":
    gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    print('Apply thresh to ',inputfile) 
elif args["preprocess"] == "blur":
    gray = cv2.medianBlur(gray, 3)
    print('Apply blur to ',inputfile) 

# write the grayscale image to disk as a temporary file
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

# load the image as a PIL/Pillow image
# apply OCR
# delete temp image
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)


 
# show the output images - blocks sometimes
#cv2.imshow("Image", image)
#cv2.imshow("Output", gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#TO-DO : Additional processing such as spellchecking for OCR errors or NLP 
print(text)

