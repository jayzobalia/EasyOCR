import pytesseract
import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\Tesseract.exe'
reader = easyocr.Reader(['en'])

class preprocessing():
    def invert_img(self, img):
        inverted_image = cv2.bitwise_not(img)
        return inverted_image

    def binarization(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_ada = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 30)
        return img_ada

    def noise_removal(self,img):
        kernal = np.ones((1, 1), np.uint8)
        dilate = cv2.dilate(img, kernal, iterations=10)
        erode = cv2.erode(dilate, kernal, iterations=10)
        new_image = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernal, iterations=1)

        return new_image

img = cv2.imread('Images/img1.jpg')

obj = preprocessing()
inverted_img = obj.invert_img(img)
binarized_img = obj.binarization(img)
final_img = obj.noise_removal(binarized_img)

titles = ['Phase1', 'Phase2', 'Phase3','Phase4']
images = [img, inverted_img, binarized_img,final_img]

print("\n"+"*"*15 +"Text Using Tessaract" + "*"*15+"\n")
text_tessaract = pytesseract.image_to_string(final_img)
print(text_tessaract)
print("*"*15 +"Text Using Easy OCR" + "*"*15+"\n")
result_easyocr = reader.readtext(final_img)
print(text_tessaract)

for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

