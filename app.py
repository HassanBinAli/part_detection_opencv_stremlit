# importing dependencies
import cv2
from utils import imageread, get_contours
import numpy as np
import streamlit as st


img, img1, img2, img3 = imageread()
img_list = [img, img1, img2, img3]
square_area = []
circle_area = []

for imgg in img_list:
    sq_area, cir_area = get_contours(imgg)
    square_area.append(sq_area)
    circle_area.append(cir_area)


st.subheader("Upload Part")
test_image = st.file_uploader('Part', type=['jpg', 'png','jpeg'] )
col1, col2 = st.columns(2)
if test_image is not None:
    # Convert the file read to the bytes array.
    file_bytes_fundus = np.asarray(bytearray(test_image.read()), dtype=np.uint8)
    # Converting the byte array into opencv image. 0 for grayscale and 1 for bgr
    test_image_decoded = cv2.imdecode(file_bytes_fundus,1) 
    col1.subheader('Uploaded Test Image')
    col1.image(test_image_decoded, channels = "BGR")

    sq_area_test, cir_area_test = get_contours(test_image_decoded)

    difference = []
    for sq, cl in zip(square_area, circle_area):
        diff = np.sqrt((sq_area_test-sq)**2+(cir_area_test-cl)**2)
        difference.append(diff)

    difference_sorted = sorted(difference)
    pred = difference.index(difference_sorted[0])
    col2.subheader('Predicted Similar Image')
    col2.image(img_list[pred], channels = "BGR")
