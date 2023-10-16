import cv2
import tensorflow as tf
import numpy as np
import scipy.ndimage as sci
import time
import os
def resizeIt(img, size=100, median=2):
    img = np.float32(img)
    r, c = img.shape
    resized_img = cv2.resize(img, (size, size))
    filtered_img = sci.median_filter(resized_img, median)
    return np.uint8(filtered_img)
def preprocessing(img0, IMG_SIZE=100):
    img_resized = resizeIt(img0, IMG_SIZE, 1)
    img_blur = cv2.GaussianBlur(img_resized, (5, 5), 0)
    imgTh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 3)
    ret, img_th = cv2.threshold(imgTh, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
    return img_th
ALPHABET =  ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 
prev = ""
word_buffer = []
word = ""
model = tf.keras.models.load_model("model_name.model")
prev_time = time.time()
cap = cv2.VideoCapture(0)
while True:
    ret, src = cap.read()
    if not ret:
        continue
    img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    IMG_SIZE = 200
    img_test = preprocessing(img_gray, IMG_SIZE)
    prediction = model.predict([img_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)])
    detected_letter = ALPHABET[int(np.argmax(prediction[0]))]
    # add text overlay on input frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(src, 'Alphabet: ' + detected_letter, (50, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    # add system title overlay on input frame
    cv2.putText(src, 'SIGN LANGUAGE RECOGNITION ', (int(src.shape[1]/2) - 200, src.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # add predicted letters in the top right corner
    if len(word_buffer) == 0:
        word = ""
    else:
        word = ''.join(word_buffer)
    cv2.putText(src, 'Word: ' + word, (src.shape[1] - 300, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    # check if the 'a' or 'b' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        # add detected_letter to the word buffer
        word_buffer.append(detected_letter)
    elif key == ord('b'):
        # reset word buffer
        word_buffer = []
    cv2.imshow("Sign Language Recognition", src)
    cv2.imshow('Processed Image', img_test)
    # check if q key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()