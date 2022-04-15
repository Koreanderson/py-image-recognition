import tensorflow
# tf.keras.preprocessing.image.img_to_array()

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import cv2


# Loading the image to predict
# img_path = 'images/car.jpg' 
# img_path = 'images/cat-1.jpg' 
img_path = 'images/cat-1.jpg' 
img = load_img(img_path)

# Resize image to 224x224 square
img = img.resize((299,299))

# Convert image to array
img_array = img_to_array(img)

# Convert image into a 4 dimensional Tensor

# Convert from (height, width, channels), ( batchsize, height, width, channels)
img_array = np.expand_dims(img_array, axis=0)

# Preprocess the input image array
img_array = preprocess_input(img_array)

# Grab our pretrained model from InceptionV3
# approx 530mb
pretrained_model = InceptionV3(weights="imagenet")

#predict using predict() method
prediction = pretrained_model.predict(img_array)

# decode  the prediction
actual_prediction = imagenet_utils.decode_predictions(prediction)

# Grab the highest percent match from list of matches
print("prediction:",actual_prediction[0][0][1])
print("with accuracy",actual_prediction[0][0][2] * 100)

predicted_text = actual_prediction[0][0][1]

disp_img = cv2.imread(img_path)
cv2.putText(disp_img, predicted_text, (20,20), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255,0,0))

cv2.imshow("Prediction",disp_img)
cv2.waitKey(5000)