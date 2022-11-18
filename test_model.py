import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
model = keras.models.load_model('mymodel.h5')

img = cv2.imread("data/seg_pred/seg_pred/foresto.jpg")
img = cv2.resize(img,(150,150),interpolation=cv2.INTER_CUBIC)
test_image = np.expand_dims(img, axis = 0)
plt.imshow(img)
pred = model.predict(test_image)
print(pred)
plt.show()