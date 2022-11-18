import csv
import numpy as np
import keras
from data_proc import samples_labels,imgs_test
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd


X_test,y_test = samples_labels(imgs_test)

model= keras.models.load_model("mymodel.h5")
#Testing
t_datagen = ImageDataGenerator(
    rescale=1./255,
)
testgen = t_datagen.flow(np.array(X_test),batch_size = 32,shuffle=False)
predictions = model.predict_generator(testgen,steps=125,verbose=1)
pred_b,pred_s,pred_f = [],[],[]
for elt in predictions:
    pred_b.append(elt[0])
    pred_s.append(elt[1])
    pred_f.append(elt[2])

results = pd.DataFrame({"id": imgs_test, "b":pred_b,"s":pred_s,"f":pred_f})
results.to_csv("results.csv", index = False)

f = open("results.csv",encoding="utf8",newline="")
rd = csv.reader(f,delimiter=",")
next(rd)
acc = 0
i = 0
for line in rd:
    i += 1
    label = line[0]
    true_label = 0
    if "forest" in label:
        true_label = 1
    elif "sea" in label:
        true_label = 2
    v1,v2,v3=float(line[1]),float(line[2]),float(line[3])
    pred_label = np.argmax([v1,v2,v3])
    if pred_label == true_label:
        acc += 1
print(acc/i)
f.close()
