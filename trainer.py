from data_proc import imgs_train,imgs_test,samples_labels,val_split,W,H,C
from keras.applications import VGG19
import tensorflow as tf
from keras.models import Sequential,Model
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau
from keras.layers import Flatten,Dense,Activation,GlobalAveragePooling2D,Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
n_classes = 3 #This is the number of our classes

###We prepare our testing data
# X_train,y_train,X_val,y_val

X,y = samples_labels(imgs_train)
X_train,X_val,y_train,y_val = val_split(X,y)

#X_test,y_test

X_test,y_test = samples_labels(imgs_test)

# MODEL = VGG19
model_vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(H,W,C))

#Copying model
x = model_vgg19.output
x = GlobalAveragePooling2D()(x)

# add fully-connected layer
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)

# add output layer
outputs = Dense(n_classes, activation='softmax')(x)
model = Model(inputs=model_vgg19.input, outputs=outputs)
for layer in model_vgg19.layers: #We disable training for vgg layers
    layer.trainable = False

#Compiling model
model.compile(optimizer='Adam', metrics=['accuracy'], loss='categorical_crossentropy')

#Augmentation of the data
datagen = ImageDataGenerator(
    rescale=1./ 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
#Batch generator
train_gen = datagen.flow(x=np.array(X_train), y=y_train, batch_size=20)
valid_gen = datagen.flow(x=np.array(X_val), y=y_val, batch_size=20)
#callback
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
checkpoint = ModelCheckpoint(filepath='mymodel.h5', 
                               verbose=1, save_best_only=True)
callbacks = [checkpoint, lr_reducer]

#Training
model.fit_generator(train_gen, steps_per_epoch=120, epochs=20, verbose=1, validation_data=valid_gen, validation_steps=60,
                    callbacks=[callbacks])


#Testing
t_datagen = ImageDataGenerator(
    rescale=1./255
)
testgen = t_datagen.flow(np.array(X_test),batch_size = 32)
predictions = model.predict_generator(testgen,steps=125,verbose=1)
pred_b,pred_s,pred_f = [],[],[]
for elt in predictions:
    pred_b.append(elt[0])
    pred_s.append(elt[1])
    pred_f.append(elt[2])

results = pd.DataFrame({"id": imgs_test, "b":pred_b,"s":pred_s,"f":pred_f})
results.to_csv("results.csv", index = False)

