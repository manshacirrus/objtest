import tensorflow as tf
AUTO = tf.data.experimental.AUTOTUNE

# Detect TPU, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() 

print("REPLICAS: ", strategy.num_replicas_in_sync)
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
import tensorflow.keras.layers as tfl
from keras import backend as K

from tensorflow.keras.datasets import cifar100

(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='coarse')
print(X_train.shape,y_train.shape)

import cv2
X_train = np.array([cv2.resize(img, (140, 140)) for img in X_train])

X_test = np.array([cv2.resize(img, (140, 140)) for img in X_test])

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
y_train=enc.fit_transform(y_train).toarray().astype(int)
y_test=enc.transform(y_test).toarray().astype(int)


print(y_train.shape)
print(y_train[0])

def identity_block(X, f, filters):
    X_shortcut=X

    X=tfl.Conv2D(filters=filters[0],kernel_size=1,strides=(1,1), padding='valid')(X)
    X=tfl.BatchNormalization(axis=3)(X, training=True)  
    X=tfl.Activation('relu')(X)

    X=tfl.Conv2D(filters=filters[1],kernel_size=f,strides=(1,1), padding='same')(X)
    X=tfl.BatchNormalization(axis=3)(X, training=True)
    X=tfl.Activation('relu')(X)

    X=tfl.Conv2D(filters=filters[2],kernel_size=1,strides=(1,1), padding='valid')(X)
    X=tfl.BatchNormalization(axis=3)(X, training=True)

    X=tfl.Add()([X_shortcut,X])
    X=tfl.Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, s=2):
    X_shortcut=X

    X=tfl.Conv2D(filters=filters[0],kernel_size=1,strides=(s,s), padding='valid')(X)
    X=tfl.BatchNormalization(axis=3)(X, training=True)
    X=tfl.Activation('relu')(X)

    X=tfl.Conv2D(filters=filters[1],kernel_size=f,strides=(1,1), padding='same')(X)
    X=tfl.BatchNormalization(axis=3)(X, training=True)
    X=tfl.Activation('relu')(X)

    X=tfl.Conv2D(filters=filters[2],kernel_size=1,strides=(1,1), padding='valid')(X)
    X=tfl.BatchNormalization(axis=3)(X, training=True)

    X_shortcut=tfl.Conv2D(filters=filters[2],kernel_size=1,strides=(s,s), padding='valid')(X_shortcut)
    X_shortcut=tfl.BatchNormalization(axis=3)(X_shortcut, training=True)
    X=tfl.Add()([X_shortcut,X])
    X=tfl.Activation('relu')(X)

    return X

def arch(input_shape):

    input_img = tf.keras.Input(shape=input_shape)

    #layer = data_augmenter()(input_img)

    layer =tfl.ZeroPadding2D((3, 3))(input_img)

    layer=tfl.Conv2D(filters=64,kernel_size=7,strides=(2,2))(layer)
    layer=tfl.BatchNormalization(axis=3)(layer, training=True)
    layer=tfl.Activation('relu')(layer)
    layer=tfl.MaxPooling2D((3, 3), strides=(2, 2))(layer)

    layer=convolutional_block(layer,3,[64,64,256],1)
    layer=identity_block(layer,3,[64,64,256])
    layer=identity_block(layer,3,[64,64,256])

    layer=convolutional_block(layer,3,[128,128,512],2)
    layer=identity_block(layer,3,[128,128,512])
    layer=identity_block(layer,3,[128,128,512])
    layer=identity_block(layer,3,[128,128,512])  
    layer=convolutional_block(layer,3, [256, 256, 1024],2)
    layer=identity_block(layer,3, [256, 256, 1024])
    layer=identity_block(layer,3, [256, 256, 1024])
    layer=identity_block(layer,3, [256, 256, 1024])
    layer=identity_block(layer,3, [256, 256, 1024])
    layer=identity_block(layer,3, [256, 256, 1024])

    layer=convolutional_block(layer,3, [512, 512, 2048],2)
    layer=identity_block(layer,3, [512, 512, 2048])
    layer=identity_block(layer,3, [512, 512, 2048])

    layer=tfl.AveragePooling2D(pool_size=(2, 2),padding='same')(layer)
    layer=tfl.Flatten()(layer)

    outputs=tfl.Dense(units= 20 , activation='softmax')(layer)
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model
# instantiating the model in the strategy scope creates the model on the TPU
with strategy.scope():
    conv_model = arch((140, 140, 3)) # define your model normally
    conv_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
conv_model.summary()
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(16 * strategy.num_replicas_in_sync)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(16 * strategy.num_replicas_in_sync)
history = conv_model.fit(train_dataset,epochs=4,validation_data=test_dataset,batch_size=16 * strategy.num_replicas_in_sync,shuffle=True)