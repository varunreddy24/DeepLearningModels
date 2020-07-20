import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Activation, Dropout, Dense, Input, MaxPooling2D
from tensorflow.keras.layers import MaxPooling2D, Add, ZeroPadding2D, BatchNormalization
from tensorflow.keras.layers import AveragePooling2D, ZeroPadding2D, Flatten
from tensorflow.keras.models import Model

def identity_block2v1(X,f,filters):
    f1,f2 = filters
    X_init = X

    X = Conv2D(f1,(3,3),strides=(1,1),padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(f1,(3,3),strides=(1,1),padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Add()([X,X_init])
    X = Activation('relu')(X)
    return X

def identity_block2v2(X,f,filters):
    f1,f2 = filters
    X_init = X

    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(f1,(3,3),strides=(1,1),padding='same')(X)
    
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(f1,(3,3),strides=(1,1),padding='same')(X)

    X = Add()([X,X_init])
    return X

def conv_block2v1(X,f,filters,s=2):
    f1,f2 = filters
    X_init = X

    X = Conv2D(f1,(3,3),strides=(s,s),padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(f2,(3,3),strides=(1,1),padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X_init = Conv2D(f1,(3,3),strides=(s,s),padding='same')(X_init)
    X_init = BatchNormalization()(X_init)

    X = Add()([X,X_init])
    X = Activation('relu')(X)
    return X

def conv_block2v2(X,f,filters,s=2):
    f1,f2 = filters
    X_init = X

    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(f1,(3,3),strides=(s,s),padding='same')(X)

    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(f2,(3,3),strides=(1,1),padding='same')(X)

    X_init = BatchNormalization()(X_init)
    X_init = Conv2D(f1,(3,3),strides=(s,s),padding='same')(X_init)
    

    X = Add()([X,X_init])
    return X

def identity_block3v1(X, f, filters):
    f1,f2,f3 = filters
    X_init = X

    X = Conv2D(f1,(1,1),strides=(1,1),padding='valid')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(f2,(f,f),strides=(1,1),padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(f3,(1,1),strides=(1,1),padding='valid')(X)
    X = BatchNormalization()(X)

    X = Add()[X,X_init]
    X = Activation('relu')
    return X

def conv_block3v1(X,f,filters,s=2):
    f1,f2,f3 = filters
    X_init = X

    X = Conv2D(f1,(1,1),strides=(s,s),padding='valid')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(f2,(f,f),strides=(1,1),padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation ('relu')(X)

    X = Conv2D(f3,(1,1),strides=(1,1),padding='valid')(X)
    X = BatchNormalization()(X)

    X_init = Conv2D(f3,(1,1),strides=(s,s),padding='valid')(X_init)
    X_init = BatchNormalization()(X_init)

    X = Add()([X,X_init])
    X = Activation('relu')(X)
    return X

def identity_block3v2(X, f, filters):
    f1,f2,f3 = filters
    X_init = X

    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(f1,(1,1),strides=(1,1),padding='valid')(X)

    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(f2,(f,f),strides=(1,1),padding='same')(X)

    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(f3,(1,1),strides=(1,1),padding='valid')(X)

    X = Add()[X,X_init]
    return X

def conv_block3v2(X,f,filters,s=2):
    f1,f2,f3 = filters
    X_init = X

    X = BatchNormlaization()(X)
    X = Activation('relu')(X)
    X = Conv2D(f1,(1,1),strides=(s,s),padding='valid')(X_init)

    X = BatchNormlaization()(X)
    X = Activation('relu')(X)
    X = Conv2D(f2,(f,f),strides=(1,1),padding='same')(X)

    X = BatchNormlaization()(X)
    X = Activation('relu')(X)
    X = Conv2D(f3,(1,1),strides=(1,1),padding='valid')(X)

    X_init = BatchNormlaization()(X_init)
    X_init = Conv2D(f3,(1,1),strides=(s,s),padding='valid')(X_init)

    X = Add()([X,X_init])
    return X

def resnet18v1(input_shape,classes):
    X_input = Input(shape=input_shape)

    X = ZeroPadding2D((3,3))(X_input)
    X = Conv2D(64,(7,7),strides=(2,2))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3),strides=(2,2))(X)

    X = identity_block2v1(X,3,[64,64])
    X = identity_block2v1(X,3,[64,64])
    X = Dropout(0.2)(X)

    X = conv_block2v1(X,3,[128,128],s=2)
    X = identity_block2v1(X,3,[128,128])
    X = Dropout(0.2)(X)

    X = conv_block2v1(X,3,[256,256],s=2)
    X = identity_block2v1(X,3,[256,256])
    X = Dropout(0.2)(X)

    X = conv_block2v1(X,3,[512,512],s=2)
    X = identity_block2v1(X,3,[512,512])
    X = Dropout(0.2)(X)
    
    X = AveragePooling2D((2,2),strides=(1,1))(X)
    X = Flatten()(X)
    X = Dense(classes,activation='softmax')(X)

    model = Model(inputs=X_input,outputs=X,name='ResNet18v1')

    return model

def resnet18v2(input_shape,classes):
    X_input = Input(shape=input_shape)

    X = ZeroPadding2D((3,3))(X_input)
    X = Conv2D(64,(7,7),strides=(2,2))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3),strides=(2,2))(X)

    X = identity_block2v2(X,3,[64,64],s=1)
    X = identity_block2v2(X,3,[64,64])
    X = Dropout(0.4)(X)

    X = conv_block2v2(X,3,[128,128],s=2)
    X = identity_block2v2(X,3,[128,128])
    X = Dropout(0.4)(X)

    X = conv_block2v2(X,3,[256,256],s=2)
    X = identity_block2v2(X,3,[256,256])
    X = Dropout(0.4)(X)

    X = conv_block2v2(X,3,[512,512],s=2)
    X = identity_block2v2(X,3,[512,512])
    X = Dropout(0.4)(X)
    
    X = AveragePooling2D((2,2),strides=(1,1))(X)
    X = Flatten()(X)
    X = Dense(classes,activation='softmax')(X)

    model = Model(inputs=X_input,outputs=X,name='ResNet18v2')

    return model

def resnet34v1(input_shape,classes):
    X_input = Input(shape=input_shape)

    X = ZeroPadding2D((3,3))(X_input)
    X = Conv2D(64,(7,7),strides=(2,2))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3),strides=(2,2))(X)

    for i in range(3):
        X = identity_block2v1(X,3,[64,64])
    
    X = conv_block2v1(X,3,[128,128],s=2)
    for i in range(3):
        X = identity_block2v1(X,3,[128,128])

    X = conv_block2v1(X,3,[256,256],s=2)
    for i in range(5):
        X = identity_block2v1(X,3,[256,256])

    X = conv_block2v1(X,3,[512,512],s=2)
    for i in range(2):
        X = identity_block2v1(X,3,[512,512])
    
    X = AveragePooling2D((2,2),strides=(1,1))(X)
    X = Flatten()(X)
    X = Dense(classes,activation='softmax')(X)

    model = Model(inputs=X_input,outputs=X,name='ResNet34v1')

    return model

def resnet34v2(input_shape,classes):
    X_input = Input(shape=input_shape)

    X = ZeroPadding2D((3,3))(X_input)
    X = Conv2D(64,(7,7),strides=(2,2))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3),strides=(2,2))(X)

    for i in range(3):
        X = identity_block2v2(X,3,[64,64])
    
    X = conv_block2v2(X,3,[128,128],s=2)
    for i in range(3):
        X = identity_block2v2(X,3,[128,128])

    X = conv_block2v2(X,3,[256,256],s=2)
    for i in range(5):
        X = identity_block2v2(X,3,[256,256])

    X = conv_block2v2(X,3,[512,512],s=2)
    for i in range(2):
        X = identity_block2v2(X,3,[512,512])
    
    X = AveragePooling2D((2,2),strides=(1,1))(X)
    X = Flatten()(X)
    X = Dense(classes,activation='softmax')(X)

    model = Model(inputs=X_input,outputs=X,name='ResNet34v2')

    return model

def resnet50v1(input_shape,classes):
    X_input = Input(shape=input_shape)

    X = ZeroPadding2D((3,3))(X_input)
    X = Conv2D(64,(7,7),strides=(2,2))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3),strides=(2,2))(X)

    X = conv_block3v1(X,3,[64,64,256],s=1)
    for i in range(2):
    X = identity_block3v1(X,3,[64,64,256])

    X = conv_block3v1(X,3,[128,128,512],s=2)
    for i in range(3):
        X = identity_block3v1(X,3,[128,128,512])
    
    X = conv_block3v1(X,3,[256,256,1024],s=2)
    for i in range(5):
        X = identity_block3v1(X,3,[256,256,1024])
    
    X = conv_block3v1(X,3,[512,512,2048],s=2)
    for i in range(2):
        X = identity_block3v1(X,3,[512,512,2048])
    
    X = AveragePooling2D((2,2),strides=(1,1))(X)
    X = Flatten()(X)
    X = Dense(classes,activation='softmax')(X)

    model = Model(inputs=X_input,outputs=X,name='ResNet50v1')

    return model

def resnet50v2(input_shape,classes):
    X_input = Input(shape=input_shape)

    X = ZeroPadding2D((3,3))(X_input)
    X = Conv2D(64,(7,7),strides=(2,2))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3),strides=(2,2))(X)

    X = conv_block3v2(X,3,[64,64,256],s=1)
    for i in range(2):
    X = identity_block3v2(X,3,[64,64,256])

    X = conv_block3v2(X,3,[128,128,512],s=2)
    for i in range(3):
        X = identity_block3v2(X,3,[128,128,512])
    
    X = conv_block3v2(X,3,[256,256,1024],s=2)
    for i in range(5):
        X = identity_block3v2(X,3,[256,256,1024])
    
    X = conv_block3v2(X,3,[512,512,2048],s=2)
    for i in range(2):
        X = identity_block3v2(X,3,[512,512,2048])
    
    X = AveragePooling2D((2,2),strides=(1,1))(X)
    X = Flatten()(X)
    X = Dense(classes,activation='softmax')(X)

    model = Model(inputs=X_input,outputs=X,name='ResNet50v2')

    return model

def resnet101v1(input_shape,classes):
    X_input = Input(shape=input_shape)

    X = ZeroPadding2D((3,3))(X_input)
    X = Conv2D(64,(7,7),strides=(2,2))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3),strides=(2,2))(X)

    X = conv_block3v1(X,3,[64,64,256],s=1)
    for i in range(2):
    X = identity_block3v1(X,3,[64,64,256])

    X = conv_block3v1(X,3,[128,128,512],s=2)
    for i in range(3):
        X = identity_block3v1(X,3,[128,128,512])
    
    X = conv_block3v1(X,3,[256,256,1024],s=2)
    for i in range(22):
        X = identity_block3v1(X,3,[256,256,1024])
    
    X = conv_block3v1(X,3,[512,512,2048],s=2)
    for i in range(2):
        X = identity_block3v1(X,3,[512,512,2048])
    
    X = AveragePooling2D((2,2),strides=(1,1))(X)
    X = Flatten()(X)
    X = Dense(classes,activation='softmax')(X)

    model = Model(inputs=X_input,outputs=X,name='ResNet101v1')

    return model

def resnet101v2(input_shape,classes):
    X_input = Input(shape=input_shape)

    X = ZeroPadding2D((3,3))(X_input)
    X = Conv2D(64,(7,7),strides=(2,2))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3),strides=(2,2))(X)

    X = conv_block3v2(X,3,[64,64,256],s=1)
    for i in range(2):
    X = identity_block3v2(X,3,[64,64,256])

    X = conv_block3v2(X,3,[128,128,512],s=2)
    for i in range(3):
        X = identity_block3v2(X,3,[128,128,512])
    
    X = conv_block3v2(X,3,[256,256,1024],s=2)
    for i in range(5):
        X = identity_block3v2(X,3,[256,256,1024])
    
    X = conv_block3v2(X,3,[512,512,2048],s=2)
    for i in range(2):
        X = identity_block3v2(X,3,[512,512,2048])
    
    X = AveragePooling2D((2,2),strides=(1,1))(X)
    X = Flatten()(X)
    X = Dense(classes,activation='softmax')(X)

    model = Model(inputs=X_input,outputs=X,name='ResNet101v2')

    return model

def resnet152v1(input_shape,classes):
    X_input = Input(shape=input_shape)

    X = ZeroPadding2D((3,3))(X_input)
    X = Conv2D(64,(7,7),strides=(2,2))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3),strides=(2,2))(X)

    X = conv_block3v1(X,3,[64,64,256],s=1)
    for i in range(2):
    X = identity_block3v1(X,3,[64,64,256])

    X = conv_block3v1(X,3,[128,128,512],s=2)
    for i in range(7):
        X = identity_block3v1(X,3,[128,128,512])
    
    X = conv_block3v1(X,3,[256,256,1024],s=2)
    for i in range(35):
        X = identity_block3v1(X,3,[256,256,1024])
    
    X = conv_block3v1(X,3,[512,512,2048],s=2)
    for i in range(2):
        X = identity_block3v1(X,3,[512,512,2048])
    
    X = AveragePooling2D((2,2),strides=(1,1))(X)
    X = Flatten()(X)
    X = Dense(classes,activation='softmax')(X)

    model = Model(inputs=X_input,outputs=X,name='ResNet152v1')

    return model

def resnet152v2(input_shape,classes):
    X_input = Input(shape=input_shape)

    X = ZeroPadding2D((3,3))(X_input)
    X = Conv2D(64,(7,7),strides=(2,2))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3),strides=(2,2))(X)

    X = conv_block3v2(X,3,[64,64,256],s=1)
    for i in range(2):
    X = identity_block3v2(X,3,[64,64,256])

    X = conv_block3v2(X,3,[128,128,512],s=2)
    for i in range(7):
        X = identity_block3v2(X,3,[128,128,512])
    
    X = conv_block3v2(X,3,[256,256,1024],s=2)
    for i in range(35):
        X = identity_block3v2(X,3,[256,256,1024])
    
    X = conv_block3v2(X,3,[512,512,2048],s=2)
    for i in range(2):
        X = identity_block3v2(X,3,[512,512,2048])
    
    X = AveragePooling2D((2,2),strides=(1,1))(X)
    X = Flatten()(X)
    X = Dense(classes,activation='softmax')(X)

    model = Model(inputs=X_input,outputs=X,name='ResNet152v2')

    return model