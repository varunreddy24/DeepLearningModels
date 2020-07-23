import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Activation, Dense, Input, MaxPooling2D
from tensorflow.keras.layers import MaxPooling2D, Add, ZeroPadding2D, BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D,GlobalAveragePooling2D
from tensorflow.keras.models import Model


def residual_block2v1(x,filters,stride=1,conv_shortcut=True):
    if conv_shortcut:
        shortcut = Conv2D(filters,1,strides=stride)(x)
    else:
        shortcut = x

    x = ZeroPadding2D(padding=((1,1),(1,1)))(x)
    x = Conv2D(filters,3,strides=stride)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D(padding=((1,1),(1,1)))(x)
    x = Conv2D(filters,3)(x)
    x = BatchNormalization()(x)

    x = Add()([x,shortcut])
    x = Activation('relu')(x)
    return x

def stack21(x,filters,blocks,stride=2):
    x = residual_block2v1(x,filters,stride)
    for i in range(blocks-1):
        x = residual_block2v1(x,filters,conv_shortcut=False)
    return x

def residual_block2v2(x,filters,kernel_size=3,stride=1,conv_shortcut=True):
    preact = BatchNormalization()(x)
    preact = Activation('relu')(x)

    if conv_shortcut:
        shortcut = Conv2D(filters,1,strides=stride)(preact)
    else:
        print("this")
        shortcut = MaxPooling2D(3,strides=stride)(x) if stride>1 else x

    x = ZeroPadding2D(padding=((1,1),(1,1)))(preact)
    x = Conv2D(filters,3,strides=stride)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=((1,1),(1,1)))(x)
    x = Conv2D(filters,3)(x)

    print(x.shape)
    print(shortcut.shape)
    x = Add()([x,shortcut])
    return x

def stack22(x,filters,blocks,stride=2):
    x = residual_block2v2(x,filters,conv_shortcut=True)
    for i in range(blocks-2):
        x = residual_block2v2(x,filters)
    x = residual_block2v2(x,filters,stride=stride)
    return x

def residual_block3v1(x,filters,kernel_size=3,stride=1,conv_shortcut = True):
    if conv_shortcut:
        shortcut = Conv2D(4*filters,1,strides=stride)(x)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = x

    x = Conv2D(filters,1,strides=stride)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters,kernel_size,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(4*filters,(1,1))(x)
    x = BatchNormalization()(x)
    
    x = Add()([x,shortcut])
    x = Activation('relu')(x) 
    return x

def stack31(x,filters,blocks,stride=2):
    x = residual_block3v1(x,filters,3,stride)
    for i in range(blocks-1):
        x = residual_block3v1(x,filters,conv_shortcut=False)
    return x

def residual_block3v2(x,filters,kernel_size=3,stride=1,conv_shortcut=False):
    preact = BatchNormalization()(x)
    preact = Activation('relu')(preact)

    if conv_shortcut:
        shortcut = Conv2D(4*filters,1,strides=stride)(preact)
    else:
        shortcut = MaxPooling2D(1,strides=stride)(x) if stride>1 else x
    
    x = Conv2D(filters,1,strides=1)(preact)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=((1,1),(1,1)))(x)
    x = Conv2D(filters,kernel_size,stride)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(4*filters,1)(x)
    
    x = Add()([x,shortcut])
    return x

def stack32(x,filters,blocks,stride=2):
    x = residual_block3v2(x,filters,conv_shortcut=True)
    for i in range(blocks-2):
        x = residual_block3v2(x,filters)
    x = residual_block3v2(x,filters,stride=stride)
    return x

def resnet(stack_fn,input_shape,model_name,classes=10,preact=False,include_top=True):
    img_input = Input(shape=(input_shape))
    x = ZeroPadding2D(((3,3), (3,3)))(img_input)
    x = Conv2D(64,7,strides=2)(x)

    if not preact:
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3,strides=2)(x)

    x = stack_fn(x)

    if preact:
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes,activation='sigmoid')(x)

    model = Model(img_input,x,name=model_name)
    return model

def ResNet18v1(input_shape,include_top=True,classes=10):
    def stack_fn(x):
        x = stack21(x,64,2,stride=1)
        x = stack21(x,128,2,stride=2)
        x = stack21(x,256,2,stride=2)
        return stack21(x,512,2,stride=2)
    
    return resnet(stack_fn,input_shape,'ResNet18',classes,include_top=include_top)

def ResNet18v2(input_shape,include_top=True,classes=10):
    def stack_fn(x):
        x = stack22(x,64,2,stride=2)
        x = stack22(x,128,2,stride=2)
        x = stack22(x,256,2,stride=2)
        return stack22(x,512,2,stride=1)
    
    return resnet(stack_fn,input_shape,'ResNet18v2',classes,True,include_top)

def ResNet34v1(input_shape,include_top=True,classes=10):
    def stack_fn(x):
        x = stack21(x,64,3,stride=1)
        x = stack21(x,128,4,stride=2)
        x = stack21(x,256,6,stride=2)
        return stack21(x,512,3,stride=2)
    
    return resnet(stack_fn,input_shape,'ResNet34',classes,include_top=include_top)

def ResNet34v2(input_shape,include_top=True,classes=10):
    def stack_fn(x):
        x = stack22(x,64,3,stride=2)
        x = stack22(x,128,4,stride=2)
        x = stack22(x,256,6,stride=2)
        return stack22(x,512,3,stride=1)
    
    return resnet(stack_fn,input_shape,'ResNet34v2',classes,True,include_top)

def ResNet50v1(input_shape,include_top = True,classes=10):
    def stack_fn(x):
        x = stack31(x,64,3,stride=1)
        x = stack31(x,128,4,stride=2)
        x = stack31(x,256,6,stride=2)
        return stack31(x,512,3,stride=2)

    return resnet(stack_fn,input_shape,'ResNet50',classes,include_top=include_top)

def ResNet50v2(input_shape,include_top = True,classes=10):
    def stack_fn(x):
        x = stack32(x,64,3,stride=2)
        x = stack32(x,128,4,stride=2)
        x = stack32(x,256,6,stride=2)
        return stack32(x,512,3,stride=1)

    return resnet(stack_fn,input_shape,'ResNet50v2',classes,include_top=include_top,preact=True)

def ResNet101v1(input_shape,include_top=True,classes=10):
    def stack_fn(x):
        x = stack31(x,64,3,stride=1)
        x = stack31(x,128,4,stride=2)
        x = stack31(x,256,23,stride=2)
        return stack31(x,512,3,stride=2)
    
    return resnet(stack_fn,input_shape,'ResNet101',classes,include_top=include_top)

def ResNet101V2(input_shape,include_top=True,classes=10):
    def stack_fn(x):
        x = stack32(x,64,3,stride=2)
        x = stack32(x,128,3,stride=2)
        x = stack32(x,256,23,stride=2)
        return stack31(x,512,3,stride=1)
    
    return resnet(stack_fn,input_shape,'ResNet101v2',classes,include_top,preact=True)

def ResNet152v1(input_shape,include_top=True,classes=10):
    def stack_fn(x):
        x = stack31(x,64,3,stride=1)
        x = stack31(x,128,8,stride=2)
        x = stack31(x,256,36,stride=2)
        return stack31(x,512,3,stride=2)
    
    return resnet(stack_fn,input_shape,'ResNet152',classes,include_top=include_top)

def ResNet152V2(input_shape,include_top=True,classes=10):
    def stack_fn(x):
        x = stack32(x,64,3,stride=2)
        x = stack32(x,128,8,stride=2)
        x = stack32(x,256,36,stride=2)
        return stack31(x,512,3,stride=1)
    
    return resnet(stack_fn,input_shape,'ResNet152v2',classes,include_top,preact=True)