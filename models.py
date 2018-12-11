import keras
from keras.layers import Dense, Dropout, Input, Flatten, Conv2D, MaxPooling2D, Concatenate
from keras.models import Sequential, Model 
from keras.optimizers import Adam 
from keras.utils import np_utils

def combined_model():
    def conv_model(x):
        x = Conv2D(2, (3,3), strides= (2,2), padding = 'same', input_shape = (256, 256, 3), activation = 'relu')(x)
        x = MaxPooling2D(pool_size =(2,2))(x)
        x = Conv2D(4, (3,3), strides= (2,2), padding = 'same', activation = 'relu')(x)
        x = MaxPooling2D(pool_size =(2,2))(x)
        x = Conv2D(8, (3,3), strides= (2,2), padding = 'same', activation = 'relu')(x)
        x = MaxPooling2D(pool_size =(2,2))(x)
        x = Flatten()(x)
        return x 

    def create_model():
        input1 = Input(shape = train_x[0][0].shape)
        input2 = Input(shape = train_x[1][0].shape)

        out1 = conv_model(input1)

        out2 = Concatenate()([out1, input2])

        x = Dense(units = 256, activation = 'relu', input_shape = out2.shape)(out2)
        x = Dropout(0.5)(x)
        x = Dense(units = 256, activation = 'relu')(x)
        x = Dense(units = 1   , activation = 'sigmoid')(x)
        return Model(inputs = [input1, input2] , outputs = x)

    model = create_model()
    adam = Adam(lr = 1e-4)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def simple_model1(input_shape, lr = 1e-4, num_classes = 2):
    if num_classes == 2:
        loss = 'binary_crossentropy'
        out_units = 1
        activation = 'sigmoid'
    else:
        loss = 'categorical_crossentropy'
        out_units = num_classes 
        activation = 'softmax'
        
    input = Input(shape = (input_shape,))
    x = Dense(units = 128, activation = 'relu')(input)
    x = Dropout(0.5)(x)
    x = Dense(units = out_units, activation = activation)(x)
    
    model = Model(inputs = input , outputs = x)
    adam = Adam(lr = lr)
    
    model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
    return model

def simple_model2(input_shape, lr = 1e-4, num_classes = 2):
    if num_classes == 2:
        loss = 'binary_crossentropy'
        out_units = 1
        activation = 'sigmoid'
    else:
        loss = 'categorical_crossentropy'
        out_units = num_classes 
        activation = 'softmax'
        
    input = Input(shape = (input_shape,))
    x = Dense(units = 128, activation = 'relu')(input)
    x = Dropout(0.5)(x)
    x = Dense(units = 256, activation = 'relu')(x)
    x = Dense(units = out_units   , activation = activation)(x)
    model = Model(inputs = input , outputs = x)
    adam = Adam(lr = lr)
    model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
    return model

def alex_model1(lr = 1e-5, num_classes = 2):
    if num_classes == 2:
        loss = 'binary_crossentropy'
        out_units = 1
        activation = 'sigmoid'
    else:
        loss = 'categorical_crossentropy'
        out_units = num_classes 
        activation = 'softmax'
        
    model = Sequential()
    model.add(Conv2D(16, (3,3), strides= (2,2), padding = 'same', input_shape = (256, 256, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size =(2,2)))
    model.add(Conv2D(32, (3,3), strides= (2,2), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size =(2,2)))
    model.add(Flatten())
    model.add(Dense(units = 128, activation = 'tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(units = out_units, activation = activation))

    adam = Adam(lr = lr)
    model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
    return model

def alex_model2(lr = 1e-5, num_classes = 2):
    if num_classes == 2:
        loss = 'binary_crossentropy'
        out_units = 1
        activation = 'sigmoid'
    else:
        loss = 'categorical_crossentropy'
        out_units = num_classes 
        activation = 'softmax'
    
    model = Sequential()
    model.add(Conv2D(16, (3,3), strides= (2,2), padding = 'same', input_shape = (256, 256, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size =(2,2)))
    model.add(Conv2D(32, (3,3), strides= (2,2), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size =(2,2)))
    model.add(Conv2D(64, (3,3), strides= (2,2), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size =(2,2)))

    model.add(Flatten())
    model.add(Dense(units = 128, activation = 'tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(units = out_units   , activation = activation))

    adam = Adam(lr = lr)
    model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
    return model

def alex_model3(lr = 1e-5):
    model = Sequential()
    model.add(Conv2D(16, (3,3), strides= (2,2), padding = 'same', input_shape = (256, 256, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size =(2,2)))
    model.add(Conv2D(32, (3,3), strides= (2,2), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size =(2,2)))
    model.add(Conv2D(64, (3,3), strides= (2,2), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size =(2,2)))

    model.add(Flatten())
    model.add(Dense(units = 128, activation = 'tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 256, activation = 'tanh'))
    model.add(Dense(units = 1   , activation = 'sigmoid'))

    adam = Adam(lr = lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model