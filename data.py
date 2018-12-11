import numpy as np
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator
from descriptors import PLAB, PHOG, PLBP
import random 
import glob 
from keras.utils import np_utils

def multi_feature(img, only_index = -1):
    plbp = PLBP(img)
    phog = PHOG(img)
    plab = PLAB(img)
    featuers = [plbp, phog, plab]
    
    if only_index == -1:
        desc = np.concatenate(featuers, axis = 0)
    else:
        desc = np.concatenate([featuers[only_index]], axis = 0)
    desc = desc / np.max(desc)
    return desc.T

def get_class(path):
    if 'Normal' in path:
        return 0
    if 'CIN1' in path:
        return 1
    if 'CIN2' in path:
        return 2
    if 'CIN3' in path:
        return 3
    if 'Cancer' in path:
        return 4
    
def load_multi_data(path, num_classes = 5, shuffle = True):
    
    #variables
    imgs = []
    features = []
    labels = []
    
    
    paths = glob.glob(path)
    
    for p in paths:
        #load the data and label them 
        img_path = p
        img = cv2.imread(img_path)
        img = cv2.imread(img_path)[:,:,::-1]
        feature = multi_feature(img)
        img = preprocess(img, normalize = True)  
        imgs.append(img[0])
        features.append(feature[0])
        label = get_class(p)
        labels.append(np_utils.to_categorical(label, num_classes = num_classes))
        
    if shuffle:
        imgs, features, labels = shuffle_data(imgs, features, labels)
        
    return imgs, features, labels
        
def load_data(path, split = 0.9):
    labels = {'negative':0, 'positive':1}
    dirs = os.listdir(path)
    
    #variables
    train_x = []
    valid_x = []
    train_y = []
    valid_y = []
    
    for dir in dirs:
        
        files = os.listdir(path+dir)
        split_idx = int(split * len(files))
        
        i = 0
        for file in files:
            #load the data and label them 
            img_path = path+dir+'/'+file
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            
            feature = multi_feature(img)
            
            if i < split_idx:
                train_x.append(feature[0])
                train_y.append(labels[dir])
            else:
                valid_x.append(feature[0])
                valid_y.append(labels[dir])
            
            i += 1
    
    #shuffle
    #train_x, train_y = shuffle_data(train_x, train_y)
    
    #numpy arrays
    train_x = np.array(train_x)
    valid_x = np.array(valid_x)
    train_y = np.array(train_y)
    valid_y = np.array(valid_y)
    
    print('Training ', train_x.shape)
    print('Validation', valid_x.shape)
    
    return train_x, train_y, valid_x, valid_y 

def preprocess(img, normalize = True):
    img = cv2.resize(img, (256, 256))
    if normalize:
        img = img/ 255. 
    img = np.expand_dims(img, 0) 
    return img  

def feature_generator(batches):
    while True:
        batch_x, batch_y = next(batches)
        feature_batch  = np.zeros((batch_x.shape[0], 4736))
        for i in range(batch_x.shape[0]):
            feature_batch[i] = multi_feature(batch_x[i])
        yield (feature_batch, batch_y)
        
def create_generators():
    target_size = (300, 250)
    batch_size = 8
    train_datagen = ImageDataGenerator()

    test_datagen = ImageDataGenerator()

    train_batch = train_datagen.flow_from_directory('NHS/cropped/train', target_size = target_size,
                                                        batch_size = batch_size, class_mode = 'binary')
    train_generator = feature_generator(train_batch)

    test_batch = test_datagen.flow_from_directory('NHS/cropped/valid', target_size = target_size,
                                                      batch_size = batch_size, class_mode = 'binary')

    test_generator = feature_generator(test_batch)
    
    return train_generator, test_generator

def load_directory(path, shuffle = True, normalize = True, splitted = True, only_index = -1):
    classes = {'negative':0, 'positive':1}
    
    #variables
    imgs = []
    features = []
    labels = []

    if splitted:
        for mode in ['train', 'valid']:
            for cls in ['negative', 'positive']:
                files = os.listdir(path+mode+'/'+cls)

                for file in files:
                    #load the data and label them 
                    img_path = path+mode+'/'+cls+'/'+file
                    img = cv2.imread(img_path)[:,:,::-1]
                    feature = multi_feature(img, only_index)
                    img = preprocess(img, normalize = normalize)  
                    imgs.append(img[0])
                    features.append(feature[0])
                    labels.append(classes[cls])
    else:
        for cls in ['negative', 'positive']:
                files = os.listdir(path+cls)
                for file in files:
                    #load the data and label them 
                    img_path = path+cls+'/'+file
                    img = cv2.imread(img_path)[:,:,::-1]
                    feature = multi_feature(img, only_index)
                    img = preprocess(img, normalize = normalize)  
                    imgs.append(img[0])
                    features.append(feature[0])
                    labels.append(classes[cls])
        
    
    if shuffle:
        imgs, features, labels = shuffle_data(imgs, features, labels)
    
    return imgs, features, labels

def shuffle_data(imgs, features, labels):
    perm = np.random.permutation(len(imgs))
    x1 = []
    x2 = []
    y = []
    for r in perm:
        x1.append(imgs[r])
        x2.append(features[r])
        y.append(labels[r])
    return x1, x2, y

def validation_split(data, labels, vfold= 0, split = 10, stratified = False):
    
    if stratified:
        return str_validation_split(data, labels, vfold= vfold, split = split)
    
    
    #calculate the split indices 
    split_length = len(data) // split 
    split_idx = vfold*split_length
    
    #extract the validation fold 
    valid_x = data[split_idx: split_idx + split_length]
    valid_y = labels[split_idx: split_idx + split_length]
    
    #remove the validation set 
    train_x = data[0:split_idx] + data[split_idx + split_length:]
    train_y = labels[0:split_idx] + labels[split_idx + split_length:]
    
    #numpy arrays
    train_x = np.array(train_x)
    valid_x = np.array(valid_x)
    train_y = np.array(train_y)
    valid_y = np.array(valid_y)
    
    return train_x, train_y, valid_x, valid_y


def str_validation_split(data, labels, vfold= 0, split = 10):
    
    
    #calculate the split indices 
    split_length = len(data) // (2*split)
    split_idx = vfold*split_length
    
    train_x = []
    train_y = []
    valid_x = []
    valid_y = []
    
    pos_cnt = neg_cnt = 0 
    i = 0 
    
    for label in labels:
        
        if label == 0: 
            if neg_cnt >= split_idx and neg_cnt < split_idx + split_length:
                valid_x.append(data[i])
                valid_y.append(labels[i])
            else:
                train_x.append(data[i])
                train_y.append(labels[i])
            neg_cnt += 1    
        if label == 1:
            if pos_cnt >= split_idx and pos_cnt < split_idx + split_length:
                valid_x.append(data[i])
                valid_y.append(labels[i])
            else:
                train_x.append(data[i])
                train_y.append(labels[i])   
            pos_cnt += 1
        i += 1
        
    #numpy arrays
    train_x = np.array(train_x)
    valid_x = np.array(valid_x)
    train_y = np.array(train_y)
    valid_y = np.array(valid_y)
    
    return train_x, train_y, valid_x, valid_y
    