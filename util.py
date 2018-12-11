import itertools
import numpy as np
import matplotlib.pyplot as plt
import random 
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(tru_lbl, prd_lbl, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    cm = confusion_matrix(tru_lbl, prd_lbl)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def get_tru_prd(model, x, y, thresh = 0.5):
    tru_lbl = []
    prd_lbl = []
    
    for i in range(0, x.shape[0]):
        
        prob = model.predict(np.expand_dims(x[i], 0))
        
        tru_lbl.append(y[i])
        if prob > thresh:
            prd_lbl.append(1)
        else:
            prd_lbl.append(0)
            
    return tru_lbl, prd_lbl 

def mult_tru_prd(model, x, y):
    tru_lbl = []
    prd_lbl = []
    
    for i in range(0, x.shape[0]):
        
        prob = model.predict(np.expand_dims(x[i], 0))
        
        tru_lbl.append(np.argmax(y[i]))
        prd_lbl.append(np.argmax(prob))
            
    return tru_lbl, prd_lbl 


def report_metrics(tru_lbl, prd_lbl):
    spec = 0 
    sens = 0
    accu = 0
    
    len_pos = 0 
    len_neg = 0
     
    
    for i in range(0, len(tru_lbl)):
        if tru_lbl[i] == prd_lbl[i] == 0:
            spec += 1
            
        if tru_lbl[i] == prd_lbl[i] == 1:
            sens += 1
            
        if tru_lbl[i] == 0:
            len_neg += 1 
            
        if tru_lbl[i] == 1:
            len_pos += 1
    
    accu  = (spec+sens)/len(tru_lbl)
    sens = sens/len_pos
    spec = spec/len_neg
    
    return sens, spec, accu

def cluster(data):
    compactness,labels,centers = cv2.kmeans(dat,2,None,criteria,10,flags)
    return labels, centers 

def format_metric(x, std = False):
    if std:
        return np.round(np.std(x)*100, 2)
    else:
        return np.round(np.mean(x)*100, 2)