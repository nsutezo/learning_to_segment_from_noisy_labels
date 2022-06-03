import numpy as np
import tensorflow.keras.backend as K  



def custom_iouloss(y_true, y_pred):
    tp = K.sum(K.abs(y_true*y_pred),axis=[1,2,3])
    union =  K.sum(y_true,[1,2,3]) + K.sum(y_pred,[1,2,3]) - tp
    iou =  1 - K.expand_dims(K.mean(tp/union, axis=0),0)
    return iou

def custom_jaccard(y_true, y_pred,smooth=100):
    tp = K.sum(K.abs(y_true*y_pred),axis=[1,2,3]) * 2  + smooth
    union =  K.sum(K.abs(y_true),[1,2,3])+ K.sum(K.abs(y_pred),[1,2,3]) + smooth
    iou = (1 -  K.expand_dims(K.mean(tp/union, axis=0),0)) * smooth
    #import ipdb; ipdb.set_trace()
    return iou


def custom_iou(y_true, y_pred):
    #import ipdb; ipdb.set_trace()
    tp = K.sum(K.abs(y_true*y_pred),axis=[1,2,3])
    union =  K.sum(y_true,[1,2,3])+ K.sum(y_pred,[1,2,3]) - tp
    
    iou = K.expand_dims(K.mean(tp/union, axis=0),0)
    #import ipdb; ipdb.set_trace()
    return iou
        

def custom_dice(y_true, y_pred):
    tp = K.sum(K.abs(y_true*y_pred),axis=[1,2,3]) * 2
    union =  K.sum(y_true,[1,2,3])+ K.sum(y_pred,[1,2,3])

    iou = K.expand_dims(K.mean(tp/union, axis=0),0)
    #import ipdb; ipdb.set_trace()
    return iou


def custom_jaccard_metric_numpy(y_true, y_pred,smooth=100):
    tp = np.sum(np.abs(y_true*y_pred)) * 2  
    union =  np.sum(np.abs(y_true))+ np.sum(np.abs(y_pred)) 
    iou = np.mean(tp/union, axis=0)
    #import ipdb; ipdb.set_trace()
    return iou


def custom_iou_numpy(y_true, y_pred):
    tp = np.sum(y_true*y_pred)
    union =  np.sum(y_true)+ np.sum(y_pred) - tp
    iou = tp/union
    return iou