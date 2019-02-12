import os, sys
from os import listdir
from os.path import isfile, join
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier

def calculatePrecisionRecallAccuracy(labels, outputs):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for label, output in zip(labels, outputs):
        if label==output and label ==0:
            tn = tn+1
        elif label==output and label==1:
            tp = tp+1
        elif label!=output and label == 0:
            fp = fp+1
        else:
            fn = fn +1
    if tp+fp ==0:
       precision = 0
    else:
       precision = tp*1.0/(tp+fp)
    if tp+fn ==0:
       precision= 0
    else:
       recall = tp*1.0/(tp+fn)
    if tp+fp+fn+tn ==0:
       accuracy = 0
    else:
       accuracy = (tp+tn)*1.0/(tp+fp+fn+tn)
    return  precision, recall, accuracy


def printcustom(number, msg):
    sys.stdout.write(msg+'{:d}'.format(number))
    sys.stdout.flush()


def getfiles(path):
    files = [os.path.join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return files

def getFeatures(path):
    files = getfiles(path)
    features = []
    for idx, f in enumerate(files):
        #print("\r processed {}.".format(idx), end="")
        features.append(pickle.load(open(f,'rb')).data.cpu().numpy().astype(float))
    return np.vstack(features)


def classifier(param):
#    return DecisionTreeClassifier(random_state=0)
#    return svm.LinearSVC(random_state=0)
#    clf = svm.SVC(gamma='scale', random_state=0, C=param, class_weight='balanced')
    clf = AdaBoostClassifier(n_estimators=500)
    print(clf)
    return clf #svm.NuSVC(kernel='rbf')


def train_classifier(train_directory, param):
    train_pos_features = getFeatures(os.path.join(train_directory, 'z-malasma'))
    train_pos_labels = [1]*len(train_pos_features)

    train_neg_features = getFeatures(os.path.join(train_directory, 'no-malasma'))
    train_neg_labels = [0]*len(train_neg_features)

    cls = classifier(param)

    weights = [0.1]*len(train_pos_features)+[5.0]*len(train_neg_features)
    cls.fit(np.vstack((train_pos_features,train_neg_features)).astype(float), np.array(train_pos_labels+train_neg_labels).astype(float))

    return cls

def test_classifer(test_directory, cls):
    test_pos_features = getFeatures(os.path.join(test_directory, 'z-malasma'))
    test_pos_labels = [1]*len(test_pos_features)
    
    test_neg_features = getFeatures(os.path.join(test_directory, 'no-malasma'))
    test_neg_labels = [0]*len(test_neg_features)
    
    
    predictions = cls.predict(np.vstack((test_pos_features, test_neg_features)))

    labels = test_pos_labels+test_neg_labels
    
    return predictions, labels


if __name__=="__main__":

   for param in range(0,1):
       c = pow(2,param)*1.0
       print("c:", c)
       cls = train_classifier('malasma_unbalanced_features_train', c)
       predictions, labels = test_classifer('malasma_unbalanced_features_test', cls)

       precision, recall, accuracy = calculatePrecisionRecallAccuracy(labels, predictions)    
       matrix = confusion_matrix(predictions, labels)
       print(matrix)
       print(precision, recall, accuracy)
