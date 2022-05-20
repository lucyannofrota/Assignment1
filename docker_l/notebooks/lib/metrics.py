from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def accuracy(TP, TN, number_of_samples):
    return (TP + TN)/number_of_samples

def precision(TP, FP):
    return TP/(TP+FP)

def recall(TP, FN):
    return TP/(TP+FN)

def AUC():
    pass

def confusion_matrix(training_set, labels):
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    for i in len(training_set):
        if training_set[i].label == 1:
            if labels[i] == 1:
                TP = 1
            else:
                FP = 1
        else:
            if labels[i] == 1:
                FN = 1
            else:
                TN = 1
    return TP, FP, FN, TN

def get_metrics(training_set, labels):
    TP, FP, FN, TN = confusion_matrix(training_set, labels)

    acc = accuracy(TP, TN, len(training_set))
    pre = precision(TP, FP)
    re = recall(TP, FN)
    auc = None #AUC
    return acc, pre, re, auc

def metrics(prediction, ground_truth):
    accuracy = accuracy_score(y_true=ground_truth, y_pred=prediction)
    roc = roc_auc_score(ground_truth, prediction)
    pre = precision_score(ground_truth, y_pred=prediction)
    rec = recall_score(y_true=ground_truth, y_pred=prediction)


    print("Metrics: \n Acc: {acc:.3f}, ROC: {roc:.3f}\n PRE: {pre:.3f}, REC: {rec:.3f}".format(acc=accuracy,roc=roc,pre=pre,rec=rec))