import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier,GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from numpy import interp
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier,GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
# used for normalization
from sklearn.preprocessing import StandardScaler

# used for cross-validation
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd
import copy

def plot_roc_curve(classifier,  X_train, X_test, y_train, y_test, name, k, n_classes = 4):
    # Binarize the output
    try:
        y_score = classifier.decision_function(X_test)
    except:
        y_score = classifier.predict_proba(X_test)
        pass
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(0,n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test.iloc[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(np.array(y_test).ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(0,n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111)
    ax.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=10)

    ax.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=15)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color,lw=15,
                 label='(area = {%.2f})' %(roc_auc[i]))

    plt.xlim([-0.1, 1.0])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Classifier %s Kfold %i'%(name, k))
    plt.legend(loc="lower right")
    ax.grid()
    plt.savefig("/data/projects/2020/OvarianCancerHH/TCGA_OV_SubGroup_Features/ROC/TCGA_OV_ROC_%s_%i.png"%(name,k))
    
    
def sort_abs(df):
    df = df.reindex(df.abs().sort_values(ascending=False).index)
    return df

def rf(X_train, y_train, X_test, y_test, lFeature = None):
    
    md = OneVsRestClassifier(RandomForestClassifier(n_estimators=1000))

    # let's normalize, anyway
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    GroupFeature = []
    md.fit(X_train, y_train)
    scoreTraining = md.score(X_train, y_train)
    scoreTest = md.score(X_test, y_test)
    
    
    for estimator in md.estimators_:   
        feature = pd.Series(estimator.feature_importances_, index = lFeature)
        feature = sort_abs(feature)
        #print(feature)
        GroupFeature.append(feature)
    
    return scoreTraining, scoreTest, GroupFeature, md



def bagging(X_train, y_train, X_test, y_test, lFeature = None):
    
    md = OneVsRestClassifier(BaggingClassifier(n_estimators=1000))

    # let's normalize, anyway
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    GroupFeature = []

    md.fit(X_train, y_train)
    scoreTraining = md.score(X_train, y_train)
    scoreTest = md.score(X_test, y_test)
    
    for estimator in md.estimators_: 
        feature_importances = np.mean([
                tree.feature_importances_ for tree in estimator.estimators_
            ], axis=0)
        feature = pd.Series(feature_importances, index = lFeature)
        feature = sort_abs(feature)
        GroupFeature.append(feature)


    return scoreTraining, scoreTest, GroupFeature, md


def ada(X_train, y_train, X_test, y_test, lFeature = None):
    
    md = OneVsRestClassifier(AdaBoostClassifier(n_estimators=1000))

    # let's normalize, anyway
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    GroupFeature = []

    md.fit(X_train, y_train)
    scoreTraining = md.score(X_train, y_train)
    scoreTest = md.score(X_test, y_test)
    
    for estimator in md.estimators_:   
        feature = pd.Series(estimator.feature_importances_, index = lFeature)
        feature = sort_abs(feature)
        #print(feature)
        GroupFeature.append(feature)

    return scoreTraining, scoreTest, GroupFeature, md


def lr(X_train, y_train, X_test, y_test, lFeature = None):
    
    md = OneVsRestClassifier(LogisticRegression(penalty="l2", max_iter=100000))
    # let's normalize, anyway
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    GroupFeature = []

    md.fit(X_train, y_train)
    scoreTraining = md.score(X_train, y_train)
    scoreTest = md.score(X_test, y_test)
    
    for estimator in md.estimators_:   
        feature = pd.Series(estimator.coef_[0], index = lFeature)
        feature = sort_abs(feature)
        #print(feature)
        GroupFeature.append(feature)


    return scoreTraining, scoreTest, GroupFeature, md


def multilabel_feature_selection(data, k = 10, topk = 100, group = "Group", dGroup = None):
    # list of classifiers, selected on the basis of our previous paper "
    
    modelList = [
                [bagging,"Bagging Classifier",],
                [rf,"Random Forest",],
                [lr,"Logistic Regresion",],
                 #[ridge,"Ridge Classfier",],
                 #[svm,"Support Vector Machine Classfier",],
                 [ada, "Ada Classifier"],
                 ]
    
    print("Loading dataset...")
    
    X, Y = data.drop(group, axis =1), data[group]
    Y = pd.DataFrame(label_binarize(Y, classes = Y.unique().tolist()))
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    indexes = [ (training, test) for training, test in skf.split(X, data[group]) ]
    lFeature = data.drop(group, axis =1).columns.tolist()

    topFeatures = {}
    for key in dGroup.keys():
        topFeatures[key] = pd.Series(dtype='float64', index=lFeature).fillna(0)

    lAll = []
    DictScore = {}
    
    for model, name in modelList :
        print("\nClassifier " + name)
        ListScore = []
        classifierTopFeatures = pd.Series(dtype='float64', name = name, index=lFeature).fillna(0)

        # iterate over all folds
        j = 0
        for train_index, test_index in indexes:
            X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
            y_train, y_test = Y.iloc[train_index,:], Y.iloc[test_index,:]

            classifier = copy.deepcopy(model)
            scoreTraining, scoreTest, GroupFeature, classifier = classifier(X_train, y_train,\
                                                X_test, y_test, lFeature = lFeature)


            print("\ttraining: %.4f, test: %.4f" % (scoreTraining, scoreTest))
            ListScore.append( scoreTest )
            # now, let's get a list of the most important features, then mark the ones in the top X
            #print(len(GroupFeature))
            for i in range(0,len(GroupFeature)):
                orderedFeatures = GroupFeature[i]
                lF = orderedFeatures.index[0:topk].tolist()
                #print(lF)
                for f in lF:

                    feature = f
                    #print(i)
                    #print(f)
                    topFeatures[list(dGroup.keys())[i]][ feature ] += 1
                    classifierTopFeatures[ feature ] += 1
            try:
                plot_roc_curve(classifier,  X_train, X_test, y_train, y_test, name, j, n_classes = 4)
            except Exception as error:
                print("[-]%s"%error)
                pass
            j += 1

        print("\ttest mean: %.4f" % (np.mean(ListScore)))
        DictScore[name] = np.mean(ListScore)
        lAll.append(classifierTopFeatures)
    
    dPerc = {}
    for key in dGroup.keys():
        feature_per = topFeatures[key].div(len(modelList)*k)*100
        dPerc[key] = feature_per.sort_values(ascending=False)[:topk]
        
    dAll = pd.DataFrame(lAll).div(k*len(dGroup.keys()))*100
    
    return dPerc, dAll, DictScore