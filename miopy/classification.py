from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier,GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
# used for normalization
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
# used for cross-validation
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd
import copy

def rf(X_train, y_train, X_test, y_test, lFeature = None):
    
    md = RandomForestClassifier(n_estimators=1000)

    # let's normalize, anyway
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    md.fit(X_train, y_train)
    y_predict = md.predict(X_test)
    scoreTraining = md.score(X_train, y_train)
    scoreTest = md.score(X_test, y_test)
    feature = pd.Series(md.feature_importances_, index = lFeature)
    md.feature_names = lFeature

    return scoreTraining, scoreTest, y_predict, feature, md


def lr(X_train, y_train, X_test, y_test, lFeature = None):
    
    md = LogisticRegression(penalty="l2", max_iter=100000)
    # let's normalize, anyway
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    md.fit(X_train, y_train)
    y_predict = md.predict(X_test)

    scoreTraining = md.score(X_train, y_train)
    scoreTest = md.score(X_test, y_test)
    feature = pd.Series(md.coef_[0], index = lFeature)
    md.feature_names = lFeature

    return scoreTraining, scoreTest, y_predict, feature, md


def ridge(X_train, y_train, X_test, y_test, lFeature = None):
    
    md = RidgeClassifier(max_iter=10000)

    # let's normalize, anyway
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    md.fit(X_train, y_train)
    scoreTraining = md.score(X_train, y_train)
    scoreTest = md.score(X_test, y_test)
    feature = pd.Series(md.coef_[0], index = lFeature)
    md.feature_names = lFeature

    return scoreTraining, scoreTest, feature, md



def svm(X_train, y_train, X_test, y_test, lFeature = None):
    
    md = SVC(kernel='linear')

    # let's normalize, anyway
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    md.fit(X_train, y_train)
    y_predict = md.predict(X_test)

    scoreTraining = md.score(X_train, y_train)
    scoreTest = md.score(X_test, y_test)
    scoreTraining = md.score(X_train, y_train)
    scoreTest = md.score(X_test, y_test)
    feature = pd.Series(md.coef_[0], index = lFeature)
    md.feature_names = lFeature

    return scoreTraining, scoreTest, y_predict, feature, md





def classification_cv(data, k = 10, name = "Random Forest", group = "event", lFeature = None):
    # list of classifiers, selected on the basis of our previous paper "
    from sklearn.metrics import roc_auc_score, roc_curve

    modelList = { "Random Forest": rf,
                "Logistic Regression": lr,
                "Support Vector Machine": svm}
    
    print("Loading dataset...")
    
    lFeature = list(set(lFeature).intersection(data.columns.tolist()))
    X, Y = data[lFeature], label_binarize(data[group], data[group].unique().tolist())[:,0]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=101)
    indexes = [ (training, test) for training, test in skf.split(X, Y) ]
        
    model = modelList[name]
    results = { 'model': name,
    'auc': [],
    'fpr': [],
    'tpr': [],
    'train': [],
    'test' : [],
    'classifier' : []
    } 

    print("\nClassifier " + name)
    # iterate over all folds
    featureDf = pd.DataFrame()
    for train_index, test_index in indexes:
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = Y[train_index], Y[test_index]

        classifier = copy.deepcopy(model)
        scoreTraining, scoreTest, y_predict, feature, model_fit = classifier(X_train, y_train,\
                                                X_test, y_test, lFeature = lFeature)

        print("\ttraining: %.4f, test: %.4f" % (scoreTraining, scoreTest))

        fpr, tpr, thresholds = roc_curve(y_test, y_predict)
        results['auc'].append(roc_auc_score(y_test, y_predict))
        results['fpr'].append(fpr)
        results["tpr"].append(tpr)
        results["train"].append(scoreTraining)
        results["test"].append(scoreTest)
        results["classifier"].append(model_fit)
        featureDf = pd.concat([featureDf,feature], axis = 1)

    #print(featureDf.mean(axis = 1))
    results["feature"] = featureDf
    print("\tTest Mean: %.4f" % (np.mean(results['test'])))

    return results




def classification_training_model(data, model = None, group = "event", lFeature = None):
    # list of classifiers, selected on the basis of our previous paper "
    from sklearn.metrics import roc_auc_score, roc_curve


    
    print("Loading dataset...")
    
    lFeature = list(set(lFeature).intersection(data.columns.tolist()))
    print(data.head())
    print(lFeature)
    X, Y = data[lFeature], label_binarize(data[group], data[group].unique().tolist())[:,0]
    name = type(model).__name__
    results = { 'model': name,
    'auc': [],
    'fpr': [],
    'tpr': [],
    'train': [],
    'test' : [],
    'classifier' : []
    } 

    print("\nClassifier " + name)
    # iterate over all folds
    print(X)
    print(model)

    y_predict = model.predict(X)
    print("predicted")
    scoreTraining = None
    scoreTest = model.score(X, Y)
    scoreTraining = model.score(X, Y)
    scoreTest = model.score(X, Y)
    
    try:
        feature = pd.Series(model.coef_[0], index = lFeature)
    except:
        feature = pd.Series(model.feature_importances_, index = lFeature)

    fpr, tpr, thresholds = roc_curve(Y, y_predict)
    results['auc'].append(roc_auc_score(Y, y_predict))
    results['fpr'].append(fpr)
    results["tpr"].append(tpr)
    results["train"].append(scoreTraining)
    results["test"].append(scoreTest)
    results["feature"] = feature


    return results