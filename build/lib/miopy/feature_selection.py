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

def sort_abs(df):
    df = df.reindex(df.abs().sort_values(ascending=False).index)
    return df

def rf(X_train, y_train, X_test, y_test, lFeature = None, seed = 123):
    seed = np.random.RandomState(seed)
    md = RandomForestClassifier(n_estimators=300, random_state = seed)

    # let's normalize, anyway
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    md.fit(X_train, y_train)
    scoreTraining = md.score(X_train, y_train)
    scoreTest = md.score(X_test, y_test)
    feature = pd.Series(md.feature_importances_, index = lFeature)
    feature = sort_abs(feature)
    
    return scoreTraining, scoreTest, feature


def gbc(X_train, y_train, X_test, y_test, lFeature = None, seed = 123):
    seed = np.random.RandomState(seed)
    md = GradientBoostingClassifier(n_estimators=300, random_state = seed)

    # let's normalize, anyway
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    md.fit(X_train, y_train)
    scoreTraining = md.score(X_train, y_train)
    scoreTest = md.score(X_test, y_test)
    feature = pd.Series(md.feature_importances_, index = lFeature)
    feature = sort_abs(feature)
    
    return scoreTraining, scoreTest, feature


def ada(X_train, y_train, X_test, y_test, lFeature = None, seed = 123):
    seed = np.random.RandomState(seed)
    md = AdaBoostClassifier(n_estimators=300, random_state = seed)

    # let's normalize, anyway
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    md.fit(X_train, y_train)
    scoreTraining = md.score(X_train, y_train)
    scoreTest = md.score(X_test, y_test)
    feature = pd.Series(md.feature_importances_, index = lFeature)
    feature = sort_abs(feature)
    
    return scoreTraining, scoreTest, feature

def lr(X_train, y_train, X_test, y_test, lFeature = None, seed = 123):
    seed = np.random.RandomState(seed)
    md = LogisticRegression(penalty="l2", max_iter=10000, random_state = seed)
    # let's normalize, anyway
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    md.fit(X_train, y_train)
    scoreTraining = md.score(X_train, y_train)
    scoreTest = md.score(X_test, y_test)
    feature = pd.Series(md.coef_[0], index = lFeature)
    feature = sort_abs(feature)

    return scoreTraining, scoreTest, feature


def ridge(X_train, y_train, X_test, y_test, lFeature = None, seed = 123):
    seed = np.random.RandomState(seed)
    md = RidgeClassifier(max_iter=10000, random_state = seed)

    # let's normalize, anyway
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    md.fit(X_train, y_train)
    scoreTraining = md.score(X_train, y_train)
    scoreTest = md.score(X_test, y_test)
    feature = pd.Series(md.coef_[0], index = lFeature)
    feature = sort_abs(feature)

    return scoreTraining, scoreTest, feature



def svm(X_train, y_train, X_test, y_test, lFeature = None, seed = 123):
    seed = np.random.RandomState(seed)
    md = SVC(kernel='linear', random_state = seed)

    # let's normalize, anyway
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    md.fit(X_train, y_train)
    scoreTraining = md.score(X_train, y_train)
    scoreTest = md.score(X_test, y_test)
    feature = pd.Series(md.coef_[0], index = lFeature)
    feature = sort_abs(feature)

    return scoreTraining, scoreTest, feature


def bagging(X_train, y_train, X_test, y_test, lFeature = None, seed = 123):
    seed = np.random.RandomState(seed)
    md = BaggingClassifier(n_estimators=300, random_state = seed)

    # let's normalize, anyway
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    md.fit(X_train, y_train)
    scoreTraining = md.score(X_train, y_train)
    scoreTest = md.score(X_test, y_test)
    feature_importances = np.mean([
            tree.feature_importances_ for tree in md.estimators_
        ], axis=0)
    feature = pd.Series(feature_importances, index = lFeature)
    feature = sort_abs(feature)

    return scoreTraining, scoreTest, feature


def feature_selection(data, k = 10, topk = 100, group = "Group", seed = 123):
    # list of classifiers, selected on the basis of our previous paper "
    
    modelList = [[rf,"Random Forest",],
                 [lr,"Logistic Regresion",],
                 [ridge,"Ridge Classfier",],
                 [svm,"Support Vector Machine Classfier",],
                 [ada, "Ada Classifier"],
                 [bagging,"Bagging Classifier",],
                 [gbc,"Gradient Boosting Classifier",]

                 ]
    
    print("Loading dataset...")
    
    X, Y = data.drop(group, axis =1), label_binarize(data[group], classes = data[group].unique().tolist())[:,0]
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=np.random.RandomState(seed))
    indexes = [ (training, test) for training, test in skf.split(X, Y) ]
    
    lFeature = data.drop(group, axis =1).columns.tolist()
    topFeatures = pd.Series(dtype='float64', index=lFeature).fillna(0)
    lAll = []
    DictScore = {}
    
    for model, name in modelList :
        print("\nClassifier " + name)
        ListScore = []
        classifierTopFeatures = pd.Series(dtype='float64', name = name, index=lFeature).fillna(0)

        # iterate over all folds
        for train_index, test_index in indexes:
            X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
            y_train, y_test = Y[train_index], Y[test_index]

            classifier = copy.deepcopy(model)
            scoreTraining, scoreTest, orderedFeatures = classifier(X_train, y_train,\
                                                X_test, y_test, lFeature = lFeature, seed = seed)


            print("\ttraining: %.4f, test: %.4f" % (scoreTraining, scoreTest))
            ListScore.append( scoreTest )
            # now, let's get a list of the most important features, then mark the ones in the top X
            lF = orderedFeatures.index[0:topk].tolist()

            for f in lF:

                feature = f

                topFeatures[ feature ] += 1
                classifierTopFeatures[ feature ] += 1



        print("\ttest mean: %.4f" % (np.mean(ListScore)))
        DictScore[name] = np.mean(ListScore)
        lAll.append(classifierTopFeatures)
        
    feature_per = topFeatures.div(len(modelList)*k)*100
    dAll = pd.DataFrame(lAll).div(k)*100
    
    return feature_per.sort_values(ascending=False)[:topk], dAll, DictScore