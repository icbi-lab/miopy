import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from multiprocessing import Pool
import numpy as np
import functools
from .correlation import intersection, header_list
import plotly
import plotly.offline as opy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, GridSearchCV
import warnings

#######################
### Sklearn Survival ##
#######################

class EarlyStoppingMonitor:

    def __init__(self, window_size, max_iter_without_improvement):
        self.window_size = window_size
        self.max_iter_without_improvement = max_iter_without_improvement
        self._best_step = -1

    def __call__(self, iteration, estimator, args):
        # continue training for first self.window_size iterations
        if iteration < self.window_size:
            return False

        # compute average improvement in last self.window_size iterations.
        # oob_improvement_ is the different in negative log partial likelihood
        # between the previous and current iteration.
        start = iteration - self.window_size + 1
        end = iteration + 1
        improvement = np.mean(estimator.oob_improvement_[start:end])

        if improvement > 1e-6:
            self._best_step = iteration
            return False  # continue fitting

        # stop fitting if there was no improvement
        # in last max_iter_without_improvement iterations
        diff = iteration - self._best_step
        return diff >= self.max_iter_without_improvement


def IPC_RIDGE(X_train, y_train, X_test, y_test, lFeature = None,  n_core = 2, seed = 123):
    from sksurv.linear_model import IPCRidge
    from sklearn.pipeline import make_pipeline

    # let's normalize, anyway
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    seed = np.random.RandomState(seed)

    y_train_log = y_train.copy()
    y_train_log["time"] = np.log1p(y_train["time"])
    y_test_log = y_test.copy()
    y_test_log["time"] = np.log1p(y_test["time"])
    #https://github.com/sebp/scikit-survival/issues/41
    
    n_alphas = 50
    alphas = np.logspace(-10, 1, n_alphas)

    
    gcv = GridSearchCV(IPCRidge(max_iter=100000),
    {"alpha":alphas},
    cv = 2,
    n_jobs=10).fit(X_train,y_train_log)

    best_model = gcv.best_estimator_.named_steps["IPCRidge"]
    

    alpha = best_model.alphas_
    scoreTraining = best_model.score(X_train,y_train_log)
    scoreTest = best_model.score(X_test,y_test_log)
    feature = pd.DataFrame(best_model.coef_, index=lFeature)[0]
    
    return scoreTraining, scoreTest, feature


def score_survival_model(model, X, y):
    from sksurv.metrics import concordance_index_censored
    prediction = model.predict(X)
    result = concordance_index_censored(y['event'], y['time'], prediction)
    return result[0]

def SurvivalSVM(X_train, y_train, X_test, y_test, lFeature = None,  n_core = 2, seed = 123):
    from sksurv.svm import FastSurvivalSVM
    import numpy as np

    # let's normalize, anyway
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    seed = np.random.RandomState(seed)
    ssvm = FastSurvivalSVM(max_iter=100, tol=1e-5, random_state=seed)


    param_grid = {'alpha': 2. ** np.arange(-12, 13, 4)}
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=seed)
    gcv = GridSearchCV(ssvm, param_grid, scoring=score_survival_model,
                       n_jobs = n_core , refit=False,
                       cv=cv)
    warnings.filterwarnings("ignore", category=FutureWarning)
    gcv = gcv.fit(X_train, y_train)
    
    ssvm.set_params(**gcv.best_params_)
    ssvm.fit(X_train, y_train)
    
    scoreTraining = ssvm.score(X_train,y_train)
    scoreTest = ssvm.score(X_test,y_test)
    feature = pd.Series(ssvm.coef_, index=lFeature)
    
    return scoreTraining, scoreTest, feature


def PenaltyCox(X_train, y_train, X_test, y_test, lFeature = None,  n_core = 2, seed = 123):
    from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
    from sklearn.pipeline import make_pipeline
    seed = np.random.RandomState(seed)
    # let's normalize, anyway
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = CoxnetSurvivalAnalysis(alpha_min_ratio=0.12, l1_ratio=0.9, max_iter=100, random_state = seed)
    #https://github.com/sebp/scikit-survival/issues/41
    
    model.set_params(max_iter = 100, n_alphas = 50)
    model.fit(X_train, y_train)
    warnings.simplefilter("ignore", ConvergenceWarning)

    alphas = model.alphas_

    
    gcv = GridSearchCV(
    make_pipeline(CoxnetSurvivalAnalysis(l1_ratio=0.9, max_iter=1000, random_state = seed)),
    param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in alphas]},
    cv = 2,
    n_jobs= n_core).fit(X_train,y_train)

    best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
    

    alpha = best_model.alphas_
    scoreTraining = best_model.score(X_train,y_train)
    scoreTest = best_model.score(X_test,y_test)
    feature = pd.DataFrame(best_model.coef_, index=lFeature)[0]
    
    return scoreTraining, scoreTest, feature


def SurvivalForest(X_train, y_train, X_test, y_test, lFeature = None,  n_core = 2, seed = 123):
    from sksurv.ensemble import RandomSurvivalForest
    from eli5.formatters import format_as_dataframe
    from eli5.sklearn import explain_weights_sklearn
    from eli5.sklearn import PermutationImportance

    # let's normalize, anyway
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    seed = np.random.RandomState(seed)
    rsf = RandomSurvivalForest(n_estimators=300,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           max_features="sqrt",
                           n_jobs= n_core,
                           random_state=seed)
    
    rsf.fit(X_train, y_train)
    scoreTraining = rsf.score(X_train,y_train)
    scoreTest = rsf.score(X_test,y_test)
    
    perm = PermutationImportance(rsf, n_iter=3, random_state=seed)
    perm.fit(X_test, y_test)
    feature = format_as_dataframe(explain_weights_sklearn(perm, feature_names=lFeature, top = len(lFeature) ))
    feature = pd.Series(feature["weight"].tolist(), index=feature["feature"].tolist())
     
    #feature = pd.DataFrame(rsf.feature_importances_, index=lFeature)

    return scoreTraining, scoreTest, feature


    
def gradient_boosted_models(X_train, y_train, X_test, y_test, lFeature = None,  n_core = 2, seed = 123):
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis

    # let's normalize, anyway
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    seed = np.random.RandomState(seed)
    model = GradientBoostingSurvivalAnalysis(
        n_estimators=1000, learning_rate=0.05, subsample=0.5,
        max_depth=1, random_state=seed
    )

    monitor = EarlyStoppingMonitor(25, 100)
    model.fit(X_train, y_train, monitor=monitor)


    scoreTraining = model.score(X_train,y_train)
    scoreTest = model.score(X_test,y_test)
    
    feature = pd.Series(model.feature_importances_, index=lFeature)
     
    return scoreTraining, scoreTest, feature


def survival_selection(data, k = 10, topk = 100, event = "event",  n_core = 2, seed = 123):
    from sksurv.datasets import get_x_y
    from sklearn.model_selection import StratifiedKFold
    import copy
    from miopy.feature_selection import sort_abs
    # list of classifiers, selected on the basis of our previous paper "
    
    modelList = [
            [gradient_boosted_models,"Gradient Boosted Models"],
            [SurvivalSVM,"Support Vector Machine"],
            #[SurvivalForest,"Random Forest",],
            [PenaltyCox,"Penalized Cox",]
            ]
    
    print("Loading dataset...")
    
    X, Y = get_x_y(data, attr_labels = [event,"time"], pos_label=0)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state = np.random.RandomState(seed))
    indexes = [ (training, test) for training, test in skf.split(X, Y) ]
    
    lFeature = X.columns.tolist()
    topFeatures = pd.Series(dtype='float64', index=lFeature).fillna(0)
    lAll = []
    DictScore = {}
    dfTopCoef = pd.DataFrame(dtype='float64', index=lFeature).fillna(0)    

    for model, name in modelList :
        print("\nClassifier " + name)
        ListScore = []
        classifierTopFeatures = pd.Series(dtype='float64', name = name, index=lFeature).fillna(0)
        
        dfTopCoefTemp = pd.DataFrame(dtype='float64', index=lFeature).fillna(0)    
        i = 1
        # iterate over all folds
        for train_index, test_index in indexes :
            X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
            y_train, y_test = Y[train_index], Y[test_index]

            try:
                classifier = copy.deepcopy(model)
                scoreTraining, scoreTest, features = classifier(X_train, y_train,\
                                                X_test, y_test, lFeature = lFeature, n_core = n_core, seed = seed)

            except Exception as error:
                print(error)
            
            else:
                print("\ttraining: %.4f, test: %.4f" % (scoreTraining, scoreTest))
                ListScore.append( scoreTest )
                # now, let's get a list of the most important features, then mark the ones in the top X
                orderedFeatures = sort_abs(features[features != 0]).round(3)
                
                if topk <= len(orderedFeatures):
                    lF = orderedFeatures.index[0:topk].tolist()
                else:
                    lF = orderedFeatures.index.tolist()
                dfTopCoefTemp.loc[:, i] = orderedFeatures

                for f in lF:
                    if orderedFeatures[f] != 0:
                        topFeatures[f] += 1
                        classifierTopFeatures[ f ] += 1

            finally:
                i +=1
        
        dfTopCoef[name] = dfTopCoefTemp.apply(lambda row: row.mean(), axis=1)



        print("\ttest mean: %.4f" % (np.mean(ListScore)))
        DictScore[name] = np.mean(ListScore)
        lAll.append(classifierTopFeatures)
        
    feature_per = topFeatures.div(len(modelList)*k)*100
    feature_per = feature_per.sort_values(ascending=False)[:topk]
    dAll = pd.DataFrame(lAll).div(k)*100
     
    return feature_per, dAll, DictScore, dfTopCoef


########################
### Survival Analysis ##
########################

def get_exprs_cutoff(exprDF, target="hsa-miR-223-3p", q = 0.5, treshold = None, optimal = True):
    from scipy import stats

    if optimal:
        q, treshold  = get_survival_cutoff(exprDF = exprDF, time = "time", event = "event", target = target)
    else:
        if treshold != None:
            q = stats.percentileofscore(exprDF[target],treshold)/100
        else:
            treshold = exprDF[target].quantile(q)

    return q, treshold

def split_by_exprs(exprDF, target="hsa-miR-223-3p", treshold = 0.5):

    exprDF["exprs"] = None
    is_higher = exprDF[target] >= float(treshold)
    exprDF["exprs"] = exprDF["exprs"].mask(is_higher, 1)
    exprDF["exprs"] = exprDF["exprs"].mask(~is_higher, 0)
    #print("Splitted")
    return exprDF

def get_survival_cutoff(exprDF = "exprDF", time = "time", event = "event", target = "target"):
    lPoint = exprDF[target].unique().tolist()
    
    df = pd.DataFrame()
    for point in lPoint:
        q, treshold = get_exprs_cutoff(exprDF, target=target, treshold = point, optimal = False)
        if 0.1 < q < 0.9:
            try:
                tRes = get_hazard_ratio(split_by_exprs(exprDF, target=target, treshold = treshold))
            except Exception as error:
                print(error)
                tRes = (0, 1,)
            dfTemp = pd.Series({"Target":target,"Q":q,"Cutpoint":treshold,"HR":tRes[0],"pval":tRes[1]})
            df = pd.concat([df,dfTemp], axis = 1)
    df = df.transpose()
    df["P_ADJ"] = df.pval.apply(lambda x: -1.63 * x * (1 + 2.35 * np.log(x)))
    df = df.sort_values("P_ADJ")
    row = df.iloc[0,:]
    return row["Q"], row["Cutpoint"]

def get_hazard_ratio(exprDF, target = "exprs"):
    np.seterr(divide='ignore', invalid='ignore')

    cph = CoxPHFitter()
    cph.fit(exprDF[[target,"time","event"]].dropna(), "time", event_col = "event")
    pval = cph.summary["p"][target]
    hr_high, hr_low = cph.summary["exp(coef) upper 95%"][target], cph.summary["exp(coef) lower 95%"][target]
    log_hr = cph.summary["exp(coef)"][target]  
    #print(cph.summary)
    return (log_hr, pval, hr_high, hr_low)


def obatin_hr(ltarget, exprDF = None):
    
    lhr = []
    for target in ltarget:
        try:
            q, treshold = get_exprs_cutoff(exprDF, target=target, q=0.5, optimal = False)
            print(q), print(treshold)

            tRes = get_hazard_ratio(split_by_exprs(exprDF, target=target, treshold = treshold))
            print("%s"%(target))
            print(tRes)
            hr = tRes[0]
        except Exception as error:
            print(error)
            hr = 1
        finally:
            lhr.append(hr)
      
    df = pd.DataFrame({"target":ltarget,"log(hr)":lhr})
    return df

def obatin_hr_by_exprs(ltarget, exprDF = None):
    
    lhr = []
    for target in ltarget:
        try:
            tRes = get_hazard_ratio(exprDF, target =  target)
            hr = tRes[0]
        except Exception as error:
            print(error)
            hr = 0 
        finally:
            lhr.append(hr)
    print("Lista HR")
    print(lhr)
    print(len(ltarget)), print(len(lhr))        
    df = pd.DataFrame({"target":ltarget,"log(hr)":lhr})
    print("DF inside obtain_hr")
    print(df)
    return df

def same_length(list_lists):
    lmax = 0
    for l in list_lists:
        lmax = max(lmax, len(l))
        new_l = []
    for l in list_lists:
        ll = len(l)
        if  ll < lmax:
            l += ["foo"] * (lmax - ll)
        new_l.append(l)
    return new_l


def hazard_ratio(lGeneUser = None, lMirUser = None, exprDF = None, n_core = 4):

    ### Intersect with Gene and Mir from table##
    lMir, lGene = header_list(exprDF=exprDF)
    if lGeneUser is not None:
        lGene = intersection(lGene, lGeneUser)
    
    if lMirUser is not None:
        lMir = intersection(lMir, lMirUser)

    lTarget = lGene+lMir
    print(exprDF)
    ##Split List
    np_list_split = np.array_split(lTarget, n_core)
    split_list = [i.tolist() for i in np_list_split]
    #split_list = same_length(split_list)

    #Fix Exprs Variable
    partial_func = functools.partial(obatin_hr, exprDF=exprDF)

    #Generating Pool
    pool = Pool(n_core)
    lres = pool.map(partial_func, split_list)
    print("lResultados")
    print(lres)
    res = pd.concat(lres)
    pool.close() 
    pool.join()
    print(res)
    return res

