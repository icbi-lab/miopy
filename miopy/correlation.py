###############
#### AMYR #####
###############
import re
from typing import DefaultDict
from pandas.core.reshape.concat import concat
import scipy.stats    
import pandas as pd
from .R_utils import tmm_normalization, deg_edger, deg_limma_array, voom_normalization
import numpy as np
import ranky as rk
from pandarallel import pandarallel
from os import path
import io
from sklearn.model_selection import StratifiedKFold, KFold
from statsmodels.stats.multitest import multipletests
from multiprocessing import Pool
import numpy as np
import functools
from .process_data import *


##################
## Miscelaneos ###
##################


def adjust_geneset(table):
    from statsmodels.stats.multitest import multipletests

    p = table["Pval"]
    mask = np.isfinite(p)

    #Creating an empty vector
    table["FDR"] = 1
    table.loc[mask,"FDR"] = multipletests(p[mask], method="fdr_bh")[1]
    return table
    

def adjPval(df):
    from statsmodels.stats.multitest import multipletests
    
    lCor = ["Rho","R","Tau"]
    method = "fdr_bh"

    for cor in lCor:
        col = "%s_%s"  %(cor,method)
        col_raw = "%s_Pval" %(cor)

        #Pval 
        p = df[col_raw]
        mask = np.isfinite(p)

        #Creating an empty vector
        df[col] = 1

        df.loc[mask,col] = multipletests(p[mask], method=method)[1]

    return df


##################
## GeneSetScore ##
##################
def calculate_gene_set_score(expr, conf):
    #print(row)
    #print(conf)
    sum_gene_predictor = sum(expr * conf)
    sum_predictor = sum(conf)
    
    try:
        GScore = sum_gene_predictor/sum_predictor
    except:
        GScore = 0
    #print(GScore)
    return GScore


def gene_set_correlation(exprDf, lGeneSet, GeneSetName = "GeneSet", lMirUser = None, n_core = 2):

    pandarallel.initialize(verbose=1, nb_workers=n_core)

    lMir, lGene = header_list(exprDF=exprDf)

    dfConf = get_confident_df(load_matrix_counts().apply(lambda col: col.str.count("1")))
    
    ### Intersect with Gene and Mir from table##
    lGene = intersection(lGene, dfConf.index.tolist())
    lMir = intersection(lMir, dfConf.columns.tolist())

    if lGeneSet is not None:
        lGene = intersection(lGene,lGeneSet)

    if lMirUser is not None:
        lMir = intersection(lMir,lMirUser)
    
    #print(lGene)
    #print(lMir)
    #print(dfConf.loc[lGene,lMir])

    dfSetScore = dfConf.loc[lGene,lMir].parallel_apply(lambda conf: \
                      exprDf[lGene].apply(lambda expr: \
                      calculate_gene_set_score(expr, conf), \
                      axis = 1), axis = 0)
                      
    dfSetScore = dfSetScore.apply(lambda col: col.dropna()) 

    cor = dfSetScore.parallel_apply(lambda col: col.corr(exprDf[col.name],method =  \
            lambda x, y: scipy.stats.pearsonr(x, y)))
    
    cor = cor.apply(lambda col: col.dropna())
    df = pd.DataFrame(cor).transpose()

    dfPval = pd.DataFrame(df.loc[:,1])
    dfCor = pd.DataFrame(df.loc[:,0])    

    dfPval.columns = [GeneSetName]
    dfCor.columns = [GeneSetName]

    return dfCor, dfPval, dfSetScore


##################
## EdgeR       ###
##################

def differential_expression_edger(fPath, metaFile, bNormal = False, bFilter = False, paired = False, group = "event"):
    """
    Function to obtain the DEG between 2 groups
    with edgeR.

    Args:
        fPath string  Path with the raw counts
        metaFile  string  Path, the first Row is sample names, second is Group
        outPath string  File output
        bNormal Bool    Bool to normalize the data with TMM
        bFilter Bool    Bool to FIlter low expression genes

    Returns:
        df  dataframe   DataFrame with The LogFC, and pvalues for genes
    """
    DEG = deg_edger(fPath = fPath, metaFile = metaFile, bNormal = str(bNormal), \
        bFilter = str(bFilter), bPaired = str(paired), group = group)

    return DEG


def tmm_normal(fPath, bFilter=True):
    """
    Function to obtain the Voom normal Count

    Args:
        fPath string  Path with the raw counts
        outPath string  File output
        bFilter Bool    Bool to FIlter low expression genes

    Returns:
        tmm  dataframe   DataFrame with the log2(TMM) counts
    """
    tmm = tmm_normalization(fPath, str(bFilter))  
    return tmm



##################
## Limma Array  ##
##################

def differential_expression_array(fPath, metaFile, bNormal = True, bFilter = True, paired = False):
    """
    Function to obtain the DEG between 2 groups
    with Limma.

    Args:
        fPath string  Path with the raw counts
        metaFile  string  Path, the first Row is sample names, second is Group
        outPath string  File output
        bNormal Bool    Bool to normalize the data with TMM
        bFilter Bool    Bool to FIlter low expression genes

    Returns:
        df  dataframe   DataFrame with The LogFC, and pvalues for genes
    """
    DEG = deg_limma_array(fPath, metaFile, str(bNormal), str(bFilter), str(paired))

    return DEG


def voom_normal(fPath, bFilter=True):
    """
    Function to obtain the Voom normal Count

    Args:
        fPath string  Path with the raw counts
        outPath string  File output
        bFilter Bool    Bool to FIlter low expression genes

    Returns:
        tmm  dataframe   DataFrame with the log2(TMM) counts
    """
    voom = voom_normalization(fPath, str(bFilter))  
    return voom


##################
### sklearn ######
##################


def CoefLarsCV(x, y, n_core = 4):
    from sklearn.linear_model import LarsCV, Lars
    from sklearn.model_selection import train_test_split

    X_train, X_test , y_train, y_test = \
    train_test_split(x, y, test_size=0.2, random_state=1)

    ## CrossValidation
    larscv = LarsCV(cv = 5, normalize=True)
    larscv.fit(X_train, y_train)

    coef = pd.Series(larscv.coef_, index = x.columns)

    return coef 


def CoefLassoCV(X, Y, k = 5, n_core = 4):
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.model_selection import train_test_split

    skf = KFold(n_splits=k, shuffle=True)
    indexes = [ (training, test) for training, test in skf.split(X, Y) ]
        # iterate over all folds

    dfTopCoefTemp = pd.DataFrame(dtype='float64', index=X.columns).fillna(0)        
    for train_index, test_index in indexes:
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = Y[train_index], Y[test_index]
        ## CrossValidation
        lassocv = LassoCV(cv = 5, max_iter=1000, normalize=True)
        lassocv.fit(X_train, y_train)
        lasso = Lasso(max_iter = 1e4, alpha=lassocv.alpha_).fit(X_train, y_train)
        dfTopCoefTemp = pd.concat([dfTopCoefTemp, pd.Series(lasso.coef_, index = X.columns).fillna(0)], axis = 1)
        #print(dfTopCoefTemp.apply(lambda row: row.mean(), axis=1))
    return dfTopCoefTemp.apply(lambda row: row.mean(), axis=1) 


def CoefLasso(x, y):
    from sklearn.linear_model import Lasso

    alphas = [0.001, 0.02, 0.01, 0.1, 0.5, 1, 5]

    lasso = Lasso(alpha = 1, max_iter = 1e4 ).fit(x, y)

    coef = pd.Series(lasso.coef_, index=x.columns)
    #coef = coef.sort_values(0, ascending=False)

    return coef 

def CoefRandomForest(x, y, n_core = 4):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV
    #from treeinterpreter import treeinterpreter as ti


    X_train, X_test , y_train, y_test = \
    train_test_split(x, y, test_size=0.2, random_state=1)

    rf = RandomizedSearchCV(RandomForestRegressor(),\
    param_distributions =  {
                  'n_estimators':np.arange(10,500,5)
                  #'max_features':np.arange(1,10,1)
               },
                  cv=5, n_iter = 20,
                  iid=False,random_state=0,refit=True,
                  scoring="neg_mean_absolute_error", n_jobs = n_core)

    rf.fit(X_train,y_train)
    rf = rf.best_estimator_
    prediction, bias, contributions = rf.predict(rf, X_test)

    totalc1 = np.mean(contributions, axis=0)
    coef = pd.Series(totalc1, index=x.columns)

    return coef


def CoefElasticNetCV(X, Y, k=5, n_core = 4):
    from sklearn.linear_model import ElasticNetCV,ElasticNet
    from sklearn.model_selection import train_test_split

    alphas = [0.001, 0.02, 0.01, 0.1, 0.5, 1, 5]

    skf = KFold(n_splits=k, shuffle=True)
    indexes = [ (training, test) for training, test in skf.split(X, Y) ]
        # iterate over all folds
    dfTopCoefTemp = pd.DataFrame(dtype='float64', index=X.columns).fillna(0)        
    for train_index, test_index in indexes:
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = Y[train_index], Y[test_index]
        elasticcv = ElasticNetCV(alphas=alphas, cv = 5, max_iter=1000, normalize=True)
        elasticcv.fit(X_train, y_train)
        elastic = ElasticNet(alpha=elasticcv.alpha_, max_iter=1e4, normalize=True).fit(X_train, y_train)
        dfTopCoefTemp = pd.concat([dfTopCoefTemp, pd.Series(elastic.coef_, index = X.columns).fillna(0)], axis = 1)

    return dfTopCoefTemp.apply(lambda row: row.mean(), axis=1) 


def CoefRidgeCV(X,Y,k=5):
    from sklearn.linear_model import Ridge, RidgeCV

    alphas = np.logspace(-10, -2, 10)
    skf = KFold(n_splits=k, shuffle=True)
    indexes = [ (training, test) for training, test in skf.split(X, Y) ]
        # iterate over all folds
    dfTopCoefTemp = pd.DataFrame(dtype='float64', index=X.columns).fillna(0)        
    for train_index, test_index in indexes:
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = Y[train_index], Y[test_index]
        ridgecv = RidgeCV(alphas=alphas, cv = 5, normalize=True)
        ridgecv.fit(X_train, y_train)
        ridge = Ridge(alpha=ridgecv.alpha_, max_iter=1e4, normalize=True).fit(X_train, y_train)
        dfTopCoefTemp = pd.concat([dfTopCoefTemp, pd.Series(ridge.coef_, index = X.columns).fillna(0)], axis =1)

    return dfTopCoefTemp.apply(lambda row: row.mean(), axis=1) 


def CoefRidge(x, y):
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.model_selection import train_test_split

    X_train, X_test , y_train, y_test = \
    train_test_split(x, y, test_size=0.2, random_state=1)

    ridge6 = Ridge(alpha = 0.01, normalize=True)
    ridge6.fit(X_train, y_train)

    coef = pd.Series(ridge6.coef_, index=x.columns) 

    return coef

##################
## Correlation ###
##################
def pearson(exprDF, lMirUser = None, lGeneUser = None, n_core = 2):
    """
    Function to calculate the Pearson correlation coefficient, and pval
    of each pair of miRNA-mRNA, return a matrix of correlation coefficients 
    with columns are miRNAs and rows are mRNAs.

    Args:   
        exprDF  df Concat Dataframe rows are samples and cols are gene/mirs

    Returns:
        Cordf   df  A  matrix that includes the Pearson correlation 
                        coefficients. Columns are genes, rows are miRNA.
        Pvaldf   df  A  matrix that includes the Pearson correlation 
                        pvalues. Columns are genes, rows are miRNA.
    """
    pandarallel.initialize(verbose=1, nb_workers=n_core)

    lMir, lGene = header_list(exprDF=exprDF)

    if lGeneUser is not None:
        lGene = intersection(lGene,lGeneUser)

    if lMirUser is not None:
        lMir = intersection(lMir,lMirUser)


    Cordf = exprDF[lGene].parallel_apply(lambda gene: exprDF[lMir].corrwith(gene, \
            method = lambda x, y: scipy.stats.pearsonr(x,y)[0]))

    Pvaldf = exprDF[lGene].parallel_apply(lambda gene: exprDF[lMir].corrwith(gene, \
            method = lambda x, y: scipy.stats.pearsonr(x,y)[1]))

    Cordf = Cordf.apply(lambda col: col.dropna())
    Pvaldf = Pvaldf.apply(lambda col: col.dropna())
    
    return Cordf, Pvaldf


def spearman(exprDF, lMirUser = None, lGeneUser = None, n_core = 2):
    """
    Function to calculate the Spearman correlation coefficient, and pval
    of each pair of miRNA-mRNA, return a matrix of correlation coefficients 
    with columns are miRNAs and rows are mRNAs.

    Args:   
        exprDF  df Concat Dataframe rows are samples and cols are gene/mirs

    Returns:
        Cordf   df  A  matrix that includes the Spearman correlation 
                        coefficients. Columns are genes, rows are miRNA.
        Pvaldf   df  A  matrix that includes the Spearman correlation 
                        pvalues. Columns are genes, rows are miRNA.
    """
    pandarallel.initialize(verbose=1, nb_workers=n_core)

    lMir, lGene = header_list(exprDF=exprDF)

    if lGeneUser is not None:
        lGene = intersection(lGene,lGeneUser)

    if lMirUser is not None:
        lMir = intersection(lMir,lMirUser)


    Cordf = exprDF[lGene].parallel_apply(lambda gene: exprDF[lMir].corrwith(gene, \
            method = lambda x, y: scipy.stats.spearmanr(x,y)[0]))

    Pvaldf = exprDF[lGene].parallel_apply(lambda gene: exprDF[lMir].corrwith(gene, \
            method = lambda x, y: scipy.stats.spearmanr(x,y)[1]))

    Cordf = Cordf.apply(lambda col: col.dropna())
    Pvaldf = Pvaldf.apply(lambda col: col.dropna())
    
    return Cordf, Pvaldf 


def kendall(exprDF, lMirUser = None, lGeneUser = None, n_core = 2):
    """
    Function to calculate the Kendall correlation coefficient, and pval
    of each pair of miRNA-mRNA, return a matrix of correlation coefficients 
    with columns are miRNAs and rows are mRNAs.

    Args:   
        exprDF  df Concat Dataframe rows are samples and cols are gene/mirs

    Returns:
        Cordf   df  A  matrix that includes the Kendall correlation 
                        coefficients. Columns are genes, rows are miRNA.
        Pvaldf   df  A  matrix that includes the Kendall correlation 
                        pvalues. Columns are genes, rows are miRNA.
    """

    pandarallel.initialize(verbose=1, nb_workers=n_core)

    lMir, lGene = header_list(exprDF=exprDF)


    if lGeneUser is not None:
        lGene = intersection(lGene,lGeneUser)

    if lMirUser is not None:
        lMir = intersection(lMir,lMirUser)


    Cordf = exprDF[lGene].parallel_apply(lambda gene: exprDF[lMir].corrwith(gene, \
            method = lambda x, y: scipy.stats.kendalltau(x,y)[0]))

    Pvaldf = exprDF[lGene].parallel_apply(lambda gene: exprDF[lMir].corrwith(gene, \
            method = lambda x, y: scipy.stats.kendalltau(x,y)[1]))

    Cordf = Cordf.apply(lambda col: col.dropna())
    Pvaldf = Pvaldf.apply(lambda col: col.dropna())
    
    return Cordf, Pvaldf 


def lasso(exprDF, lMirUser = None, lGeneUser = None, n_core = 2):
    """
    Function to calculate the Lasso correlation coefficient 
    of each pair of miRNA-mRNA, return a matrix of correlation coefficients 
    with columns are miRNAs and rows are mRNAs.

    Args:   
        exprDF  df Concat Dataframe rows are samples and cols are gene/mirs

    Returns:
        Cordf   df  A  matrix that includes the lasso correlation 
                        coefficients. Columns are genes, rows are miRNA.
    """

    pandarallel.initialize(verbose=1, nb_workers=n_core)

    lMir, lGene = header_list(exprDF=exprDF)


    if lGeneUser is not None:
        lGene = intersection(lGene,lGeneUser)

    if lMirUser is not None:
        lMir = intersection(lMir,lMirUser)


    Cordf = exprDF[lGene].parallel_apply(lambda gene: \
            CoefLassoCV(exprDF[lMir], gene))

    Cordf = Cordf.apply(lambda col: col.dropna())

    
    return Cordf


def ridge (exprDF, lMirUser = None, lGeneUser = None, n_core = 2):
    """
    Function to calculate the Ridge correlation coefficient
    of each pair of miRNA-mRNA, return a matrix of correlation coefficients 
    with columns are miRNAs and rows are mRNAs.

    Args:   
        exprDF  df Concat Dataframe rows are samples and cols are gene/mirs

    Returns:
        Cordf   df  A  matrix that includes the Ridge correlation 
                        coefficients. Columns are genes, rows are miRNA.
    """
    pandarallel.initialize(verbose=1, nb_workers=n_core)

    lMir, lGene = header_list(exprDF=exprDF)

    if lGeneUser is not None:
        lGene = intersection(lGene,lGeneUser)

    if lMirUser is not None:
        lMir = intersection(lMir,lMirUser)


    Cordf = exprDF[lGene].parallel_apply(lambda gene: \
            CoefRidgeCV(exprDF[lMir],gene))


    Cordf = Cordf.apply(lambda col: col.dropna())

    
    return Cordf


def elasticnet(exprDF, lMirUser = None, lGeneUser = None, n_core = 2):
    """
    Function to calculate the ElasticNet correlation coefficient 
    of each pair of miRNA-mRNA, return a matrix of correlation coefficients 
    with columns are miRNAs and rows are mRNAs.

    Args:   
        exprDF  df Concat Dataframe rows are samples and cols are gene/mirs

    Returns:
        Cordf   df  A  matrix that includes the lasso correlation 
                        coefficients. Columns are genes, rows are miRNA.
    """

    pandarallel.initialize(verbose=1, nb_workers=n_core)

    lMir, lGene = header_list(exprDF=exprDF)

    if lGeneUser is not None:
        lGene = intersection(lGene,lGeneUser)

    if lMirUser is not None:
        lMir = intersection(lMir,lMirUser)


    Cordf = exprDF[lGene].parallel_apply(lambda gene: \
            CoefElasticNetCV(exprDF[lMir],gene, n_core=n_core))


    Cordf = Cordf.apply(lambda col: col.dropna())

    
    return Cordf


def randomforest(exprDF, lMirUser = None, lGeneUser = None, n_core = 2):
    """
    Function to calculate the RandomForest Regression coefficient 
    of each pair of miRNA-mRNA, return a matrix of correlation coefficients 
    with columns are miRNAs and rows are mRNAs.

    Args:   
        exprDF  df Concat Dataframe rows are samples and cols are gene/mirs

    Returns:
        Cordf   df  A  matrix that includes the lasso correlation 
                        coefficients. Columns are genes, rows are miRNA.
    """

    pandarallel.initialize(verbose=1, nb_workers=n_core)

    lMir, lGene = header_list(exprDF=exprDF)

    if lGeneUser is not None:
        lGene = intersection(lGene,lGeneUser)

    if lMirUser is not None:
        lMir = intersection(lMir,lMirUser)


    Cordf = exprDF[lGene].parallel_apply(lambda gene: \
            CoefRandomForest(exprDF[lMir],gene))


    Cordf = Cordf.apply(lambda col: col.dropna())

    
    return Cordf



def lars(exprDF, lMirUser = None, lGeneUser = None, n_core = 2):
    """
    Function to calculate the RandomForest Regression coefficient 
    of each pair of miRNA-mRNA, return a matrix of correlation coefficients 
    with columns are miRNAs and rows are mRNAs.

    Args:   
        exprDF  df Concat Dataframe rows are samples and cols are gene/mirs

    Returns:
        Cordf   df  A  matrix that includes the lasso correlation 
                        coefficients. Columns are genes, rows are miRNA.
    """

    pandarallel.initialize(verbose=1, nb_workers=n_core)

    lMir, lGene = header_list(exprDF=exprDF)

    if lGeneUser is not None:
        lGene = intersection(lGene,lGeneUser)

    if lMirUser is not None:
        lMir = intersection(lMir,lMirUser)


    Cordf = exprDF[lGene].parallel_apply(lambda gene: \
            CoefLarsCV(exprDF[lMir],gene))


    Cordf = Cordf.apply(lambda col: col.dropna())

    
    return Cordf


def hoeffding (exprDF, lMirUser = None, lGeneUser = None, n_core = 2):
    """
    Function to calculate the hoeffding correlation coefficient
    of each pair of miRNA-mRNA, return a matrix of correlation coefficients 
    with columns are miRNAs and rows are mRNAs.

    Args:   
        exprDF  df Concat Dataframe rows are samples and cols are gene/mirs

    Returns:
        Cordf   df  A  matrix that includes the hoeffding correlation 
                        coefficients. Columns are genes, rows are miRNA.
    
    Ref:
        https://github.com/PaulVanDev/HoeffdingD    
    """

    from  .metrics import hoeffding

    pandarallel.initialize(verbose=1, nb_workers=n_core)

    lMir, lGene = header_list(exprDF=exprDF)


    if lGeneUser is not None:
        lGene = intersection(lGene,lGeneUser)

    if lMirUser is not None:
        lMir = intersection(lMir,lMirUser)


    Cordf = exprDF[lGene].parallel_apply(lambda gene: exprDF[lMir].corrwith(gene, \
            method = lambda x, y: hoeffding(x,y)))

    Cordf = Cordf.apply(lambda col: col.dropna())
    
    return Cordf 




def rdc (exprDF, lMirUser = None, lGeneUser = None, n_core = 2):
    """
    Function to calculate the Randomized Dependence Coefficient coefficient
    of each pair of miRNA-mRNA, return a matrix of correlation coefficients 
    with columns are miRNAs and rows are mRNAs.

    Args:   
        exprDF  df Concat Dataframe rows are samples and cols are gene/mirs

    Returns:
        Cordf   df  A  matrix that includes the rdc correlation 
                        coefficients. Columns are genes, rows are miRNA.
    
    Ref:
        https://github.com/garydoranjr/rdc    
    """

    from .metrics import rdc

    pandarallel.initialize(verbose=1, nb_workers=n_core)

    lMir, lGene = header_list(exprDF=exprDF)


    if lGeneUser is not None:
        lGene = intersection(lGene,lGeneUser)

    if lMirUser is not None:
        lMir = intersection(lMir,lMirUser)


    Cordf = exprDF[lGene].parallel_apply(lambda gene: exprDF[lMir].corrwith(gene, \
            method = lambda x, y: rdc(x,y)))

    Cordf = Cordf.apply(lambda col: col.dropna())
    
    return Cordf


def hazard_ratio_mirgen(exprDF, table, lMirUser = None, lGeneUser = None, n_core = 2):
    """
    Function to calculate the Spearman correlation coefficient, and pval
    of each pair of miRNA-mRNA, return a matrix of correlation coefficients 
    with columns are miRNAs and rows are mRNAs.

    Args:   
        exprDF  df Concat Dataframe rows are samples and cols are gene/mirs

    Returns:
        Cordf   df  A  matrix that includes the Spearman correlation 
                        coefficients. Columns are genes, rows are miRNA.
        Pvaldf   df  A  matrix that includes the Spearman correlation 
                        pvalues. Columns are genes, rows are miRNA.
    """
    from .survival import hazard_ratio

    pandarallel.initialize(verbose=1, nb_workers=n_core)

    lMir, lGene = header_list(exprDF=exprDF)

    if lGeneUser is not None:
        lGene = intersection(lGene,lGeneUser)

    if lMirUser is not None:
        lMir = intersection(lMir,lMirUser)


    hr = hazard_ratio(exprDF=exprDF, lMirUser=lMir, lGeneUser=lGene, n_core = n_core)
    hr.index = hr.target

    table["GENE_HR"] = table.apply(lambda x: hr.loc[x["Gene"],"log(hr)"], axis = 1)
    table["MIR_HR"] = table.apply(lambda x: hr.loc[x["Mir"],"log(hr)"], axis = 1)
    
    return table 


def all_methods(exprDF, lMirUser = None, lGeneUser = None, n_core = 2, hr = False, k = 10):
    """
    Function to calculate all coefficient
    of each pair of miRNA-mRNA, return a matrix of correlation coefficients 
    with columns are miRNAs and rows are mRNAs.

    Args:   
        exprDF  df Concat Dataframe rows are samples and cols are gene/mirs

    Returns:
  
    """
    import copy


    lMir, lGene = header_list(exprDF=exprDF)
   
    if lGeneUser is not None:
        lGene = intersection(lGene,lGeneUser)

    if lMirUser is not None:
        lMir = intersection(lMir,lMirUser)

    print("Obtain Concat Gene and MIR")

    
    modelList = [[spearman,"Rho"],
                 [pearson,"R"],
                 [kendall,"Tau"],
                 [rdc, "RDC"],
                 [hoeffding,"Hoeffding"],
                 [ridge,"Ridge"],
                 [lasso,"Lasso"],
                 [elasticnet,"ElasticNet"],
                # [hazard_ratio_mirgen, "Log(HR)" ]
                 ]
    

    print("Loading dataset...")
    
    
    lTuple = []
    for model, name in modelList :
        print("\nClassifier " + name)
        classifier = copy.deepcopy(model)
        try:
            if name in ["Rho", "R","Tau"]:
                dfCor, pval = classifier(exprDF, lGeneUser=lGene, lMirUser=lMir, n_core=n_core)
                lTuple.append((pval,"%s_Pval"%name))

            else:
                dfCor = classifier(exprDF, lGeneUser=lGene, lMirUser=lMir,n_core=n_core)
        except Exception as error:
            print(error)
            pass
        else:
            #dfCor.to_csv("~/%s.csv"%name)
            lTuple.append((dfCor,name))

    table = process_matrix_list(lTuple, add_target=True)

    table = table.loc[~table.duplicated(), :]

    table = adjPval(table)

    if hr:
        table = hazard_ratio_mirgen(exprDF, table, lGeneUser=lGene, lMirUser=lMir, n_core=n_core)
    return table, lTuple[1][0]



#################
### Reshaping ###
#################

def matrix2table(Cordf, value_name = "Value"):
    """
    Function to reshaping a correlation matrix where Columns are genes, 
    rows are miRNA to Table.


    Args:   
        Cordf   df  A  matrix that includes the correlation 
                        coefficients. Columns are genes, rows are miRNA.

    Returns:
        Table   df  A  matrix that includes 3 col: gene, mir and value
    
    Ref:
        https://pandas.pydata.org/docs/user_guide/reshaping.html    
    """

    table = Cordf.melt(ignore_index=False)
    #table["Mir"] = table.index.tolist()
    #table = table.loc[["Mir", "Gene", "Value"],:]
    table = table.reset_index()
    table.columns = ["Mir","Gene", value_name]
    
    return table


def merge_tables(lDf, method = "inner"):
    """
    Function to concat the correlations Tables.

    Args:   
        lDf   list  List of DFs with the correlation data

    Returns:
        df   df  A concat df
    
    Ref:
        https://stackoverflow.com/questions/23668427/pandas-three-way-joining-multiple-dataframes-on-columns    
    """
    from functools import reduce

    df = reduce(lambda left, right: pd.merge(left,right,on=['Gene','Mir'], how = method), lDf)

    return df 


def process_matrix_list(lTuple, add_target=True, method = "outer"):
    """
    Function to reshape a list of correlation matrix
    in the correlations Tables.

    Args:   
        lTuple   list  List of tuples with the correlation matrix, an the
                        name of the analysis (df,"value_name")

    Returns:
        table   df  A table with al the values merged
    """
    lDf = []
    if add_target:
        df = lTuple[0][0]
        target = load_matrix_counts()

        lGen = intersection(df.columns.tolist(), target.index.tolist() ) 
        lMir = intersection(df.index.tolist(), target.columns.tolist() )

        target = target.loc[lGen,lMir].transpose()
        lTuple.append((target, "Prediction Tools"))

    for tuple in lTuple:
        df = matrix2table(tuple[0], tuple[1])
        lDf.append(df)

    table = merge_tables(lDf, method = method)

    return table    


def filter_table(table, low_coef = -0.2, high_coef = 0.2, nDB = 3, pval = 0.05):
    query_string = f"""
                ((R <= {low_coef} | R >= {high_coef} ) | \
                 (Rho <= {low_coef} | Rho >= {high_coef} ) | \
                 (Tau <= {low_coef} | Tau >= {high_coef} ))  \
                & (Tool Number >= {nDB}) & \
                ((Spear_fdr_bh <= {pval}) | \
                 (Pear_fdr_bh <= {pval}) | \
                 (Kendall_fdr_bh <= {pval}))
                 """
    table = table.query(query_string)
    return table

def _mir_gene_ratio(mir, gene, df):
        name = "%s/%s"%(mir.name,gene.name)
        df[name] = pd.Series(mir/gene)
        return None

def get_mir_gene_ratio(exprDF, lMirUser = None, lGeneUser = None,  filter_pair = False, low_coef = -0.5, min_db = 20):
    """
    Function to calculate all coefficient
    of each pair of miRNA-mRNA, return a matrix of correlation coefficients 
    with columns are miRNAs and rows are mRNAs.

    Args:   
        exprDF  df Concat Dataframe rows are samples and cols are gene/mirs

    Returns:
  
    """

    lMir, lGene = header_list(exprDF=exprDF)
   
    if lGeneUser is not None:
        lGene = intersection(lGene,lGeneUser)

    if lMirUser is not None:
        lMir = intersection(lMir,lMirUser)
        
    df = pd.DataFrame()

    if filter_pair:
        dfCor, pval = pearson(exprDF, lGeneUser=lGene, lMirUser=lMir, n_core=6)
        table = process_matrix_list([(dfCor,"R")], add_target=True)
        table = table.loc[table["R"] <= low_coef, :]
        table = table.loc[table["Prediction Tools"].str.count("1") > min_db,:]

    exprDF[lGene].apply(lambda gene: 
                            exprDF[lMir].apply(lambda mir:
                                 _mir_gene_ratio(mir, gene, df)))

    lKeep = table.apply(lambda row: "%s/%s"%(row["Mir"],row["Gene"]), axis = 1)
    df = df[lKeep]
    return df

#################
### Sorting   ###
#################

def obtain_top_matrix(Cordf, topk=100, value_name = "Value"):
    """
    Function to obtain the top X miRNAs-mRNA interactions
    from corr matrix.

    Args:

    Returns:

    """
    
    table = matrix2table(Cordf, value_name) 
    top = obtain_top_table(table, topk, value_name)

    return top


def obtain_top_table(table, topk=100, value_name=None):
    """
    Function to obtain the top X miRNAs-mRNA interactions
    from corr matrix.

    Args:

    Returns:

    """
    table = table.dropna()

    if value_name is None:
        table = table.reindex(table[table.columns.tolist()[2]].abs().sort_values(ascending=False).index)

    else:
        
        table = table.reindex(table[value_name].abs().sort_values(ascending=False).index)


    toprank = table.head(topk)

    return toprank

   
def borda_table(table, lMethod = None):
    """
    Function to use the Borda count election
    to integrate the rankings from different miRNA
    coefficients.

    Args:   
        table   df  A table with al the values merged
        lMethod     list    List of the columns to rank

    Returns:
        TableRank   df  A table with al the values merged and ranking
    """


    TableRank = table
    
    if lMethod == None:
        lMethod = GetMethodList(table.columns.tolist())

    TableRank.loc[:,"Ranking"] = rk.borda(TableRank[lMethod].abs(), reverse=False)
    TableRank.loc[:,"Ranking"] = TableRank["Ranking"].round(0)
    return TableRank.sort_values("Ranking", ignore_index = True).round(3)


def borda_matrix(lTuple):
    """
    Function to use the Borda count election
    to integrate the rankings from different miRNA
    coefficients.

    Args:   
        lTuple   list  List of tuples with the correlation matrix, an the
                        name of the analysis (df,"value_name")

    Returns:
        TableRank   df  A table with al the values merged and ranking
    """

    lDf = []
    

    for tuple in lTuple:
        df = matrix2table(tuple[0], tuple[1])
        lDf.append(df)

    TableRank = merge_tables(lDf)

    TableRank = borda_table(TableRank)

    return TableRank


def borda_top_matrix(lTuple, topk=100):
    """
    Function to obtain the the consensus ranking from the TopK
    pair miRNA-mRNA for each correlation method. We only keep
    the pairs present in all the top

    Args:   
        table   df  A table with al the values merged
        lMethod     list    List of the columns to rank

    Returns:
        TableRank   df  A table with al the values merged and ranking

    """

    lDf = []
    

    for tuple in lTuple:
        df = obtain_top_matrix(tuple[0], value_name = tuple[1], topk = topk)
        lDf.append(df)

    TableRank = merge_tables(lDf, method = "outer")

    TableRank = borda_table(TableRank)

    return TableRank


def borda_top_table(table, topk=100, only_negative = False, method = "outer"):
    """
    Function to obtain the the consensus ranking from the TopK
    pair miRNA-mRNA for each correlation method. We only keep
    the pairs present in all the top

    Args:   
        lTuple   list  List of tuples with the correlation matrix, an the
                        name of the analysis (df,"value_name")

    Returns:
        TableRank   df  A table with al the values merged and ranking

    """
    lMethod = GetMethodList(table.columns.tolist())

    lDf = []

    if only_negative:
        for m in lMethod:
            if m != "RDC" and m != "Hoeffding":
                table = table[table[m] <= 0]

        #Table = Table[Table["DB"]!="00000000000000000000000"]

    for m in lMethod:
        df = obtain_top_table(table[["Mir","Gene",m]], topk=topk, value_name=m)
        lDf.append(df)

    TableRank = merge_tables(lDf,  method = method)

    TableRank = borda_table(TableRank)
    TableRank = merge_tables([TableRank, table[["Mir","Gene","Prediction Tools"]]], method="inner")
 
    return TableRank.sort_values("Ranking", ignore_index = True)


def opposite_correlation(Table1, Table2, method="R"):

    keepCol = ["Gene","Mir",method]
    mergeTable = merge_tables([Table1[keepCol],Table2[keepCol]])
    mergeTable["Anti"] = mergeTable["%s_x"%method] * mergeTable["%s_y"%method]

    mergeTable = mergeTable[(mergeTable["Anti"] < 0)].sort_values("Anti",ignore_index=True)

    return mergeTable


def predict_lethality(gene, table, topk=100, method = "outer"):
    ##Get Methods##
    lMethod = GetMethodList(table.columns.tolist())
    slDF = load_synthetic()
    
    gene_list = slDF[(slDF["GeneA"]== gene)]["GeneB"].tolist()

    match = table[table["Gene"].str.contains("|".join(gene_list))]

    topmatch = borda_top_table(match, topk=topk, method = method )

    #topmatch = merge_tables([topmatch, table[["Mir","Gene","DB"]]], method="inner")

    return topmatch

def FilterDF(table = None, matrix = None, join = "or", lTool = [], low_coef = -0.5, high_coef = 0.5, pval = 0.05, analysis = "Correlation", min_db = 1, method = "R"):


    dbQ = get_target_query(join, lTool)
    #print(dbQ)

    #Filter DF
    if analysis == "Correlation":
        if method in ["R","Rho","Tau"]:
            query_string = f"""
                    (({method} <= {low_coef} | {method} >= {high_coef} )) \
                    & \
                    (({method}_fdr_bh <= {pval}))
            """
        else:
            query_string = f"""
                    (({method} <= {low_coef} | {method} >= {high_coef} ))
                    """
                    
        table = table[table["Prediction Tools"].str.match(dbQ)==True]
        table["Number Prediction Tools"] = table["Prediction Tools"].str.count("1")
        table = table[table["Number Prediction Tools"] >= min_db]

    elif analysis == "GeneSetScore":
        query_string = f"""
                ((R <= {low_coef} | R >= {high_coef} ) \
                &  (FDR <= {pval}))
        """
    table = table.query(query_string)#Query the Correlation Table
    table = borda_table(table)
    #print("Filtrado")
    #print(table.head())
            

    gene = table["Gene"].unique().tolist()#Obtain Unique Gene after filter the table
    mir = table["Mir"].unique().tolist()#Obtain Unique mir after filter the table

    if matrix is not None:
        try:
            matrix = matrix.loc[mir,gene]#Subset the Correlation matrix to the heatmap
        except:
            matrix = matrix.loc[gene,mir]#Subset the Correlation matrix to the heatmap

        return table, matrix
    else:
        return table


def predict_target(table = None, matrix = None, lTarget = None, lTools = None, method = "or", min_db = 10, low_coef = -0.5, high_coef = 0.5, pval = 0.05):

    lTools = lTools if lTools != None else load_matrix_header()

    if len(lTarget) > 0: 
        if table is not None:
            #print("Holi")
            if matrix is not None:
                table, matrix = FilterDF(table = table, matrix = matrix, join = method, lTool = lTools, \
                    low_coef = low_coef, high_coef = high_coef, pval = pval)
            else:
                table = FilterDF(table = table, matrix = matrix, join = method, lTool = lTools, \
                    low_coef = low_coef, high_coef = high_coef, pval = pval)
        else:
            table = load_table_counts()
            dbQ = get_target_query(method, lTools)
            table = table[table["Prediction Tools"].str.match(dbQ)==True]
                #Read DF
            
        #Obtain Target Table
        target = table[table["Gene"].isin(lTarget)|table["Mir"].isin(lTarget)]
        del table
        #Filter by number
        target["Number Prediction Tools"] = target["Prediction Tools"].str.count("1")
        target = target[target["Number Prediction Tools"] >= min_db]

        if not target.empty and matrix is not None:
            gene = target["Gene"].unique().tolist()#Obtain Unique Gene after filter the table
            mir = target["Mir"].unique().tolist()#Obtain Unique mir after filter the table

            try:
                matrix = matrix.loc[mir,gene]#Subset the Correlation matrix to the heatmap
            except:
                matrix = matrix.loc[gene,mir]#Subset the Correlation matrix to the heatmap
        else:
            matrix = None

    else:
        target, matrix = None, None
    return target, matrix


def ora_mir(lGene, matrix, mir_name, q):
    from scipy.stats import fisher_exact

    total_number_gene_universe = len(set(matrix.index.tolist()))
    total_number_gene_list = len(set(lGene))
    target_gene_list = matrix.loc[matrix[mir_name].str.count("1") >= q, mir_name].index.tolist()
    target_number_universe = len(set(target_gene_list))
    target_number_list = len(set(lGene).intersection(set(target_gene_list)))
    
    in_list, not_list = target_number_list, total_number_gene_list - target_number_list
    in_universe, not_universe = target_number_universe, total_number_gene_universe - target_number_universe
    
    data = {"List":[in_list, not_list], "Universe": [in_universe, not_universe]}
    res = pd.DataFrame.from_dict(data)
    
    odd, pval = fisher_exact(res)
    
    expected = (in_universe / total_number_gene_universe) * total_number_gene_list
    
    return pd.Series([mir_name, target_number_list, expected, odd, pval], \
                     index = ["microRNA","Target Number","Expected Number", "Fold Enrichment", "Raw P-value"], name = mir_name)



def ora_mir_list(lMir, lGene, matrix, minDB):
    df = pd.DataFrame()
    for mir in lMir:
        res = ora_mir(lGene, matrix, mir, minDB)
        df = pd.concat([df,res], axis = 1)
    return df


def ora_mir_parallel(lGene, matrix, lMir, minDB, n_core = 2):

    ##Split List
    np_list_split = np.array_split(lMir, n_core)
    split_list = [i.tolist() for i in np_list_split]
    #split_list = same_length(split_list)

    #Fix Exprs Variable 
    partial_func = functools.partial(ora_mir_list, matrix = matrix, lGene = lGene,  minDB=minDB)

    #Generating Pool
    pool = Pool(n_core)
    lres = pool.map(partial_func, split_list)
    res = pd.concat(lres, axis = 1)
    res = res.transpose()
    pool.close() 
    pool.join()
    
    res["FDR"] = multipletests(res["Raw P-value"], method = "fdr_bh")[1]
    return res


def predict_lethality2(table = None, matrix = None, lQuery = None, lTools = None, method = "or", min_db = 10, low_coef = -0.5, high_coef = 0.5, pval = 0.05):
    ##Get Methods##
    import pandas as pd

    slDF = load_synthetic()
    
    qA = slDF.loc[slDF["GeneA"].isin(lQuery),:]
    qA.columns = ["Query", "Synthetic Lethal"]
    qB = slDF.loc[slDF["GeneB"].isin(lQuery),:]
    qB.columns = ["Synthetic Lethal","Query"]

    qSl = pd.concat([qA,qB])
    lTarget = qSl["Synthetic Lethal"].tolist()

    if len(lTarget) > 0:
        target,matrix = predict_target(table = table, matrix = matrix, lTarget = lTarget, lTools = lTools, method = method, min_db = min_db, low_coef = low_coef, high_coef = high_coef, pval = pval)
        target = pd.merge(qSl,target, left_on="Synthetic Lethal", right_on="Gene")

        res = ora_mir_parallel(lTarget, load_matrix_counts(), target["Mir"].unique().tolist(), min_db)
    else:
        target = None
    
    return target, matrix, res
    

