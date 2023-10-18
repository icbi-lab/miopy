################
#### MIOPY #####
################

import pandas as pd
import numpy as np
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LarsCV, Lasso, LassoCV, ElasticNetCV, ElasticNet, Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from statsmodels.stats.multitest import multipletests

from os import path
from .R_utils import tmm_normalization, deg_edger, deg_limma_array, voom_normalization
from .process_data import *
from .immune import ips
from .data import load_gene_ips
from multiprocessing import Pool
import functools

import scipy.stats
import ranky as rk
from .process_data import *

##################
## Miscelaneos ###
##################


def adjust_geneset(table):
    """
    Function to adjust the p-values of the geneset
    Args:
        table   df  Dataframe long table with the pvalues from the geneset
    """

    p = table["Pval"]
    mask = np.isfinite(p)

    #Creating an empty vector
    table["FDR"] = 1
    table.loc[mask,"FDR"] = multipletests(p[mask], method="fdr_bh")[1]
    return table
    

def adj_pval(df):
    """
    Function to adjust the p-values of the correlation results
    Args:
        df  df  Dataframe with the correlation results
    Returns:
        df  df  Dataframe with the correlation results and the adjusted pvalues
    """

    l_cor = ["Rho","R","Tau","Background"]
    method = "fdr_bh"

    for cor in l_cor:
        col = "%s_%s"  %(cor,method)
        col_raw = "%s_Pval" %(cor)

        #Pval
        try: 
            p = df[col_raw]
            mask = np.isfinite(p)

            #Creating an empty vector
            df[col] = 1

            df.loc[mask,col] = multipletests(p[mask], method=method)[1]
        except Exception as e:
            print(e)

    return df


##################
## GeneSetScore ##
##################
def calculate_gene_set_score(expr, conf):
    """
    Function to calculate the GeneSetScore
    Args:
        expr    df  Dataframe with the expression of the genes
        conf    df  Dataframe with the targeting of the genes
    Returns:
        GScore  float   GeneSetScore
    """

    sum_gene_predictor = sum(expr * conf)
    sum_predictor = sum(conf)
    
    try:
        GScore = sum_gene_predictor/sum_predictor
    except:
        GScore = 0
    #print(GScore)
    return GScore

def gene_set_correlation(exprDf, lGeneSet, GeneSetName="GeneSet", lMirUser=None, n_core=6):
    """
    Calculate the correlation between a GeneSet and the expression of microRNA, considering microRNA/gene targeting.

    Args:
        exprDf (DataFrame): Dataframe with gene expression data.
        lGeneSet (list): List of genes in the GeneSet.
        GeneSetName (str): Name of the GeneSet.
        lMirUser (list): List of microRNA to use.
        n_core (int): Number of CPU cores to use for parallel processing.

    Returns:
        dfCor (DataFrame): Dataframe with correlation values between the GeneSet and microRNA.
        dfPval (DataFrame): Dataframe with p-values of the correlation between the GeneSet and microRNA.
        dfSetScore (DataFrame): Dataframe with the GeneSetScore of the GeneSet.
    """

    # Initialize parallel processing with the specified number of cores
    pandarallel.initialize(verbose=1, nb_workers=n_core)

    # Extract the list of microRNA and gene names from the expression dataframe
    lMir, lGene = header_list(exprDF=exprDf)

    # Load a confident dataframe and count "1" occurrences in the expression matrix
    dfConf = get_confident_df(load_matrix_counts().apply(lambda col: col.str.count("1")))

    # Intersect gene and microRNA lists with the confident dataframe indices and columns
    lGene = intersection(lGene, dfConf.index.tolist())
    lMir = intersection(lMir, dfConf.columns.tolist())

    # Intersect the gene list with the provided GeneSet if it is given
    if lGeneSet is not None:
        lGene = intersection(lGene, lGeneSet)

    # Intersect the microRNA list with the provided list if it is given
    if lMirUser is not None:
        lMir = intersection(lMir, lMirUser)

    # Calculate the GeneSetScore for each microRNA and gene pair using parallel processing
    dfSetScore = dfConf.loc[lGene, lMir].parallel_apply(lambda conf:
                                                        exprDf[lGene].apply(lambda expr:
                                                                           calculate_gene_set_score(expr, conf),
                                                                           axis=1), axis=0)

    # Remove columns with NaN values from the GeneSetScore dataframe
    dfSetScore = dfSetScore.apply(lambda col: col.dropna())

    # Calculate the correlation between GeneSetScore and gene expression for each microRNA
    cor = dfSetScore.parallel_apply(lambda col:
                                   col.corr(exprDf[col.name], method=lambda x, y: scipy.stats.pearsonr(x, y)))

    # Remove columns with NaN values from the correlation dataframe
    cor = cor.apply(lambda col: col.dropna())

    # Create dataframes for correlation and p-values
    dfPval = pd.DataFrame(cor.loc[:, 1])
    dfCor = pd.DataFrame(cor.loc[:, 0])

    # Rename the columns with the GeneSetName
    dfPval.columns = [GeneSetName]
    dfCor.columns = [GeneSetName]

    return dfCor, dfPval, dfSetScore

###########################
##### Immunephenoscore ####
###########################


def ips_correlation(exprDf, lMirUser = None, n_core = 2):

    pandarallel.initialize(verbose=1, nb_workers=n_core)

    lMir, lGene = header_list(exprDF=exprDf)


    if lMirUser is not None:
        lMir = intersection(lMir,lMirUser)
    
    gene_ips = load_gene_ips()

    dfIPS = exprDf[lGene].parallel_apply(lambda x: \
                      ips(x, gene_ips), \
                      axis = 1)
                      
    dfExpr = pd.concat((dfIPS,exprDf[lMir]), axis = 1)

    table, dfCor = all_methods(dfExpr, lMirUser = lMir, lGeneUser = ["AZ",], n_core = n_core, hr = False, k = 5, background = False, test = False, add_target = False)
    
    return table, dfCor, dfIPS



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


def CoefLarsCV(x, y, n_core=4):
    """
    Compute the coefficients using the Least Angle Regression (LARS) Cross-Validation.

    Args:
        x (DataFrame): Features data.
        y (Series): Target variable.
        n_core (int): Number of CPU cores to use for parallel processing.

    Returns:
        coef (Series): Coefficients obtained from LARS CV.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # Cross-validation with LARS
    larscv = LarsCV(cv=5)
    larscv.fit(X_train, y_train)

    # Get coefficients as a Series
    coef = pd.Series(larscv.coef_, index=x.columns)

    return coef

def CoefLassoCV(X, Y, k=3, n_core=4):
    """
    Compute the coefficients using Lasso with Cross-Validation.

    Args:
        X (DataFrame): Features data.
        Y (Series): Target variable.
        k (int): Number of folds for cross-validation.
        n_core (int): Number of CPU cores to use for parallel processing.

    Returns:
        coef (Series): Mean coefficients obtained from Lasso CV.
    """

    # Split data into k-folds for cross-validation
    skf = KFold(n_splits=k, shuffle=True)
    indexes = [(training, test) for training, test in skf.split(X, Y)]

    dfTopCoefTemp = pd.DataFrame(dtype='float64', index=X.columns).fillna(0)

    for train_index, test_index in indexes:
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = Y[train_index], Y[test_index]

        # Cross-validation with Lasso
        lassocv = LassoCV(cv=5, max_iter=1000)
        lassocv.fit(X_train, y_train)
        lasso = Lasso(max_iter=1e4, alpha=lassocv.alpha_).fit(X_train, y_train)

        dfTopCoefTemp = pd.concat([dfTopCoefTemp, pd.Series(lasso.coef_, index=X.columns).fillna(0)], axis=1)

    # Compute the mean coefficients
    coef = dfTopCoefTemp.apply(lambda row: row.mean(), axis=1)

    return coef

def CoefLasso(x, y):
    """
    Compute the coefficients using Lasso.

    Args:
        x (DataFrame): Features data.
        y (Series): Target variable.

    Returns:
        coef (Series): Coefficients obtained from Lasso.
    """

    alphas = [0.001, 0.02, 0.01, 0.1, 0.5, 1, 5]

    # Lasso regression with a specific alpha
    lasso = Lasso(alpha=1, max_iter=1e4).fit(x, y)

    # Get coefficients as a Series
    coef = pd.Series(lasso.coef_, index=x.columns)

    return coef

def CoefRandomForest(x, y, n_core=4):
    """
    Compute feature importances using Random Forest Regressor.

    Args:
        x (DataFrame): Features data.
        y (Series): Target variable.
        n_core (int): Number of CPU cores to use for parallel processing.

    Returns:
        coef (Series): Feature importances obtained from Random Forest.
    """
 
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # Hyperparameter tuning with RandomizedSearchCV
    rf = RandomizedSearchCV(RandomForestRegressor(),
                           param_distributions={
                               'n_estimators': np.arange(10, 500, 5)
                           },
                           cv=5, n_iter=20,
                           iid=False, random_state=0, refit=True,
                           scoring="neg_mean_absolute_error", n_jobs=n_core)
    rf.fit(X_train, y_train)
    rf = rf.best_estimator_

    # Calculate feature importances
    importances = rf.feature_importances_

    # Create a Series of feature importances
    coef = pd.Series(importances, index=x.columns)

    return coef

def CoefElasticNetCV(X, Y, k=3, n_core=4):
    """
    Compute the coefficients using Elastic Net with Cross-Validation.

    Args:
        X (DataFrame): Features data.
        Y (Series): Target variable.
        k (int): Number of folds for cross-validation.
        n_core (int): Number of CPU cores to use for parallel processing.

    Returns:
        coef (Series): Mean coefficients obtained from Elastic Net CV.
    """

    alphas = [0.001, 0.02, 0.01, 0.1, 0.5, 1, 5]

    skf = KFold(n_splits=k, shuffle=True)
    indexes = [(training, test) for training, test in skf.split(X, Y)]

    dfTopCoefTemp = pd.DataFrame(dtype='float64', index=X.columns).fillna(0)

    for train_index, test_index in indexes:
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = Y[train_index], Y[test_index]

        # Cross-validation with Elastic Net
        elasticcv = ElasticNetCV(alphas=alphas, cv=5, max_iter=1000)
        elasticcv.fit(X_train, y_train)
        elastic = ElasticNet(alpha=elasticcv.alpha_, max_iter=1e4).fit(X_train, y_train)

        dfTopCoefTemp = pd.concat([dfTopCoefTemp, pd.Series(elastic.coef_, index=X.columns).fillna(0)], axis=1)

    # Compute the mean coefficients
    coef = dfTopCoefTemp.apply(lambda row: row.mean(), axis=1)

    return coef

def CoefRidgeCV(X, Y, k=3):
    """
    Compute the coefficients using Ridge with Cross-Validation.

    Args:
        X (DataFrame): Features data.
        Y (Series): Target variable.
        k (int): Number of folds for cross-validation.

    Returns:
        coef (Series): Mean coefficients obtained from Ridge CV.
    """

    alphas = np.logspace(-10, -2, 10)

    skf = KFold(n_splits=k, shuffle=True)
    indexes = [(training, test) for training, test in skf.split(X, Y)]

    dfTopCoefTemp = pd.DataFrame(dtype='float64', index=X.columns).fillna(0)

    for train_index, test_index in indexes:
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = Y[train_index], Y[test_index]

        # Cross-validation with Ridge
        ridgecv = RidgeCV(alphas=alphas, cv=5)
        ridgecv.fit(X_train, y_train)
        ridge = Ridge(alpha=ridgecv.alpha_, max_iter=1e4).fit(X_train, y_train)

        dfTopCoefTemp = pd.concat([dfTopCoefTemp, pd.Series(ridge.coef_, index=X.columns).fillna(0)], axis=1)

    # Compute the mean coefficients
    coef = dfTopCoefTemp.apply(lambda row: row.mean(), axis=1)

    return coef

def CoefRidge(x, y):
    """
    Compute the coefficients using Ridge.

    Args:
        x (DataFrame): Features data.
        y (Series): Target variable.

    Returns:
        coef (Series): Coefficients obtained from Ridge.
    """

    alphas = [0.001, 0.02, 0.01, 0.1, 0.5, 1, 5]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # Ridge regression with a specific alpha
    ridge6 = Ridge(alpha=0.01)
    ridge6.fit(X_train, y_train)

    # Get coefficients as a Series
    coef = pd.Series(ridge6.coef_, index=x.columns)

    return coef

##################
## Correlation ###
##################

def pearson(exprDF, lMirUser = None, lGeneUser = None, n_core = 2, pval = True):
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
    if pval:
        Pvaldf = exprDF[lGene].parallel_apply(lambda gene: exprDF[lMir].corrwith(gene, \
                method = lambda x, y: scipy.stats.pearsonr(x,y)[1]))
    else:
        Pvaldf = pd.DataFrame()

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

    table["HR_GENE"] = table.apply(lambda x: hr.loc[x["Gene"],"log(hr)"], axis = 1)
    table["HR_MIR"] = table.apply(lambda x: hr.loc[x["Mir"],"log(hr)"], axis = 1)
    
    return table 


def filter_low_express_feature(exprDF, treshold = 5, sample_percentage = 0.4):
    """
    Function to filter the low expressed miRNA and mRNA.
    Args:   
        exprDF  df Concat Dataframe rows are samples and cols are gene/mirs
        treshold int treshold of min number of reads
        sample_percentage float percentage of samples that have more than treshold reads
    Returns:
        dfExprs_filtered   df  A  matrix that includes the filtered expression matrix
    """
    # Set expression threshold and sample percentage for filtering
    threshold = np.log2(treshold)

    # Filter features with low expression in more than 90% of samples
    dfExprs_filtered = exprDF.loc[:, (exprDF > threshold).sum(axis=0) / exprDF.shape[0] >= sample_percentage]
    return dfExprs_filtered


def all_methods(exprDF, lMirUser = None, lGeneUser = None, n_core = 2, hr = False, k = 10, background = True, \
                test = False, add_target = True, filter_low_express = True):
    """
    Function to calculate all coefficient
    of each pair of miRNA-mRNA, return a matrix of correlation coefficients 
    with columns are miRNAs and rows are mRNAs.

    Args:   
        exprDF  df Concat Dataframe rows are samples and cols are gene/mirs

    Returns:
  
    """
    import copy

    if filter_low_express:
        exprDF = filter_low_express_feature(exprDF)

    lMir, lGene = header_list(exprDF=exprDF)
   
    if lGeneUser is not None:
        lGene = intersection(lGene,lGeneUser)

    if lMirUser is not None:
        lMir = intersection(lMir,lMirUser)

    print("Obtain Concat Gene and MIR")
    print("Number of genes: " + str(len(lGene)),flush=True)
    print("Number of miRNAs: " + str(len(lMir)),flush=True)
    print("Number of samples: " + str(exprDF.shape[0]),flush=True)
    print("Number of features: " + str(exprDF.shape[1]),flush=True)
    print(lMirUser,flush=True)
    print(exprDF.head(),flush=True)
    
    if test:
        modelList = [[spearman,"Rho"],
                    [pearson,"R"],
                    [kendall,"Tau"],
                    ]
    else:
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

    table = process_matrix_list(lTuple, add_target=add_target)

    table = table.loc[~table.duplicated(), :]


    if hr:
        print("\nHazard Ratio")
        table = hazard_ratio_mirgen(exprDF, table, lGeneUser=lGene, lMirUser=lMir, n_core=n_core)

    if background:
        print("\nBackground")
        table = background_estimation(exprDF, table, n_gene=3000, n_core=n_core, pval=False)
        
    table = adj_pval(table)

    return table, lTuple[1][0]


#################
### Background ##
#################
def process_zscore(lMir, table_random):
    lMean = []
    lStd = []
    for mir in lMir:
        x = table_random.loc[table_random.Mir == mir,:].R.tolist()
        try:
            mean = np.mean(x)
            standard_deviation = np.std(x)
        except:
            mean = np.nan
            standard_deviation = np.nan
        finally:
            lMean.append(mean)
            lStd.append(standard_deviation)
            
    df = pd.DataFrame({"Mir":lMir,"Mean":lMean,"Standard Deviation":lStd})
    
    return df


def get_mean_deviation(table_random, n_core = 4):
    import functools
    from multiprocessing import Pool

    ### Intersect with Gene and Mir from table##
    lMir = table_random.Mir.unique().tolist()

    ##Split List
    np_list_split = np.array_split(lMir, n_core)
    split_list = [i.tolist() for i in np_list_split]
    #split_list = same_length(split_list)

    #Fix Exprs Variable
    partial_func = functools.partial(process_zscore, table_random=table_random)

    #Generating Pool
    pool = Pool(n_core)
    lres = pool.map(partial_func, split_list)
    res = pd.concat(lres)
    pool.close() 
    pool.join()
    return res


def z_score(value, mean, std):
    return (value - mean) / std


def background_estimation(exprDF, table, n_gene = 3000, method = "R", n_core = 6, pval = False):
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
    import random 
    from pandarallel import pandarallel
    pandarallel.initialize(verbose=1, nb_workers=n_core)

    lMir, lGene = header_list(exprDF=exprDF)

    #Select Random Genes for correlation
    random.seed(10)
    lGene = random.sample(lGene, n_gene)

    dfCor, dfPval = pearson(exprDF, lGeneUser = lGene, n_core = n_core, pval = pval)

    ## Get table with number
    table_random = process_matrix_list([(dfCor,"R")], add_target=True)
    table_random["Number"] = table_random["Prediction Tools"].str.count("1")
    table_filter = table_random.loc[table_random["Number"] <= 1,:]

    #Get Mean and Std
    res = get_mean_deviation(table_filter)
    res.index = res.Mir
    
    #Get Z-score, and P-value
    res_z_p = table.parallel_apply(lambda row: \
                              z_score_value(row.R, res, row.Mir), axis = 1)
    #print(res_z_p)
    #res_z_p = res_z_p.apply(lambda col: col.dropna())
    table["Background Z-Score"] = res_z_p.apply(lambda x: x[0])
    table["Background_Pval"] = res_z_p.apply(lambda x: x[1])

    return table

def z_score_value(x, res, mir):
    from scipy.stats import norm

    try:
        z = z_score(x, res.loc[mir,"Mean"], res.loc[mir,"Standard Deviation"])
        p_value = norm.cdf(x=x,loc=res.loc[mir,"Mean"],scale=res.loc[mir,"Standard Deviation"])
    except:
        z, p_value = 0, 1
    return z, p_value

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
    else:
        if method in ["R","Rho","Tau"]:
            query_string = f"""
                    (({method} < {low_coef} | {method} > {high_coef} )) \
                    & \
                    (({method}_fdr_bh < {pval}))
            """
        else:
            query_string = f"""
                    (({method} < {low_coef} | {method} > {high_coef} ))
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
            try:
                matrix = matrix.loc[gene,mir]#Subset the Correlation matrix to the heatmap
            except:
                matrix = None

        return table, matrix
    else:
        matrix = None
        return table, matrix


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
    

