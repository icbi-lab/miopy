
import pandas as pd
import numpy as np
from os import path
import io
import numpy as np
import re

def _get_path_data():
    return path.join(path.dirname(__file__), 'data')

def get_target_query(method = "and", lTarget = []):
   #Build DB query:
    dbQ = ""
    lHeader = load_matrix_header()

    if method == "and":
        for db in lHeader:
            if db in lTarget:
                dbQ += "1"
            else:
                dbQ += "."

    elif method == "or":
        lQ = []
        nullString = "."*len(lHeader)
        for db in lHeader:
            if db in lTarget:
                i = lHeader.index(db)
                q = list(nullString)
                q[i] = "1"
                lQ.append("".join(q))
        dbQ="("+"|".join(lQ)+")"
    return dbQ


def load_matrix_header():
    import codecs
    import pkg_resources
    """Return a dataframe about miRNA/Gene prediction tool.

    Contains the following fields:
        col          gene symbol
        index        mirbase mature id
    """
    stream = pkg_resources.resource_stream(__name__, 'data/MATRIX_LIST.txt')
    return stream.read().decode('utf-8').split()



def load_matrix_counts():
    import pkg_resources
    """Return a dataframe about miRNA/Gene prediction tool.

    Contains the following fields:
        col          gene symbol
        index        mirbase mature id
    """
    stream = pkg_resources.resource_filename(__name__, 'data/MATRIX.pickle.gz')
    return pd.read_pickle(stream)



def load_table_counts():
    import pkg_resources
    """Return a dataframe about miRNA/Gene prediction tool.

    Contains the following fields:
        col          gene symbol
        index        mirbase mature id
    """
    stream = pkg_resources.resource_filename(__name__, 'data/MATRIX_TABLE.pickle.gz')
    return pd.read_pickle(stream)


def load_synthetic():
    import pkg_resources
    """Return a dataframe about Gene/Gene synthetic lehal

    Contains the following fields:
        col1        gene symbol
        col2        gene symbol
    """
    stream = pkg_resources.resource_stream(__name__, 'data/SL.tsv')
    return pd.read_csv(stream, sep = "\t",header=None, names = ["GeneA","GeneB"])


def load_dataset():
    import pkg_resources
    """Return a 3dataframe about Gene/Gene synthetic lehal

    Contains the following fields:
        col1        gene symbol
        col2        gene symbol
    """
    stream = pkg_resources.resource_stream(__name__, 'dataset/TCGA-OV_miRNAs.csv')
    dfMir = pd.read_csv(stream, index_col=0)

    stream = pkg_resources.resource_stream(__name__, 'dataset/TCGA-OV_RNAseq.csv')
    dfRna = pd.read_csv(stream, index_col=0)

    stream = pkg_resources.resource_stream(__name__, 'dataset/metadata.csv')
    metadata = pd.read_csv(stream, index_col=0)
    return dfMir, dfRna, metadata



def read_count(file_path, sep = "\t"):
    """
    Function to read dataset from the csv

    Args:
        file_path string path to the csv file

    Return:
        df  dataframe DataFrame with the matrix count
    """
    df = pd.read_csv(file_path, index_col=0, sep = sep)

    return df


def count_db(table):
    table["Tool Number"] = table["Prediction Tools"].str.count("1")

    return table


def concat_matrix(mrnaDF, mirDF):
    """
    Function to concat the miR and Gene expression Matrix.
    The sample name have to be the same in both Matrix.
    With dropna, pandas remove all genes/mir with NA.

    Args:
        mrnaDF  df  Dataframe rows are genes and cols are samples
        mirDF   df  Dataframe rows are mirs and cols are samples
    
    Return:
        exprDF  df Concat Dataframe rows are samples and cols are gene/mirs

    """
    # axis = 0 -> axis = 1 \/
    exprDF = pd.concat([mrnaDF, mirDF]).dropna(axis = 1).transpose()

    return exprDF


def header_list (exprDF):
    """
    Function to obtain a list of the miR and genes present in the dataframe

    Args:
        exprDF  df Concat Dataframe rows are samples and cols are gene/mirs

    Return:
        lMir    list    miR List
        lGene   list    Gene List
    """
    lAll = exprDF.columns.tolist()
    patMir = re.compile("^hsa-...-*")
        
    lMir = [i for i in lAll if patMir.match(i)]
    lGene = [i for i in lAll if not patMir.match(i)]

    return lMir, lGene


def intersection(lst1, lst2): 
    """
    Python program to illustrate the intersection 
    of two lists using set() method
    """
    return list(set(lst1).intersection(lst2)) 


def GetMethodList(lHeader):
    lDefault = ["R", "Rho","Tau","Hoeffding","RDC","Lasso", "Ridge","ElasticNet","Lars","Random Forest","Log(HR)"]
    return intersection(lHeader,lDefault)


def get_confident_df(df):
    dfConf = df / 40
    return dfConf


def get_confident_serie(serie):
    serie = serie / 40
    return serie
