
import pandas as pd
import numpy as np
from os import path
import io
import numpy as np


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