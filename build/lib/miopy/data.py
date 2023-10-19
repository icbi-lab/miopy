
from importlib import resources
import pandas as pd
from os import path
import os
import requests
import zipfile

resources = {
    "target_matrix" :  "https://data.mendeley.com/public-files/datasets/nxcrssvwgs/files/aead9866-5692-4ae6-92c3-c3c1c7f92311/file_downloaded",
    "target_table" : "https://data.mendeley.com/public-files/datasets/nxcrssvwgs/files/4f9096f1-55d3-4ef8-bb5d-6c86df314675/file_downloaded",
    "mio_datasets": "https://data.mendeley.com/public-files/datasets/nxcrssvwgs/files/0e6f58ba-6e68-494c-bd1b-80fe9a978815/file_downloaded"
}
    
###############################################################################

#####################
## Obtain Data dir ##
#####################

def _get_path_data():
    """Return the path to the data directory."""
    custom_path = os.environ.get("MIO_CACHE_DIR", None)

    if custom_path is not None:
        path_data = custom_path
    else:
        path_data = str(os.path.dirname(__file__))
    
    # Create data directory if not exists
    if not path.exists(path_data):
        os.mkdir(path_data)

    return path_data

    
###############################################################################

############################
## Download data from web ##
############################


import os
import requests
import zipfile
from os import path

def download_target_matrix():
    """Download target matrix from web."""

    url = resources["target_matrix"]
    path_data = _get_path_data()
    print("[+] Downloading target matrix from web...", flush=True)
    path_file = path.join(path_data, 'MATRIX.pickle.gz')
    
    # Use requests to download the file
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(path_file, 'wb') as file:
            file.write(response.content)
        print("[+] Done!", flush=True)
    else:
        print("[!] Failed to download the target matrix. Status code: ", response.status_code)
###############################################################################
def download_target_table():
    """Download target table from web."""

    url = resources["target_table"]
    path_data = _get_path_data()
    print("[+] Downloading target table from web...", flush=True)
    path_file = path.join(path_data, 'MATRIX_TABLE.pickle.gz')
    
    # Use requests to download the file
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(path_file, 'wb') as file:
            file.write(response.content)
        print("[+] Done!", flush=True)
    else:
        print("[!] Failed to download the target table. Status code: ", response.status_code)
###############################################################################
def download_mio_datasets():
    """
    Function to download the MIO datasets from the web.
    The zip file contains a directory with the following files:
        - PROJECT_miRNAs.csv
        - PROJECT_RNAseq.csv
        - metadata.csv
    Dataset will be extracted in the Dataset/ directory.
    """
    url = resources["mio_datasets"]
    path_data = _get_path_data()
    path_datasets = path.join(path_data, 'dataset')
    if not path.exists(path_datasets):
        os.mkdir(path_datasets)
    print("[+] Downloading MIO datasets from the web...", flush=True)
    path_file = path.join(path_data, 'mio_datasets.zip')
    
    # Use requests to download the file
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(path_file, 'wb') as file:
            file.write(response.content)
        print("[+] Done!", flush=True)
        print("[+] Extracting MIO datasets...", flush=True)
        with zipfile.ZipFile(path_file, 'r') as zip_ref:
            zip_ref.extractall(path_datasets)
        print("[+] Done!", flush=True)
        print("[+] Removing zip file...", flush=True)
        os.remove(path_file)
    else:
        print("[!] Failed to download MIO datasets. Status code: ", response.status_code)

###############################################################################

###############
## Load data ##
###############

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
    Download from web if not exists.

    Contains the following fields:
        col          gene symbol
        index        mirbase mature id
    """
    path_file = path.join(_get_path_data(), 'MATRIX.pickle.gz')
    if not path.exists(path_file):
        download_target_matrix()
    return pd.read_pickle(path_file)



def load_table_counts():
    import pkg_resources
    """Return a long table with miRNA/Gene prediction tool.
    """
    path_file = path.join(_get_path_data(), 'MATRIX_TABLE.pickle.gz')
    if not path.exists(path_file):
        download_target_table()
    return pd.read_pickle(path_file)

def load_synthetic():
    import pkg_resources
    """Return a dataframe about Gene/Gene synthetic lehal

    Contains the following fields:
        col1        gene symbol
        col2        gene symbol
    """
    stream = pkg_resources.resource_stream(__name__, 'data/SL.tsv')
    return pd.read_csv(stream, sep = "\t",header=None, names = ["GeneA","GeneB"])

def load_dataset(project_name):
    import pkg_resources
    """Return three dataframe about miRNA and RNAseq expression and metadata
    from a specific project.
    """
    project_name = project_name.upper()
    dataset_path = path.join(_get_path_data(), 'dataset')
    if not path.exists(dataset_path):
        download_mio_datasets()
    project_path = path.join(dataset_path, project_name)

    dfMir = pd.read_csv(path.join(project_path, project_name + '_miRNAs.csv'), index_col=0)
    dfRna = pd.read_csv(path.join(project_path, project_name + '_RNAseq.csv'), index_col=0)
    metadata = pd.read_csv(path.join(project_path, 'metadata.csv'), index_col=0)
    return dfMir, dfRna, metadata

def load_gene_ips():
    import pkg_resources
    """Return a dataframe about Gene/Gene synthetic lehal

    Contains the following fields:
        col1        gene symbol
        col2        gene symbol
    """
    stream = pkg_resources.resource_stream(__name__, 'data/IPS_genes.txt')
    return pd.read_csv(stream, sep = "\t",header=None, names = ["GENE","NAME", 	"CLASS", "WEIGHT"])
    

def test():
    """Test function."""
    print(load_matrix_header())
    print(load_matrix_counts())
    print(load_table_counts())
    print(load_synthetic())
    print(load_dataset("TCGA-LUAD"))
    print(load_gene_ips())
