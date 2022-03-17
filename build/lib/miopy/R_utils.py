import subprocess
from os import path, remove
import pandas as pd
import uuid

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


def tmm_normalization(fPath, bFilter):
    lRemove = []
    r_path = path.join(path.dirname(__file__), 'Rscript')

    if not isinstance(fPath, str):
        tempPath = "%s.csv"%(str(uuid.uuid1()))
        fPath.to_csv(tempPath)
        fPath = tempPath
        lRemove.append(tempPath)

    outPath = "%s.csv"%(str(uuid.uuid1()))

    r_path = path.join(path.dirname(__file__), 'Rscript')
    cmd = ["Rscript", path.join(r_path,"get_normal_counts.r"),\
            "-f", fPath, "-o", outPath, "-t", bFilter]

    a = subprocess.run(cmd,stdout=subprocess.PIPE)
    
    df = read_count(outPath,",")
    lRemove.append(outPath)

    for p in lRemove:
        remove(p)

    return df

def get_survival_cutoff(fPath, time = "time", event = "event", target = "target"):
    lRemove = []
    r_path = path.join(path.dirname(__file__), 'Rscript')

    if not isinstance(fPath, str):
        tempPath = "%s.csv"%(str(uuid.uuid1()))
        fPath.to_csv(tempPath)
        fPath = tempPath
        lRemove.append(tempPath)

    outPath = "%s.csv"%(str(uuid.uuid1()))

    r_path = path.join(path.dirname(__file__), 'Rscript')
    cmd = ["Rscript", path.join(r_path,"get_survival_cutoff.r"),\
            "-f", fPath, "-o", outPath,"-t", time,"-e",event,"-g",target]

    a = subprocess.run(cmd,stdout=subprocess.PIPE)
    
    df = pd.read_csv(outPath,",")
    print(df)
    lRemove.append(outPath)

    for p in lRemove:
        remove(p)

    return float(df["cutpoint"])

def deg_edger(fPath, metaFile, bNormal="False", bFilter="False", bPaired="False", group = "event"):
    
    lRemove = []
    r_path = path.join(path.dirname(__file__), 'Rscript')

    if not isinstance(fPath, str):
        tempPath = "%s.csv"%(str(uuid.uuid1()))
        fPath.to_csv(tempPath)
        fPath = tempPath
        lRemove.append(tempPath)

    if not isinstance(metaFile, str):
        tempMetaFile = "%s.csv"%(str(uuid.uuid1()))
        metaFile.to_csv(tempMetaFile)
        metaFile = tempMetaFile
        lRemove.append(tempMetaFile)
        
    outPath = "%s.csv"%(str(uuid.uuid1()))

    cmd = ["Rscript", path.join(r_path,"get_deg.r"),\
            "-f", fPath, "-m",metaFile,"-o", outPath,\
            "-n", bNormal,"-t",bFilter, "-p", bPaired,\
            "-g",group]

    a = subprocess.run(cmd,stdout=subprocess.PIPE)
    
    DegDf = read_count(outPath,",")

    lRemove.append(outPath)

    for p in lRemove:
        remove(p)

    return DegDf


def voom_normalization(fPath, bFilter):
    lRemove = []
    r_path = path.join(path.dirname(__file__), 'Rscript')

    if not isinstance(fPath, str):
        tempPath = "%s.csv"%(str(uuid.uuid1()))
        fPath.to_csv(tempPath)
        fPath = tempPath
        lRemove.append(tempPath)

    outPath = "%s.csv"%(str(uuid.uuid1()))

    r_path = path.join(path.dirname(__file__), 'Rscript')
    cmd = ["Rscript", path.join(r_path,"get_voom.r"),\
            "-f", fPath, "-o", outPath, "-t", bFilter]

    a = subprocess.run(cmd,stdout=subprocess.PIPE)
    
    df = read_count(outPath,",")
    lRemove.append(outPath)

    for p in lRemove:
        remove(p)

    return df


def deg_limma_array(fPath, metaFile, bNormal,bFilter, bPaired):
    
    lRemove = []
    r_path = path.join(path.dirname(__file__), 'Rscript')

    if not isinstance(fPath, str):
        tempPath = "%s.csv"%(str(uuid.uuid1()))
        fPath.to_csv(tempPath)
        fPath = tempPath
        lRemove.append(tempPath)

    if not isinstance(metaFile, str):
        tempMetaFile = "%s.csv"%(str(uuid.uuid1()))
        metaFile.to_csv(tempMetaFile)
        metaFile = tempMetaFile
        lRemove.append(tempMetaFile)
        
    outPath = "%s.csv"%(str(uuid.uuid1()))

    cmd = ["Rscript", path.join(r_path,"get_de_array_limma.r"),\
            "-f", fPath, "-m",metaFile,"-o", outPath,\
            "-n", bNormal,"-t",bFilter, "-p", bPaired]

    a = subprocess.run(cmd,stdout=subprocess.PIPE)
    
    DegDf = read_count(outPath,",")

    lRemove.append(outPath)

    for p in lRemove:
        remove(p)

    return DegDf