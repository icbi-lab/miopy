import pandas as pd
import argparse
from miopy.correlation import (read_count, concat_matrix, all_methods, tmm_normal, voom_normal, 
                             get_mir_gene_ratio, header_list, 
                             intersection, gene_set_correlation,
                             process_matrix_list)
from pathlib import Path


def arg():
    """
    Función que recibe los parámetros introducidos como argumento a la hora
    de llamar al programa desde la terminal. Facilita su automatización
    
    Returns:
        args (argparse) Objeto que almacena los argumentos
    """
    
    parser=argparse.ArgumentParser (
            description ='''           
            '''
            )
    
    parser.add_argument("-m","--mir", dest="mirPath", action="store", required=True, \
            help = "Path to the miRNA expression file", default = None)
        
    parser.add_argument("-g","--gene", dest="genePath", action="store", required=True, \
            help = "Path to the gene expression file", default = None)
    
    parser.add_argument("-gs","--geneset", dest="gsPath", action="append", required=False, \
            help = "Path to the geneset file. Call multiple time to use different GS", default = None)

    parser.add_argument("-mt","--meta", dest="metadataPath", action="store", required=False, \
            help = "Path to the clinical file. Columns have to be: sample, group, time, event", default = None)

    parser.add_argument("-o","--out", dest="outPath", action="store", required=False, \
            help = "Directory to save the output files", default = ".")

    parser.add_argument("-t","--technology", dest="technology", action="store", required=False, \
            help = "Sequencing technology used in the dataset: sequencing or microarray", default = "sequencing")

    parser.add_argument("-mode","--mode", dest="method", action="store", required=False, \
            help = "Mode to run the correlation: Correlation or GeneSetScore", default = "Correlation")

    parser.add_argument("-f","--filter", dest="FilterChoice", action="store", required=False, \
            help = """Apply to differential expresion analysis to filter the gene and mirna:
                    NF:  No Filter,
                    CCU: Condition1 vs Condition2 Unpaired
                    CCP: Condition1 vs Condition2 Paired """, default = "NF")

    parser.add_argument("-fc","--logfc", dest="logfc", type = int, action="store", required=False, \
            help = "Absolute value of the Log(FC), use to filter the DE genes and mirnas", default = 1.22)

    parser.add_argument("-pv","--adjust-pval", dest="pval", type = int, action="store", required=False, \
            help = "Absolute value of the pval, use to filter the DE genes and mirnas", default = 0.005)

    parser.add_argument("-hr","--hazard", dest="survival", action="store", type=bool, nargs='?',
                        default=False, help="Obtain log(Hazard Ratio for the gene/miRNA")

    parser.add_argument("-n","--normal", dest="normal", action="store", type=bool, nargs='?',
                        default=False, help="Normalize matrix counts")
    parser.add_argument("-p","--processor", dest="n_core", type = int, action="store", required=False, \
            help = "NUmber of cores", default = 2)

    args = parser.parse_args()
    
    return args

def run_correlation(mirPath, genePath, gsPath=None, metadataPath = None, outPath=".", technology = "sequencing", method = "Correlation", FilterChoice = "NF", \
                    normal = False, logfc = 1.2, pval = 0.005, survival = False, n_core = 2, ratio = False):
    #Get ExprDF
            
    if mirPath.endswith(".csv"):
        sep = ","
    else:
        sep = "\t"

    print("Obtenido los ficheros")
            
            #Read DF
    if normal and technology == "sequencing":
        mirExpr = tmm_normal(mirPath)
        geneExpr = tmm_normal(genePath)
        print("Normalizados los ficheros")

    elif normal and technology == "microarray":
        mirExpr = voom_normal(mirPath)
        geneExpr = voom_normal(genePath)
        print("Normalizados los ficheros")

    else:
        mirExpr = read_count(mirPath, sep)
        geneExpr = read_count(genePath, sep)
        print("Ya normalizados")

    dfExpr = concat_matrix(mirExpr, geneExpr)
    dfMeta = pd.read_csv(metadataPath, index_col=0)

    lMir, lGene = header_list(exprDF=dfExpr)

    if gsPath != None:
        lGene = []
        for gs in gsPath:
            lGene += list(open(gs,"r").read().split())
        lGene = list(set(lGene))


    #Create Ratio mir/gene
    if ratio:
        dfExpr = get_mir_gene_ratio(dfExpr, lGeneUser=lGene)

        #Add Label Column
    if survival:
        dfMeta = dfMeta[["event","time"]]
        dfExpr = pd.concat([dfMeta,dfExpr], axis = 1).dropna()



    if method == "Correlation":
        table, dfPearson = all_methods(dfExpr, lMirUser = lMir, lGeneUser = lGene, n_core = n_core, hr = survival)
            
        table = table.round(4)
        table.to_csv(outPath+"/CorrelationTable.csv")
        dfPearson.to_csv(outPath+"/CorrelationMatrix.csv")

    return None



def main():
    argu = arg()
    run_correlation(argu.mirPath, argu.genePath, gsPath=argu.gsPath, metadataPath=argu.metadataPath, \
        technology=argu.technology, method=argu.method, FilterChoice=argu.FilterChoice, survival=argu.survival, \
        n_core=argu.n_core, normal=argu.normal, logfc = argu.logfc, pval=argu.pval)


if __name__ == "__main__":
    main()