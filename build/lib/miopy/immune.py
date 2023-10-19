import numpy as np
import pandas as pd

def ipsmap(x):
    if x<=0:
        ips = 0
    elif x >= 3: 
        ips = 10
    else:
        ips = round(x*10/3,0)
    return ips

def ips(row,gene_ips):
    mGE = row.mean()
    sGE = row.std()
    gene_rows = row.index.tolist()
    gene_set = gene_ips["GENE"].tolist()
    common_gene = list(set(gene_rows).intersection(gene_set))
    unique_ips_genes = gene_ips["NAME"].unique().tolist()

    #print(f"[+] {len(common_gene)}/{len(gene_set)}")
    Z1  = (row[common_gene] - mGE) / sGE
    W1 = gene_ips["WEIGHT"]
    W1.index = gene_ips.GENE
    #print(W1)
    WEIGHT = pd.Series(index=unique_ips_genes,dtype='float64')
    MIG = pd.Series(index=unique_ips_genes,dtype='float64')
    for name in unique_ips_genes:
        lGene = gene_ips.loc[gene_ips.NAME == name,"GENE"].tolist()
        MIG[name] = Z1.loc[list(set(Z1.index).intersection(lGene))].mean()
        WEIGHT[name] = W1.loc[list(set(W1.index).intersection(lGene))].mean()
    WG = MIG * WEIGHT
    MHC = WG[gene_ips.loc[gene_ips.CLASS == "MHC","NAME"].tolist()].mean()
    CP = WG[gene_ips.loc[gene_ips.CLASS == "CP","NAME"].tolist()].mean()
    EC = WG[gene_ips.loc[gene_ips.CLASS == "EC","NAME"].tolist()].mean()
    SC = WG[gene_ips.loc[gene_ips.CLASS == "SC","NAME"].tolist()].mean()
    AZ = np.sum((MHC,CP,EC,SC))
    IPS = ipsmap(AZ)
    
    return pd.Series([MHC, CP, EC, SC, AZ, IPS], index = ("MHC","CP","EC","SC","AZ","IPS"), name = row.name)
    

   