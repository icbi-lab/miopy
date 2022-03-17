de_edger = function(path, meta_path, out_path, bNormal,bFilter, bPaired, group){
  pacman::p_load(limma,dplyr,edgeR)

  #Read DF  
  raw_count = read.csv(path, check.names = F, header = TRUE, row.names = 1)
  metadata = read.csv(meta_path, check.names = F, header = TRUE, row.names = 1) 
  ##Intersect
  lSample = intersect(rownames(metadata), colnames(raw_count))
  
  #Filter by Sample
  raw_count = raw_count[lSample]
  metadata = metadata[lSample,]
  print(head(raw_count))

  ##Create DGEList

                
    
  if (bNormal) {
    print("Applyin TMM")
    dgList = DGEList(counts = raw_count, 
                     genes = rownames(raw_count))
    dgList <- calcNormFactors(dgList, method = "TMM")
      if (bFilter){
        drop <- which(apply(edgeR::cpm(dgList), 1, max) < 1)
        dgList = dgList[-drop,]
      }
    
    counts = voom(dgList)
    
  } else{
    
      if (bFilter){
        drop <- which(apply(edgeR::cpm(raw_count), 1, max) < 1)
        raw_count = raw_count[-drop,]
      }
    
    counts = raw_count
  }
  
  if (bPaired){
    design <- model.matrix(as.formula(sprintf("~sample+%s", group)), data = metadata)
  } else{
    design <- model.matrix(as.formula(sprintf("~%s", group)), data = metadata)
  }
  
  print(counts)
  vfit <- lmFit(counts, design)
  efit <- eBayes(vfit)
  dfPval = topTable(efit, number = Inf, coef = 2)
  
  write.csv(dfPval, out_path)
  
return(dfPval)
}


arguments = function(){
  pacman::p_load(optparse)
  option_list = list(
    make_option(c("-f", "--file"), type="character", default=NULL, 
                help="dataset file name", metavar="character"),
    
    make_option(c("-m", "--metadata"), type="character", default=NULL, 
                help="Metadata file name", metavar="character"),
    
    make_option(c("-o", "--out"), type="character", default=NULL, 
                help="Metadata file name", metavar="character"),
    
    make_option(c("-n", "--normalize"), type="logical", default=NULL, 
                help="Normalize data. If True the script applies TMM normalziation", metavar="bool"),
    
    make_option(c("-t", "--filter"), type="logical", default=NULL, 
                help="Filter Low Expression", metavar="character"),

    make_option(c("-p", "--paired"), type="logical", default=FALSE, 
                help="Paired Samples", metavar="character"),

    make_option(c("-g", "--group"), type="character", default="event", 
                help="Group Header", metavar="character")
  ); 
  
  opt_parser = OptionParser(option_list=option_list);
  opt = parse_args(opt_parser);
  return(opt)
}
arg = arguments()

de_edger(path = arg$file,meta_path = arg$metadata, out_path = arg$out, bNormal = arg$normalize,  bFilter = arg$filter, bPaired = arg$paired, group = arg$group)