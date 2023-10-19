de_array_limma = function(path, meta_path, out_path, bNormal,bFilter,bPaired){
  
  pacman::p_load(limma,oligo, genefilter)

  #Read DF  
  eset = read.csv(path, check.names = F, header = TRUE, row.names = 1)
  metadata = read.csv(meta_path, check.names = F, header = TRUE, row.names = 1) 
  
  if(bPaired){
    colnames(metadata) = c("sample","condition","patient")
  
  }else{
    colnames(metadata) = c("sample","condition")

  }
  
  ##Intersect
  lSample = intersect(metadata$sample, colnames(eset))
  
  
  if (bNormal) {
    print("Applyin RMA")
    eset = limma::voom(eset)[[1]]
    
  }
  
  if (bFilter){
    eset = eset[rowSums(eset) > 5,]
  }
  
  #Filter by Sample
  eset = eset[lSample]
  metadata = metadata[metadata$sample %in% lSample,]
  
  #Design Matrix
  if (bPaired){
  design <- model.matrix(~metadata$patient + metadata$condition)
  } else{
   design <- model.matrix(~metadata$condition)
  }
  #Limma
  fit <- lmFit(eset, design.mat)
  fit<- eBayes(fit)
  
  #Get DE Table
   if (bPaired){
    res = topTable(fit, coef=5, adjust.method = "BH", sort.by="P", number=Inf)
  } else{
    res = topTable(fit, coef=1, adjust.method = "BH", sort.by="P", number=Inf)
  }
  #Save File
  write.csv(res, out_path)
  
}


arguments = function(){
  pacman::p_load(optparse)
  option_list = list(
    make_option(c("-f", "--file"), type="character", default=NULL, 
                help="dataset file name", metavar="character"),
    
    make_option(c("-m", "--metadata"), type="character", default=NULL, 
                help="Metadata file name", metavar="character"),
    
    make_option(c("-o", "--out"), type="character", default=NULL, 
                help="Out Res file name", metavar="character"),
    
    make_option(c("-n", "--normalize"), type="logical", default=NULL, 
                help="Normalize data. If True the script applies TMM normalziation", metavar="bool"),
    
    make_option(c("-t", "--filter"), type="logical", default=NULL, 
                help="Filter Low Expression", metavar="character")
    make_option(c("-p", "--paired"), type="logical", default=FALSE, 
                help="Paired Samples", metavar="character")
  ); 
  
  opt_parser = OptionParser(option_list=option_list);
  opt = parse_args(opt_parser);
  return(opt)
}


arg = arguments()

de_array_limma(path = arg$file, meta_path = arg$metadata, out_path = arg$out, bNormal = arg$normalize,  bFilter = arg$filter, , bPaired = arg$paired)