rma_norm = function(path, out_path, bFilter){
  pacman::p_load(edgeR)
    
  eset = read.csv(path, check.names = F, header = TRUE, row.names = 1)
  
  eset = limma::voom(eset)[[1]]
    
  
  
  if (bFilter){
    eset = eset[rowSums(eset) > 5,]
  }
  write.csv(eset, file = out_path)

return(eset)
}

arguments = function(){
  pacman::p_load(optparse)
  option_list = list(
    make_option(c("-f", "--file"), type="character", default=NULL, 
                help="dataset file name", metavar="character"),
    make_option(c("-o", "--out"), type="character", default=NULL, 
                help="Out file name/path", metavar="character"),
    make_option(c("-t", "--filter"), type="logical", default=NULL, 
                help="Filter Low Expression", metavar="character")
  ); 
  
  opt_parser = OptionParser(option_list=option_list);
  opt = parse_args(opt_parser);
  return(opt)
}

arg = arguments()

print(rma_norm(arg$file, arg$out, arg$filter))