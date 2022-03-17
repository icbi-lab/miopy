tmm_norm = function(path, out_path, bFilter){
  pacman::p_load(edgeR)
    
  raw_count = read.csv(path, check.names = F, header = TRUE, row.names = 1)
  
  dgList = DGEList(counts = raw_count, 
                    genes = rownames(raw_count))
   
  if(bFilter){
    drop <- which(apply(cpm(dgList), 1, max) < 1)
  
    dgList = dgList[-drop,]
  } 
 
  
  dgList <- calcNormFactors(dgList, method = "TMM")
  
  Analysis <- estimateDisp(dgList, robust = TRUE)
  Analysis$common.dispersion
  dgList <- estimateCommonDisp(dgList)
  dgList <- estimateTagwiseDisp(dgList)
  norm_counts.table <- voom(dgList)
  
  TMM_norm = norm_counts.table
  write.csv(TMM_norm, file = out_path)
return(TMM_norm)
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

print(tmm_norm(arg$file, arg$out, arg$filter))