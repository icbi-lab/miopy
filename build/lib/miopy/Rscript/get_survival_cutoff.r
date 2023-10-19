rma_norm = function(path, out_path, time, event,target){
  pacman::p_load(survminer)
    
  data = read.csv(path, check.names = F, header = TRUE, row.names = 1)
  
  res = surv_cutpoint(data = data, time = time, event = event, variables = target)
  
  write.csv(res$cutpoint, file = out_path)
  
return(df_result_r)
}

arguments = function(){
  pacman::p_load(optparse)
  option_list = list(
    make_option(c("-f", "--file"), type="character", default=NULL, 
                help="dataset file name", metavar="character"),
    make_option(c("-o", "--out"), type="character", default=NULL, 
                help="Out file name/path", metavar="character"),
    make_option(c("-t", "--time"), type="character", default="time", 
                help="Time Column", metavar="character"),
    make_option(c("-e", "--event"), type="character", default="event", 
                help="Event Column", metavar="character"),
    make_option(c("-g", "--target"), type="character", default="target", 
                help="Event Column", metavar="character")
  ); 
  
  opt_parser = OptionParser(option_list=option_list);
  opt = parse_args(opt_parser);
  return(opt)
}

arg = arguments()

print(rma_norm(arg$file, arg$out, arg$time, arg$event,arg$target))