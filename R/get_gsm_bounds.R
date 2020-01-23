#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

library("optparse")
library(ldbounds)

option_list = list(
  make_option(c("-b", "--batches"), type="numeric", default=5,help="num batches", metavar="numeric"),
  make_option(c("-f", "--factor"), type="numeric", default=0.05,help="asf factor", metavar="numeric"),
  make_option(c("-a", "--alpha"), type="numeric", default=0.05,help="total alpha to spend", metavar="numeric"),
  make_option(c("-o", "--out"), type="character", default="out.txt",help="output file name", metavar="character")
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

start_time <- 1/opt$batches
time <- seq(start_time, 1, length = opt$batches)
# my_asf <- function(t){(1 - opt$factor^t)/(1 - opt$factor)}
# CUSTOM_ASF = 5
# asf_bounds = bounds(time, alpha=opt$alpha, iuse=CUSTOM_ASF, asf=my_asf)

# These are just pocock bounds I think
POWER_ASF = 3
ASF_PHI = 1
asf_bounds = bounds(time, alpha=opt$alpha, iuse=POWER_ASF)

#print(asf_bounds)

write.table(
  asf_bounds$upper.bounds,
  file=opt$out,
  row.names=FALSE,
  col.names=FALSE)
