file <- commandArgs()[6]
m <- read.table(file,TRUE,'\t',row.names=1)
n <- m[,names(sort(colSums(m==0))[1:150])]
write.table(n, paste0(file,'_filtered'), col.names=NA, quote=FALSE, sep='\t')
