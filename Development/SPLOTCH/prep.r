library(stringr)
file <- commandArgs()[6]
m <- read.table(file,TRUE,'\t',row.names=1)
rownames(m) <- lapply(rownames(m), function(x) {str_replace(x, "x", "_")})
write.table(t(m), paste0(file,'_splotch'), col.names=NA, quote=FALSE, sep='\t')
annotation <- strsplit(file, '_')[[1]][3]
n <- t(m)[1,,drop=FALSE]
n[] <- 1
rownames(n) <- annotation
if (annotation=='hp') {
    n <- rbind(0, n)
} else {
    n <- rbind(n, 0)
}
rownames(n) <- c("dt", "hp")
write.table(n, paste0(file,'_splotch_annot'), col.names=NA, quote=FALSE, sep='\t')