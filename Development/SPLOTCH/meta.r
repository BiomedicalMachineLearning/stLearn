library(stringi)
m <- matrix(,,7)
colnames(m) <- c("Name", "Level 1", "Level 2", "Level 3", "Count file", "Annotation file", "Image file")
files <- list.files(pattern="unified")
for (file in files) {
    annot <- strsplit(file, '_')
    annot[[1]][6] <- 'splotch_annot'
    m <- rbind(m, c(file, strsplit(file, '_')[[1]][1], strsplit(file, '_')[[1]][2], strsplit(file, '_')[[1]][3], file, stri_join_list(annot, '_'), NA))
}
m <- m[which(apply(is.na(m),1,sum) < ncol(m)),]
m <- noquote(m)
write.table(m, 'metadata.tsv', col.names=NA, quote=FALSE, sep='\t')
