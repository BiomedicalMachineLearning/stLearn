genes <- read.csv('genes',FALSE)
effects <- read.table('effects',FALSE,sep=' ')
remove <- which(duplicated(genes[,1]))
if (sum(remove) != 0) {
    genes <- genes[-remove,]
    effects <- effects[-remove,]
}
effects[,5] <- effects[,2] / (effects[,1] + effects[,2] + effects[,3] + effects[,4])
rownames(effects) <- genes
colnames(effects) <- c('intrinsic', 'interactions', 'environmental', 'noise','cc')
write.table(effects[order(-effects$cc),], 'sorted', sep=',', quote=FALSE)
