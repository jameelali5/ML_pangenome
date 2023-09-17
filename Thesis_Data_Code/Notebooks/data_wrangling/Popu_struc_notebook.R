#Set working directories
getwd()
setwd('/Users/jameelali/Desktop/Machine_Learning_Project/Ecoli_Project/Data')

#Generates population strcuture matrix from core genome alignement
#Specify names and paths for alignment file and output file 

library(readr)
library(ape)
library(adegenet)

#path="core_gene_alignment_400_ecoli.aln"
path='PA_core_gene_alignment.aln'
dna<-read.dna(path,format = "fasta")
print(dna)
distdna<-dist.dna(dna,model ="N",pairwise.deletion = TRUE,as.matrix = TRUE)

distt<-c()
tmp_mat= data.frame(matrix(data=NA,nrow=dim(distdna)[1], ncol = length(seq(2,max(distdna),1))))
count=1
for(i in seq(2,max(distdna),1)){
  print(i)
  clust<-gengraph(distdna, plot=FALSE,cutoff=i)
  distt<-c(distt,clust[2][[1]][[3]])
  tmp_mat[,count]=paste0("Cl",as.character(clust$clust$membership))
  count=count+1
}
row.names(tmp_mat)=colnames(distdna)
tmp_mat_dedup<-tmp_mat[,colnames(unique(as.matrix(tmp_mat), MARGIN=2))]
#write.csv(tmp_mat_dedup, "population_structure_400_ecoli.csv", quote = FALSE)
write.csv(tmp_mat_dedup, "PA_population_structure.csv", quote = FALSE)
return(tmp_mat_dedup)