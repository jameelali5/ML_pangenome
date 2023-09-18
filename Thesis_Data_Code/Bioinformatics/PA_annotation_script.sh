#! /usr/bin/env bash

for file in PA_assembly/*_assembly
do
echo $file
out=${file#'PA_assembly/'}
out=${out%'_assembly'}_annotation
echo $out
prokka --outdir PA_annotation/$out --addgenes --metagenome --proteins Pseudomonas_aeruginosa_UCBPP-PA14_109.gbk --cpus 12 $file/assembly.fasta
done

