#! /usr/bin/env bash

for file in assembly_ecoli/*_assembly
do
echo $file
out=${file#'assembly_ecoli/'}
out=${out%'_assembly'}_annotation
echo $out
prokka --outdir ecoli_annotation/$out --addgenes --metagenome --cpus 12 $file/assembly.fasta
done

