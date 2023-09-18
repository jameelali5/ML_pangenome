#! /usr/bin/env bash

mkdir PA_annotation/copy_gff

for file in PA_annotation/*_annotation
do
echo $file
out=${file#'PA_annotation/'}
out=${out%'_annotation'}
echo $out
cp $file/*.gff ./PA_annotation/copy_gff/$out'copy.gff'
done

roary -f PA_pangenome -e -n -i 95 -v -p 12 PA_annotation/copy_gff/*.gff



