#! /usr/bin/env bash

mkdir ecoli_annotation/copy_gff

for file in ecoli_annotation/*_annotation
do
echo $file
out=${file#'ecoli_annotation/'}
out=${out%'annotation'}
echo $out
cp $file/*.gff ./ecoli_annotation/copy_gff/$out'copy.gff'
done

roary -f ecoli_pangenome -e -n -i 95 -v -p 12 ecoli_annotation/copy_gff/*.gff

echo script completed

