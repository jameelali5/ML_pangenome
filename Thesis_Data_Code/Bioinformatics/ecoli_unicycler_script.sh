#! /usr/bin/env bash

for file_1 in ecoli_fq/*_1.fastq
do
file_2=${file_1/_1/_2}
out=${file_1%_1.fastq}_assembly
out=${out#ecoli_fq/}
echo $out
unicycler -1 $file_1 -2 $file_2 -o ./assembly_ecoli/$out -t 12 --mode conservative
done

echo completed script
