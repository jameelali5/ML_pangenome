#! /usr/bin/env bash

for file_1 in PA_fastq/*_1.fastq
do
file_2=${file_1/_1/_2}
out=${file_1%_1.fastq}_assembly
out=${out#PA_fastq/}
echo $out
unicycler -1 $file_1 -2 $file_2 -o ./PA_assembly/$out -t 12 --mode conservative
done

echo completed script
