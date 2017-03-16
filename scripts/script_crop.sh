#!/bin/bash

# assumes dataset/1/train/  OR  dataset/1/test/  OR  dataset/2/
cd $1

for file in */*
do
    echo "Cropping file:" $file 
    #python ../../../scripts/random_crop.py $file
    python ../../../scripts/crop.py $file
done