#!/bin/bash

class=0

# remove files if they already exists
rm cntk_validation.txt
rm cntk_train.txt
rm cntk_test.txt

# assumes cropped/1.1/ OR dataset/1/test/
for dir in $1*
# $1 = path to datasets
do
    nfiles=`ls $dir | wc -l`
	toVal=$((nfiles*1/10))
	counter=0
    
	echo "Directory" $dir "has" $nfiles "files"
	if [ "$2" != "test" ]
	then
		echo "   From those" $nfiles "," $toVal "of them go for validation"
	fi

    for file in $dir/*
    do
		echo "\tGenerating CNTK file format for:" $file
		if [ "$2" != "test" ]
		then
			if [ $counter -lt $toVal ]
			then
				python scripts/img2txt.py $class $file >> cntk_validation.txt
			else
				python scripts/img2txt.py $class $file >> cntk_train.txt
			fi	
			counter=$((counter+1))
		else
			python scripts/img2txt.py $class $file >> cntk_test.txt
		fi 
    done 
    
    class=$((class+1))
done