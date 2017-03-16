#!/bin/bash
class=0

# removes already existent files
rm train_map.txt
rm test_map.txt

for dir in $1
do	
	nfiles=`ls -l $dir | wc -l`
	
	toTrain=$((nfiles*2/3))
	counter=0
	for file in $dir/*
	do
		if [ $counter -le $toTrain ]
		then
			echo $file $class >> train_map.txt
		else
			echo $file $class >> test_map.txt
		fi	
		counter=$((counter+1))
	done
	class=$((class+1))
done
