#!/bin/bash
class=0

# removes already existent files
(rm train_map.txt && rm test_map.txt && rm train_test_map.txt) || continue

for dir in $1/*
do	
	echo "[INFO] Parsing directory" $dir 
	
	nfiles=`ls -l $dir | wc -l`
	
	echo "[INFO] Directory" $dir "has" $nfiles "files"
	
	toTrain=$((nfiles*2/3))
	toTest=$((nfiles-toTrain))
	counter=0

	echo "[INFO] Training images" $toTrain
	echo "[INFO]  Testing images" $toTest "\n"

	for file in $dir/*
	do
		filepath=`pwd $file`
		if [ $counter -le $toTrain ]
		then
			echo $file $class >> train_map.txt
		else
			echo $file $class >> test_map.txt
		fi

		echo $file $class >> train_test_map.txt 

		counter=$((counter+1))
	done
	class=$((class+1))
done

echo "[INFO] All done"
