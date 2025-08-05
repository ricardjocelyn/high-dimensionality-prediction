#! /bin/bash

inDir="/home/vanessa/Documents/OAS2/OAS2_RAW_PART1"

cd ${inDir}

for i in ls -d -- */; do #For each directory in current directory
    if [[ $i == *"MR"[3-9]* ]]; then #If directory contains MR3-MR9, then
        rm -r $i/* #Remove directory contents
        rmdir $i #Remove directory itself
    fi
done
#For each subject name
    #add to the true subject array

#For eachsubject of the array
    #CP to the
