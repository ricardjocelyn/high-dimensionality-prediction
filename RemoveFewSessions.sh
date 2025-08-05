#! /bin/bash

inDir="/home/jovyan/shared/data/OASIS-2/OAS2_RAW_PART1/"
particFile="/home/jovyan/GitRepos/high-dimensionality-prediction/particToProcess.txt"
touch ${particFile}

cd ${inDir}

for i in ls -d -- */; do #For each directory in current directory
    if [[ $i == *"MR"[3]* ]]; then #If directory contains MR3-MR9, then
        #rm -r $i/* #Remove directory contents
        #rmdir $i #Remove directory itself
        session3=$i
        session1=${session3::-2}
        session1=${session1}1/

        session3=${inDir}${session3}
        session1=${inDir}${session1}
        echo $session1 >> $particFile
        echo $session3 >> $particFile
        echo "" >> $particFile

        #Open the txt file, pipe in the dir and then the folders
    fi
done

