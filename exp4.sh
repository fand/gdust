#!/bin/sh

echo "{"

for (( i=1; i<10; i++ ))
do
    for (( j=2; j<11; j++ ))
    do
        echo "\"${i}-${j}\" : {"

        temp=`nvidia-smi --query-gpu=temperature.gpu --id=1 --format=csv,noheader`
        while [ $temp -gt 60 ]
        do
            # echo "####### too hot! #######"
            sleep 60
            temp=`nvidia-smi --query-gpu=temperature.gpu --id=1 --format=csv,noheader`
        done

        bin/gdustdtw --exp 4 data/Gun_Point_error_${i} data/Gun_Point_error_${j}

        echo "},"
    done
done

echo "}"
