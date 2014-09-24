#!/bin/sh

echo "{"

temp=`nvidia-smi --query-gpu=temperature.gpu --id=1 --format=csv,noheader`
while [ $temp -gt 60 ]
do
    # echo "####### too hot! #######"
    sleep 60
    temp=`nvidia-smi --query-gpu=temperature.gpu --id=1 --format=csv,noheader`
done

bin/gdustdtw --exp 4 data/Gun_Point_error_3 data/Gun_Point_error_8

echo "}"
