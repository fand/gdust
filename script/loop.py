# -*- coding:utf-8 -*-
import sys
import os
import glob
from numpy.random import normal as nrand

if __name__ == '__main__':

    argv = sys.argv
    if len(argv) != 2:
        print("argv arror!")
        sys.exit()

    input = argv[1]
    f_in = open(input, 'r')

    f_out = []
    for i in range(1, 11):
        f_out.append(open(input + '_trunk_' + str(i * 50), 'w'))

    lines = f_in.readlines()

    p0 = 0.0
    p1 = 0.0  
    for l in lines:
        seq = l.rstrip().split(" ")

        for d in seq[:50]:
            v = d.split(":")
            if len(v) != 3:
                print("data format error!")
                exit()

            for i in range(len(f_out)):
                for j in range(1, i+2):
                    d0 = str(float((float(v[0]) - p0) * j) / (i+1) + p0)
                    d1 = str(float((float(v[1]) - p1) * j) / (i+1) + p1)
                    f_out[i].write(d0 + ':' + d1 + ':'  + v[2] + ' ')
                    
            p0 = float(v[0])
            p1 = float(v[1]) 

        p0 = 0.0
        p1 = 0.0

        for f in f_out:
            f.write("\n")

            
    for e in f_out:
        e.close()                
            
