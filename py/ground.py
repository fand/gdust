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
    output_raw = input + "_raw"
    output_error = [input + "_error_" + str(n) for n in range(1, 11)]

    print(len(output_error))
    
    f_in = open(input, 'r')
    f_out_raw = open(output_raw, 'w')
    f_out_error = []
    for e in output_error:
        f_out_error.append(open(e, 'w'))

    lines = f_in.readlines()
    for l in lines:
        seq = l.rstrip().split(" ")

        for data in seq:
            f_out_raw.write(data + ":" + data + ":" + '0' + " ")

            for j in range(1, 11):
                stddev = 0.2 * j
                error = nrand(data, stddev)
                f_out_error[j-1].write(data + ":" + str(error) + ":" + str(stddev) + " ")
                
        f_out_raw.write("\n")
        for e in f_out_error:
            e.write("\n")

    f_in.close()
    f_out_raw.close()
    for e in f_out_error:
        e.close()    
            
            
