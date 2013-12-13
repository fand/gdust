//  nvcc -Iinclude -lcurand -lcutil -lgsl -lgslcblas t/distance.cu src/randomvariable.o src/timeseries.o src/timeseriescollection.o src/dust.o src/euclidean.o src/kernel.o src/gdust.o src/integrator.o

#include <iostream>
#include <cutil.h>
#include <math.h>
#include <vector>

#include "kernel.hpp"
#include "randomvariable.hpp"
#include "dust.hpp"
#include "gdust.hpp"
#include "timeseries.hpp"
#include "timeseriescollection.hpp"


int TestDistance(){

    const int N = 10;

    std::vector< RandomVariable > v1;
    std::vector< RandomVariable > v2;
        
    float *seq_H, *seq_D;
    seq_H = (float*)malloc(sizeof(float) * N * 6);
    for (int i=0; i<N; i++) {
        RandomVariable r1(RANDVAR_UNIFORM, 0, (i+20)/10.0, 0.123);        
        RandomVariable r2(RANDVAR_UNIFORM, 0, 1.0 - i/10.0, 0.123);

        seq_H[(i*6) + 0] = (float)r1.distribution;
        seq_H[(i*6) + 1] = (float)r1.observation;
        seq_H[(i*6) + 2] = (float)r1.stddev;
        seq_H[(i*6) + 3] = (float)r2.distribution;
        seq_H[(i*6) + 4] = (float)r2.observation;
        seq_H[(i*6) + 5] = (float)r2.stddev;

        v1.push_back(r1);
        v2.push_back(r2);
    }

    TimeSeries ts1(v1);
    TimeSeries ts2(v2);

    TimeSeriesCollection tsc;
        
    DUST dust(tsc);
    GDUST gdust(tsc);
    float cdustdist = dust.distance(ts1, ts2, -1);
    float gdustdist = gdust.distance(ts1, ts2, -1);

    std::cout << "cdust: " << cdustdist << std::endl;
    std::cout << "gdust: " << gdustdist << std::endl;
    
    return 0;
}
