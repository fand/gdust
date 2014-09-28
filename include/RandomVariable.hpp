#pragma once

#define RANDVAR_UNIFORM 1
#define RANDVAR_NORMAL 2
#define RANDVAR_EXP 3

class RandomVariable
{
public:
    RandomVariable( int distribution, float groundtruth, float observation, float stddev );
    int distribution;
    float stddev;
    float observation;
    float groundtruth;

    operator float()
    {
        return static_cast<float>(observation);
    }
    operator double()
    {
        return static_cast<double>(observation);
    }
    operator int()
    {
        return static_cast<int>(observation);
    }

    RandomVariable & operator= ( const float observation )
    {
        this->observation = observation;
        return *this;
    }
};
