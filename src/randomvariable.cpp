#include "randomvariable.hpp"

RandomVariable::RandomVariable( int distribution, float groundtruth, float observation, float stddev )
{
    this->distribution = distribution;
    this->groundtruth = groundtruth;
    this->observation = observation;
    this->stddev = stddev;
}
