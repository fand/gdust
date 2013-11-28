#include <time.h>
#include <sys/time.h>

#include "watch.hpp"

Watch::Watch()
{
    gettimeofday( &(this->t_start), NULL );
    gettimeofday( &(this->t_stop), NULL );    
}


void Watch::start()
{
    gettimeofday( &(this->t_start), NULL );
}


void Watch::stop()
{
    gettimeofday( &(this->t_stop), NULL );
}


double Watch::getInterval()
{
    double a, b;
    a = this->t_start.tv_sec + this->t_start.tv_usec * 1e-6;
    b = this->t_stop.tv_sec + this->t_stop.tv_usec * 1e-6;

    return ( b - a );
}
