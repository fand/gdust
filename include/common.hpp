#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <sstream>


template< class T >
void SAFE_FREE( T x )
{
    if( x )
    {
        free( x );
        x = NULL;
    }
}


template< class T >
void FATAL( T x )
{
    std::cerr << "Fatal error at " << __FILE__ << ":" << __FUNCTION__ <<":" << __LINE__ <<": "<< x << "; exit forced.\n";
    exit(1);
}


template< class T >
T MAX( T x, T y )
{
    return (x)>(y) ? (x) : (y);
}


template< class T >
T MIN( T x, T y )
{
    return (x)<(y) ? (x) : (y);
}


template< class T >
T ABS( T x )
{
    return (x) >= 0 ? (x) : (-(x));
}


template < class T >
bool NOTNANINF( T x )
{
    bool isn = isnan(x);
    bool isi = isinf(x);
    return !(isn || isi);
}




template< class T >
bool POSITIVE0( T x )
{
    return (x) == -0 ? 0 : (x);
}


template< class T >
std::string TO_STR( T x )
{
    std::ostringstream ss;
    ss << x;
    return ss.str();
}


#define uint unsigned int

