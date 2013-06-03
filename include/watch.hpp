#include <time.h>
#include <sys/time.h>

class Watch
{
private:
    struct timeval t_start, t_stop;
    
public:
    Watch();
    void start();
    void stop();
    double getInterval();
};
