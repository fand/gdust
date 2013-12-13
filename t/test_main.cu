#include <cutil.h>
#include <stdio.h>
#include <assert.h>
#define NEAR(x,y) (assert( abs((x) - (y)) < 0.1 ))



//////////////////////////////
// Test files.
//////////////////////////////

#include "test_kernel.hpp"
#include "test_distance.hpp"



int main() {
    TestKernel();
    TestDistance();

    printf("All Test passed! :D\n");
    
    return 0;
}
