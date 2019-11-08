/**
* Define the system propagate function in psopt format (adouble)
*/
#include "bvp/psopt_system.hpp"

#define _USE_MATH_DEFINES


#include <cmath>
void psopt_point_t::dynamics(adouble* derivatives, adouble* path, adouble* states, adouble* controls, adouble* parameters,
                           adouble& time, adouble* xad, int iphase, Workspace* workspace)
{
    derivatives[0] = controls[0]*cos(controls[1]);
    derivatives[1] = controls[0]*sin(controls[1]);
}
