/**
* Define the system propagate function in psopt format (adouble)
*/
#include "bvp/psopt_system.hpp"

#define _USE_MATH_DEFINES
#define MIN_W -7
#define MAX_W 7

#define MIN_TORQUE -1
#define MAX_TORQUE 1

#define LENGTH 1
#define MASS 1
#define DAMPING .05


#include <cmath>
void psopt_pendulum_t::dynamics(adouble* derivatives, adouble* path, adouble* states, adouble* controls, adouble* parameters,
                           adouble& time, adouble* xad, int iphase, Workspace* workspace)
{
    adouble temp0 = states[0];
    adouble temp1 = states[1];
    derivatives[0] = temp1;
    derivatives[1] = ((control[0] - MASS * (9.81) * LENGTH * cos(temp0)*0.5 - DAMPING * temp1)* 3 / (MASS * LENGTH * LENGTH));
}
