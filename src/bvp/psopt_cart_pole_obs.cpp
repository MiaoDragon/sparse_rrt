/**
* Define the system propagate function in psopt format (adouble)
*/
#include "bvp/psopt_system.hpp"

#define _USE_MATH_DEFINES


#include <cmath>
#define I 10
#define L 2.5
#define M 10
#define m 5
#define g 9.8

#define STATE_X 0
#define STATE_V 1
#define STATE_THETA 2
#define STATE_W 3
#define CONTROL_A 0

#define MIN_X -30
#define MAX_X 30
#define MIN_V -40
#define MAX_V 40
#define MIN_W -2
#define MAX_W 2

void psopt_cart_pole_obs_t::dynamics(adouble* derivatives, adouble* path, adouble* states, adouble* controls, adouble* parameters,
                           adouble& time, adouble* xad, int iphase, Workspace* workspace)
{
    adouble _v = states[STATE_V];
    adouble _w = states[STATE_W];
    adouble _theta = states[STATE_THETA];
    adouble _a = controls[CONTROL_A];
    adouble mass_term = (M + m)*(I + m * L * L) - m * m * L * L * cos(_theta) * cos(_theta);

    derivatives[STATE_X] = _v;
    derivatives[STATE_THETA] = _w;
    mass_term = (1.0 / mass_term);
    derivatives[STATE_V] = ((I + m * L * L)*(_a + m * L * _w * _w * sin(_theta)) + m * m * L * L * cos(_theta) * sin(_theta) * g) * mass_term;
    derivatives[STATE_W] = ((-m * L * cos(_theta))*(_a + m * L * _w * _w * sin(_theta))+(M + m)*(-m * g * L * sin(_theta))) * mass_term;

}
