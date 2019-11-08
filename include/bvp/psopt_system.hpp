/**
* Define the system propagate function in psopt format (adouble)
*/

#ifndef PSOPT_SYSTEM_HPP
#define PSOPT_SYSTEM_HPP

#include "psopt.h"
#include "systems/system.hpp"
#include "systems/car.hpp"
#include "systems/cart_pole_obs.hpp"
#include "systems/cart_pole.hpp"
#include "systems/pendulum.hpp"
#include "systems/point.hpp"
#include "systems/rally_car.hpp"
#include "systems/two_link_acrobot.hpp"

class psopt_system_t : public system_t
{
    psopt_system_t();
    virtual void dynamics(adouble* derivatives, adouble* path, adouble* states, adouble* controls, adouble* parameters,
        adouble& time, adouble* xad, int iphase, Workspace* workspace) = 0;
};

#endif