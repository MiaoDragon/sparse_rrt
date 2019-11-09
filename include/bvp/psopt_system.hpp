/**
* Define the system propagate function in psopt format (adouble)
*/

#ifndef PSOPT_SYSTEM_HPP
#define PSOPT_SYSTEM_HPP

#ifndef PSOPT_H
#define PSOPT_H
#include "psopt.h"
#endif


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
public:
    psopt_system_t();
};

#endif
