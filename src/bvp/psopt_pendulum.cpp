/**
* Define the system propagate function in psopt format (adouble)
*/
#include "bvp/psopt_pendulum.hpp"

#define _USE_MATH_DEFINES
#define MIN_W -7
#define MAX_W 7

#define MIN_TORQUE -1
#define MAX_TORQUE 1

#define LENGTH 1
#define MASS 1
#define DAMPING .05
#define STATE_N 2

#include <cmath>
#include "utilities/random.hpp"

std::string psopt_pendulum_t::get_name() const
{
    return "pendulum";
}
double psopt_pendulum_t::max_distance() const
{
    return sqrt(M_PI*M_PI+(MAX_W-MIN_W)*(MAX_W-MIN_W));
}
bool psopt_pendulum_t::propagate(
    const double* start_state, unsigned int state_dimension,
    const double* control, unsigned int control_dimension,
    int num_steps, double* result_state, double integration_step)
{
	temp_state[0] = start_state[0]; temp_state[1] = start_state[1];
	bool validity = true;
	for(int i=0;i<num_steps;i++)
	{
		double temp0 = temp_state[0];
		double temp1 = temp_state[1];
		temp_state[0] += integration_step*temp1;
		temp_state[1] += integration_step*
							((control[0] - MASS * (9.81) * LENGTH * cos(temp0)*0.5
										 - DAMPING * temp1)* 3 / (MASS * LENGTH * LENGTH));
		enforce_bounds();
		validity = validity && valid_state();
	}
	result_state[0] = temp_state[0];
	result_state[1] = temp_state[1];
	return validity;
}

void psopt_pendulum_t::enforce_bounds()
{
	if(temp_state[0]<-M_PI)
		temp_state[0]+=2*M_PI;
	else if(temp_state[0]>M_PI)
		temp_state[0]-=2*M_PI;

	if(temp_state[1]<MIN_W)
		temp_state[1]=MIN_W;
	else if(temp_state[1]>MAX_W)
		temp_state[1]=MAX_W;
}


bool psopt_pendulum_t::valid_state()
{
	return true;
}

std::tuple<double, double> psopt_pendulum_t::visualize_point(const double* state, unsigned int state_dimension) const
{
	double x = (state[0]+M_PI)/(2*M_PI);
	double y = (state[1]-MIN_W)/(MAX_W-MIN_W);
	return std::make_tuple(x, y);
}

std::vector<std::pair<double, double> > psopt_pendulum_t::get_state_bounds() const {
    // here we increase the bound to make sure that the state can wrap around
	return {
			//{M_PI,M_PI},
            [-inf, inf},  // we don't add constraint here to angle
			{MIN_W,MAX_W},
	};
}


std::vector<std::pair<double, double> > psopt_pendulum_t::get_control_bounds() const {
	return {
			{MIN_TORQUE,MAX_TORQUE},
	};
}

std::vector<bool> psopt_pendulum_t::is_circular_topology() const {
	return {
            true,
			false
	};
}

void psopt_pendulum_t::dynamics(adouble* derivatives, adouble* path, adouble* states, adouble* controls, adouble* parameters,
                           adouble& time, adouble* xad, int iphase, Workspace* workspace)
{
    adouble temp0 = states[0];
    adouble temp1 = states[1];
    derivatives[0] = temp1;
    derivatives[1] = ((controls[0] - MASS * (9.81) * LENGTH * cos(temp0)*0.5 - DAMPING * temp1)* 3 / (MASS * LENGTH * LENGTH));
}



adouble psopt_pendulum_t::endpoint_cost(adouble* initial_states, adouble* final_states, adouble* parameters, adouble& t0,
                                 adouble& tf, adouble* xad, int iphase, Workspace* workspace)
{
    // Since we already set endpoint constraint in events, we don't need it here
    // TODO: maybe we can set one end free, but try to reduce the cost only
    // Here we use the time as endpoint cost for minimum time control
    //return tf;
    return 0.;
}

adouble psopt_pendulum_t::integrand_cost(adouble* states, adouble* controls, adouble* parameters, adouble& time, adouble* xad,
                      int iphase, Workspace* workspace)
{
    //adouble retval = 0.0;
    // here we try minimizing the state trajectory length instead of the time
    // thus need to calculate the derivatives
    adouble derivatives[2];
    adouble temp0 = states[0];
    adouble temp1 = states[1];
    derivatives[0] = temp1;
    derivatives[1] = ((controls[0] - MASS * (9.81) * LENGTH * cos(temp0)*0.5 - DAMPING * temp1)* 3 / (MASS * LENGTH * LENGTH));
    return sqrt(derivatives[0]*derivatives[0]+derivatives[1]*derivatives[1]);


    //return retval;
}

void psopt_pendulum_t::events(adouble* e, adouble* initial_states, adouble* final_states, adouble* parameters, adouble& t0,
            adouble& tf, adouble* xad, int iphase, Workspace* workspace)
{
  for (unsigned i=0; i < STATE_N; i++)
  {
      e[i] = initial_states[i];
      e[STATE_N+i] = final_states[i];
  }
}

void psopt_pendulum_t::linkages(adouble* linkages, adouble* xad, Workspace* workspace)
{
  // No linkages in this single phase problem
}
