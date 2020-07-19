/**
* Define the system propagate function in psopt format (adouble)
*/
#include "bvp/psopt_cart_pole_obs.hpp"

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

std::string psopt_cart_pole_obs_t::get_name() const
{
    return "cartpole_obs";
}
double psopt_cart_pole_obs_t::max_distance() const
{
    return sqrt((MAX_X-MIN_X)*(MAX_X-MIN_X)+(MAX_V-MIN_V)*(MAX_V-MIN_V)+(MAX_W-MIN_W)*(MAX_W-MIN_W)+M_PI*M_PI);
}

bool psopt_cart_pole_obs_t::propagate(
    const double* start_state, unsigned int state_dimension,
    const double* control, unsigned int control_dimension,
    int num_steps, double* result_state, double integration_step)
{
        temp_state[0] = start_state[0];
        temp_state[1] = start_state[1];
        temp_state[2] = start_state[2];
        temp_state[3] = start_state[3];
        bool validity = true;
        for(int i=0;i<num_steps;i++)
        {
                update_derivative(control);
                temp_state[0] += integration_step*deriv[0];
                temp_state[1] += integration_step*deriv[1];
                temp_state[2] += integration_step*deriv[2];
                temp_state[3] += integration_step*deriv[3];
                enforce_bounds();
                validity = validity && valid_state();
        }
        result_state[0] = temp_state[0];
        result_state[1] = temp_state[1];
        result_state[2] = temp_state[2];
        result_state[3] = temp_state[3];
        return validity;
}

void psopt_cart_pole_obs_t::enforce_bounds()
{
        //if(temp_state[0]<MIN_X)
        //        temp_state[0]=MIN_X;
        //else if(temp_state[0]>MAX_X)
        //        temp_state[0]=MAX_X;

        if(temp_state[1]<MIN_V)
                temp_state[1]=MIN_V;
        else if(temp_state[1]>MAX_V)
                temp_state[1]=MAX_V;

        if(temp_state[2]<-M_PI)
                temp_state[2]+=2*M_PI;
        else if(temp_state[2]>M_PI)
                temp_state[2]-=2*M_PI;

        if(temp_state[3]<MIN_W)
                temp_state[3]=MIN_W;
        else if(temp_state[3]>MAX_W)
                temp_state[3]=MAX_W;
}


bool psopt_cart_pole_obs_t::valid_state()
{
    // check the pole with the rectangle to see if in collision
    // calculate the pole state
    // check if the position is within bound
    if (temp_state[0] < MIN_X or temp_state[0] > MAX_X)
    {
        return false;
    }
    double pole_x1 = temp_state[0];
    double pole_y1 = H;
    double pole_x2 = temp_state[0] + L * sin(temp_state[2]);
    double pole_y2 = H + L * cos(temp_state[2]);
    //std::cout << "state:" << temp_state[0] << "\n";
    //std::cout << "pole point 1: " << "(" << pole_x1 << ", " << pole_y1 << ")\n";
    //std::cout << "pole point 2: " << "(" << pole_x2 << ", " << pole_y2 << ")\n";
    for(unsigned int i = 0; i < obs_list.size(); i++)
    {
        // check if any obstacle has intersection with pole
        //std::cout << "obstacle " << i << "\n";
        //std::cout << "points: \n";
        for (unsigned int j = 0; j < 8; j+=2)
        {

            //std::cout << j << "-th point: " << "(" << obs_list[i][j] << ", " << obs_list[i][j+1] << ")\n";
        }
        for (unsigned int j = 0; j < 8; j+=2)
        {
            // check each line of the obstacle
            double x1 = obs_list[i][j];
            double y1 = obs_list[i][j+1];
            double x2 = obs_list[i][(j+2) % 8];
            double y2 = obs_list[i][(j+3) % 8];
            if (lineLine(pole_x1, pole_y1, pole_x2, pole_y2, x1, y1, x2, y2))
            {
                // intersect
                return false;
            }
        }
    }
    return true;
}


std::tuple<double, double> psopt_cart_pole_obs_t::visualize_point(const double* state, unsigned int state_dimension) const
{
    double x = state[STATE_X] + (L / 2.0) * sin(state[STATE_THETA]);
    double y = -(L / 2.0) * cos(state[STATE_THETA]);

    x = (x-MIN_X)/(MAX_X-MIN_X);
    y = (y-MIN_X)/(MAX_X-MIN_X);
    return std::make_tuple(x, y);
}

void psopt_cart_pole_obs_t::update_derivative(const double* control)
{
    double _v = temp_state[STATE_V];
    double _w = temp_state[STATE_W];
    double _theta = temp_state[STATE_THETA];
    double _a = control[CONTROL_A];
    double mass_term = (M + m)*(I + m * L * L) - m * m * L * L * cos(_theta) * cos(_theta);

    deriv[STATE_X] = _v;
    deriv[STATE_THETA] = _w;
    mass_term = (1.0 / mass_term);
    deriv[STATE_V] = ((I + m * L * L)*(_a + m * L * _w * _w * sin(_theta)) + m * m * L * L * cos(_theta) * sin(_theta) * g) * mass_term;
    deriv[STATE_W] = ((-m * L * cos(_theta))*(_a + m * L * _w * _w * sin(_theta))+(M + m)*(-m * g * L * sin(_theta))) * mass_term;
}


std::vector<std::pair<double, double> > psopt_cart_pole_obs_t::get_state_bounds() const {
    return {
            {MIN_X,MAX_X},
            {MIN_V,MAX_V},
            //{-M_PI,M_PI},
            {-inf,inf},
            {MIN_W,MAX_W},
    };
}


std::vector<std::pair<double, double> > psopt_cart_pole_obs_t::get_control_bounds() const {
    return {
            {-300,300},
    };
}


std::vector<bool> psopt_cart_pole_obs_t::is_circular_topology() const {
    return {
            false,
            false,
            true,
            false
    };
}



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

adouble psopt_cart_pole_obs_t::endpoint_cost(adouble* initial_states, adouble* final_states, adouble* parameters, adouble& t0,
                                 adouble& tf, adouble* xad, int iphase, Workspace* workspace)
{
    // Since we already set endpoint constraint in events, we don't need it here
    // TODO: maybe we can set one end free, but try to reduce the cost only
    // Here we use the time as endpoint cost for minimum time control
   // Since we already set endpoint constraint in events, we don't need it here
    // TODO: maybe we can set one end free, but try to reduce the cost only
    // Here we use the time as endpoint cost for minimum time control
    //return 0.1*tf;
    //return 0.;
    double* goal = (double*) workspace->problem->user_data;
    //std::cout << "goal extracteed" << std::endl;
    adouble sum_of_square = 0;
    // wrap the angle to [-pi, pi]
    // if dx > pi, then dx = dx - 2pi
    // if dx < -pi, then dx = dx + 2pi
    for (unsigned i=0; i<4; i++)
    {
        // wrap the angle to [-pi, pi]
        // if dx > pi, then dx = dx - 2pi
        // if dx < -pi, then dx = dx + 2pi
        adouble dx = final_state[i] - goal[i];
        sum_of_square = sum_of_square + dx*dx;
    }
    //std::cout << "after sum_of_square" << std::endl;
    return sum_of_square;
}

adouble psopt_cart_pole_obs_t::integrand_cost(adouble* states, adouble* controls, adouble* parameters, adouble& time, adouble* xad,
                      int iphase, Workspace* workspace)
{
    adouble retval = 0.0;
    return retval;
}

void psopt_cart_pole_obs_t::events(adouble* e, adouble* initial_states, adouble* final_states, adouble* parameters, adouble& t0,
            adouble& tf, adouble* xad, int iphase, Workspace* workspace)
{
  for (unsigned i=0; i < STATE_N; i++)
  {
      e[i] = initial_states[i];
      //e[STATE_N+i] = final_states[i];
  }
}

void psopt_cart_pole_obs_t::linkages(adouble* linkages, adouble* xad, Workspace* workspace)
{
  // No linkages in this single phase problem
}
