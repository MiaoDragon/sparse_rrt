/**
* Define the system propagate function in psopt format (adouble)
*/
#include "bvp/psopt_acrobot_obs.hpp"

#define _USE_MATH_DEFINES

#include <cmath>

#define LENGTH 20.0
#define m 1.0

#define lc  .5
#define lc2  .25
#define l2  1
#define I1  0.2
#define I2  1.0
#define l  1.0
#define g  9.8


#define STATE_N 4
#define STATE_THETA_1 0
#define STATE_THETA_2 1
#define STATE_V_1 2
#define STATE_V_2 3
#define CONTROL_T 0

#define MIN_V_1 -6
#define MAX_V_1 6
#define MIN_V_2 -6
#define MAX_V_2 6
#define MIN_T -4
#define MAX_T 4

std::string psopt_acrobot_obs_t::get_name() const
{
    return "acrobot";
}
double psopt_acrobot_obs_t::distance(const double* point1, const double* point2, unsigned int state_dimension)
{
        double x = (LENGTH) * cos(point1[STATE_THETA_1] - M_PI / 2)+(LENGTH) * cos(point1[STATE_THETA_1] + point1[STATE_THETA_2] - M_PI / 2);
        double y = (LENGTH) * sin(point1[STATE_THETA_1] - M_PI / 2)+(LENGTH) * sin(point1[STATE_THETA_1] + point1[STATE_THETA_2] - M_PI / 2);
        double x2 = (LENGTH) * cos(point2[STATE_THETA_1] - M_PI / 2)+(LENGTH) * cos(point2[STATE_THETA_1] + point2[STATE_THETA_2] - M_PI / 2);
        double y2 = (LENGTH) * sin(point2[STATE_THETA_1] - M_PI / 2)+(LENGTH) * sin(point2[STATE_THETA_1] + point2[STATE_THETA_2] - M_PI / 2);
        return std::sqrt(pow(x-x2,2.0)+pow(y-y2,2.0));
}

double psopt_acrobot_obs_t::max_distance() const
{
    return sqrt((MAX_V_1-MIN_V_1)*(MAX_V_1-MIN_V_1)+(MAX_V_2-MIN_V_2)*(MAX_V_2-MIN_V_2)+(MAX_T-MIN_T)*(MAX_T-MIN_T)+M_PI*M_PI);
}
bool psopt_acrobot_obs_t::propagate(
    const double* start_state, unsigned int state_dimension,
    const double* control, unsigned int control_dimension,
    int num_steps, double* result_state, double integration_step)
{
        temp_state[0] = start_state[0];
        temp_state[1] = start_state[1];
        temp_state[2] = start_state[2];
        temp_state[3] = start_state[3];
        bool validity = false;
        // find the last valid position, if no valid position is found, then return false
        for(int i=0;i<num_steps;i++)
        {
                update_derivative(control);
                temp_state[0] += integration_step*deriv[0];
                temp_state[1] += integration_step*deriv[1];
                temp_state[2] += integration_step*deriv[2];
                temp_state[3] += integration_step*deriv[3];
                enforce_bounds();
                //validity = validity && valid_state();
                if (valid_state() == true)
                {
                    result_state[0] = temp_state[0];
                    result_state[1] = temp_state[1];
                    result_state[2] = temp_state[2];
                    result_state[3] = temp_state[3];
                    validity = true;
                }
                else
                {
                    // Found the earliest invalid position. break the loop and return
                    validity = false; // need to update validity because one node is invalid, the propagation fails
                    break;
                }
        }
        //result_state[0] = temp_state[0];
        //result_state[1] = temp_state[1];
        //result_state[2] = temp_state[2];
        //result_state[3] = temp_state[3];
        return validity;
}

void psopt_acrobot_obs_t::enforce_bounds()
{
        if(temp_state[0]<-M_PI)
                temp_state[0]+=2*M_PI;
        else if(temp_state[0]>M_PI)
                temp_state[0]-=2*M_PI;
        if(temp_state[1]<-M_PI)
                temp_state[1]+=2*M_PI;
        else if(temp_state[1]>M_PI)
                temp_state[1]-=2*M_PI;
        if(temp_state[2]<MIN_V_1)
                temp_state[2]=MIN_V_1;
        else if(temp_state[2]>MAX_V_1)
                temp_state[2]=MAX_V_1;
        if(temp_state[3]<MIN_V_2)
                temp_state[3]=MIN_V_2;
        else if(temp_state[3]>MAX_V_2)
                temp_state[3]=MAX_V_2;
}


bool psopt_acrobot_obs_t::valid_state()
{
    // check the pole with the rectangle to see if in collision
    // calculate the pole state
    double pole_x0 = 0.;
    double pole_y0 = 0.;
    double pole_x1 = (LENGTH) * cos(temp_state[STATE_THETA_1] - M_PI / 2);
    double pole_y1 = (LENGTH) * sin(temp_state[STATE_THETA_1] - M_PI / 2);
    double pole_x2 = pole_x1 + (LENGTH) * cos(temp_state[STATE_THETA_1] + temp_state[STATE_THETA_2] - M_PI / 2);
    double pole_y2 = pole_y1 + (LENGTH) * sin(temp_state[STATE_THETA_1] + temp_state[STATE_THETA_2] - M_PI / 2);

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
            if (lineLine(pole_x0, pole_y0, pole_x1, pole_y1, x1, y1, x2, y2))
            {
                // intersect
                return false;
            }
            if (lineLine(pole_x1, pole_y1, pole_x2, pole_y2, x1, y1, x2, y2))
            {
                // intersect
                return false;
            }
        }
    }
    return true;
}

std::tuple<double, double> psopt_acrobot_obs_t::visualize_point(const double* state, unsigned int state_dimension) const
{
    double x = (LENGTH) * cos(state[STATE_THETA_1] - M_PI / 2)+(LENGTH) * cos(state[STATE_THETA_1] + state[STATE_THETA_2] - M_PI / 2);
    double y = (LENGTH) * sin(state[STATE_THETA_1] - M_PI / 2)+(LENGTH) * sin(state[STATE_THETA_1] + state[STATE_THETA_2] - M_PI / 2);
    x = (x+2*LENGTH)/(4*LENGTH);
    y = (y+2*LENGTH)/(4*LENGTH);
    return std::make_tuple(x, y);
}

void psopt_acrobot_obs_t::update_derivative(const double* control)
{
    double theta2 = temp_state[STATE_THETA_2];
    double theta1 = temp_state[STATE_THETA_1] - M_PI / 2;
    double theta1dot = temp_state[STATE_V_1];
    double theta2dot = temp_state[STATE_V_2];
    double _tau = control[CONTROL_T];

    //extra term m*lc2
    double d11 = m * lc2 + m * (l2 + lc2 + 2 * l * lc * cos(theta2)) + I1 + I2;

    double d22 = m * lc2 + I2;
    double d12 = m * (lc2 + l * lc * cos(theta2)) + I2;
    double d21 = d12;

    //extra theta1dot
    double c1 = -m * l * lc * theta2dot * theta2dot * sin(theta2) - (2 * m * l * lc * theta1dot * theta2dot * sin(theta2));
    double c2 = m * l * lc * theta1dot * theta1dot * sin(theta2);
    double g1 = (m * lc + m * l) * g * cos(theta1) + (m * lc * g * cos(theta1 + theta2));
    double g2 = m * lc * g * cos(theta1 + theta2);

    deriv[STATE_THETA_1] = theta1dot;
    deriv[STATE_THETA_2] = theta2dot;

    double u2 = _tau - 1 * .1 * theta2dot;
    double u1 = -1 * .1 * theta1dot;
    double theta1dot_dot = (d22 * (u1 - c1 - g1) - d12 * (u2 - c2 - g2)) / (d11 * d22 - d12 * d21);
    double theta2dot_dot = (d11 * (u2 - c2 - g2) - d21 * (u1 - c1 - g1)) / (d11 * d22 - d12 * d21);

    deriv[STATE_V_1] = theta1dot_dot;
    deriv[STATE_V_2] = theta2dot_dot;
}


std::vector<std::pair<double, double> > psopt_acrobot_obs_t::get_state_bounds() const {
    return
    {
        //{-M_PI,M_PI},
        {-inf,inf},
        //{-M_PI,M_PI},
        {-inf,inf},
        {MIN_V_1,MAX_V_1},
        {MIN_V_2,MAX_V_2},
    };
}


std::vector<std::pair<double, double> > psopt_acrobot_obs_t::get_control_bounds() const {
    return {
            {MIN_T,MAX_T}
    };
}


std::vector<bool> psopt_acrobot_obs_t::is_circular_topology() const {
    return {
            true,
            true,
            false,
            false
    };
}


bool psopt_acrobot_obs_t::lineLine(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4)
// compute whether two lines intersect with each other
{
    // ref: http://www.jeffreythompson.org/collision-detection/line-rect.php
    // calculate the direction of the lines
    double uA = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1));
    double uB = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1));

    // if uA and uB are between 0-1, lines are colliding
    if (uA >= 0 && uA <= 1 && uB >= 0 && uB <= 1)
    {
        // intersect
        return true;
    }
    // not intersect
    return false;
}



void psopt_acrobot_obs_t::dynamics(adouble* derivatives, adouble* path, adouble* states, adouble* controls, adouble* parameters,
                           adouble& time, adouble* xad, int iphase, Workspace* workspace)
{
    adouble theta2 = states[STATE_THETA_2];
    adouble theta1 = states[STATE_THETA_1] - M_PI / 2;
    adouble theta1dot = states[STATE_V_1];
    adouble theta2dot = states[STATE_V_2];
    adouble _tau = controls[CONTROL_T];

    //extra term m*lc2
    adouble d11 = m * lc2 + m * (l2 + lc2 + 2 * l * lc * cos(theta2)) + I1 + I2;

    adouble d22 = m * lc2 + I2;
    adouble d12 = m * (lc2 + l * lc * cos(theta2)) + I2;
    adouble d21 = d12;

    //extra theta1dot
    adouble c1 = -m * l * lc * theta2dot * theta2dot * sin(theta2) - (2 * m * l * lc * theta1dot * theta2dot * sin(theta2));
    adouble c2 = m * l * lc * theta1dot * theta1dot * sin(theta2);
    adouble g1 = (m * lc + m * l) * g * cos(theta1) + (m * lc * g * cos(theta1 + theta2));
    adouble g2 = m * lc * g * cos(theta1 + theta2);

    derivatives[STATE_THETA_1] = theta1dot;
    derivatives[STATE_THETA_2] = theta2dot;

    adouble u2 = _tau - 1 * .1 * theta2dot;
    adouble u1 = -1 * .1 * theta1dot;
    adouble theta1dot_dot = (d22 * (u1 - c1 - g1) - d12 * (u2 - c2 - g2)) / (d11 * d22 - d12 * d21);
    adouble theta2dot_dot = (d11 * (u2 - c2 - g2) - d21 * (u1 - c1 - g1)) / (d11 * d22 - d12 * d21);

    derivatives[STATE_V_1] = theta1dot_dot;
    derivatives[STATE_V_2] = theta2dot_dot;
}

adouble psopt_acrobot_obs_t::endpoint_cost(adouble* initial_states, adouble* final_states, adouble* parameters, adouble& t0,
                                 adouble& tf, adouble* xad, int iphase, Workspace* workspace)
{
    // Since we already set endpoint constraint in events, we don't need it here
    // TODO: maybe we can set one end free, but try to reduce the cost only
    // Here we use the time as endpoint cost for minimum time control
    //return 0.1*tf;
    //return 0.;
    double* goal = (double*) workspace->problem->user_data;
    //std::cout << "goal extracteed" << std::endl;
    adouble sum_of_square = 0;

    for (unsigned i=0; i < STATE_N; i++)
    {
        // wrap the angle to [-pi, pi]
        // if dx > pi, then dx = dx - 2pi
        // if dx < -pi, then dx = dx + 2pi
        adouble dx = final_state[i] - goal[i];
        dx = dx - M_PI*2 * (dx>M_PI);
        dx = dx + M_PI*2 * (dx<-M_PI);
        sum_of_square = sum_of_square + (final_states[i] - goal[i]) * (final_states[i] - goal[i]);
    }
    //std::cout << "after sum_of_square" << std::endl;
    return sum_of_square;
}

adouble psopt_acrobot_obs_t::integrand_cost(adouble* states, adouble* controls, adouble* parameters, adouble& time, adouble* xad,
                      int iphase, Workspace* workspace)
{
    adouble retval = 0.0;
    return retval;
}

void psopt_acrobot_obs_t::events(adouble* e, adouble* initial_states, adouble* final_states, adouble* parameters, adouble& t0,
            adouble& tf, adouble* xad, int iphase, Workspace* workspace)
{
  for (unsigned i=0; i < STATE_N; i++)
  {
      // it is angle then allow them to have 2pi difference
      // the mapping to [0, 2pi] are equal
      //e[STATE_N+i] = final_states[i];

      //if (i == 0 || i == 1)
      //{
          //std::cout << "initial_states[i]: " << (initial_states[i]).getValue() << std::endl;
          //e[i] = initial_states[i]-2*M_PI*ceil(floor(initial_states[i]/M_PI)/2);
          //e[i] = initial_states[i];
          //std::cout << "wrapped initial states[i]: " << (e[i]).getValue() << std::endl;
          //std::cout << "final_states[i]: " << (final_states[i]).getValue() << std::endl;
          //e[STATE_N+i] = final_states[i]-2*M_PI*ceil(floor(final_states[i]/M_PI)/2);
          //std::cout << "wrapped final states[i]: " << (e[STATE_N+i]).getValue() << std::endl;
          //e[STATE_N+i] = final_states[i]-2*M_PI*floor(final_states[i]/2/M_PI);
          //if (e[STATE_N+i] > M_PI)
          //{
          //      e[STATE_N+i] = e[STATE_N+i] - 2*M_PI;
          //}
      //}
      //else
      //{
          e[i] = initial_states[i];
          //e[STATE_N+i] = final_states[i];
          //e[STATE_N+i] = 0.;
      //}
  }

}

void psopt_acrobot_obs_t::linkages(adouble* linkages, adouble* xad, Workspace* workspace)
{
  // No linkages in this single phase problem
}
