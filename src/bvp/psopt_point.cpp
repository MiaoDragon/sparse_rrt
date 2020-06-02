/**
 * @file point.cpp
 *
 * @copyright Software License Agreement (BSD License)
 * Original work Copyright (c) 2014, Rutgers the State University of New Jersey, New Brunswick
 * Modified work Copyright 2017 Oleg Y. Sinyavskiy
 * All Rights Reserved.
 * For a full description see the file named LICENSE.
 *
 * Original authors: Zakary Littlefield, Kostas Bekris
 * Modifications by: Oleg Y. Sinyavskiy
 *
 */

#include "bvp/psopt_point.hpp"
#include "utilities/random.hpp"
#include "image_creation/svg_image.hpp"
#include <cmath>
#include <assert.h>

#define MIN_X -10
#define MAX_X 10
#define MIN_Y -10
#define MAX_Y 10

#define MIN_V 0
#define MAX_V 10
#define MIN_THETA -3.14
#define MAX_THETA 3.14
#define STATE_N 2
std::string psopt_point_t::get_name() const
{
    return "point";
}
double psopt_point_t::max_distance() const
{
    return sqrt((MAX_X-MIN_X)*(MAX_X-MIN_X)+(MAX_Y-MIN_Y)*(MAX_Y-MIN_Y));
}

int psopt_point_t::propagate(
    const double* start_state, unsigned int state_dimension,
    const double* control, unsigned int control_dimension,
    int num_steps, double* result_state, double integration_step)
{
	temp_state[0] = start_state[0];
	temp_state[1] = start_state[1];
	bool validity = false;
    int actual_num_steps = 0;

	for(int i=0;i<num_steps;i++)
	{
		temp_state[0] += integration_step*control[0]*cos(control[1]);
		temp_state[1] += integration_step*control[0]*sin(control[1]);
		enforce_bounds();
        if (valid_state())
        {
            result_state[0] = temp_state[0];
            result_state[1] = temp_state[1];
            validity = true;
            actual_num_steps += 1;
        }
        else
        {
            // Found the earliest invalid position. break the loop and return
            validity = false; // need to update validity because one node is invalid, the propagation fails
            break;
        }
	}
	return actual_num_steps;
}

void psopt_point_t::enforce_bounds()
{
	if(temp_state[0]<MIN_X)
		temp_state[0]=MIN_X;
	else if(temp_state[0]>MAX_X)
		temp_state[0]=MAX_X;

	if(temp_state[1]<MIN_Y)
		temp_state[1]=MIN_Y;
	else if(temp_state[1]>MAX_Y)
		temp_state[1]=MAX_Y;
}


bool psopt_point_t::valid_state()
{
	bool obstacle_collision = false;
	//any obstacles need to be checked here
	for(unsigned i=0;i<obstacles.size() && !obstacle_collision;i++)
	{
		if(	temp_state[0]>obstacles[i].low_x &&
			temp_state[0]<obstacles[i].high_x &&
			temp_state[1]>obstacles[i].low_y &&
			temp_state[1]<obstacles[i].high_y)
		{
			obstacle_collision = true;
		}
	}

	return !obstacle_collision &&
			(temp_state[0]!=MIN_X) &&
			(temp_state[0]!=MAX_X) &&
			(temp_state[1]!=MIN_Y) &&
			(temp_state[1]!=MAX_Y);
}

std::tuple<double, double> psopt_point_t::visualize_point(const double* state, unsigned int state_dimension) const
{
	double x = (state[0]-MIN_X)/(MAX_X-MIN_X);
	double y = (state[1]-MIN_Y)/(MAX_Y-MIN_Y);
	return std::make_tuple(x, y);
}

std::string psopt_point_t::visualize_obstacles(int image_width, int image_height) const
{
    svg::Dimensions dims(image_width, image_height);
    svg::DocumentBody doc(svg::Layout(dims, svg::Layout::BottomLeft));
	double temp[2];
	for(unsigned i=0;i<obstacles.size();i++)
	{
		temp[0] = obstacles[i].low_x;
		temp[1] = obstacles[i].high_y;
		double x, y;
		std::tie(x, y) = this->visualize_point(temp, 2);

		doc<<svg::Rectangle(svg::Point(x*dims.width, y*dims.height),
							(obstacles[i].high_x-obstacles[i].low_x)/(MAX_X-MIN_X) * dims.width,
							(obstacles[i].high_y-obstacles[i].low_y)/(MAX_Y-MIN_Y) * dims.height,
							svg::Color::Red);
	}
    return doc.toString();
}

std::vector<std::pair<double, double> > psopt_point_t::get_state_bounds() const {
	return {
			{MIN_X,MAX_X},
			{MIN_Y,MAX_Y}
	};
}


std::vector<std::pair<double, double> > psopt_point_t::get_control_bounds() const {
	return {
			{MIN_V, MAX_V},
			{MIN_THETA, MAX_THETA},
	};
}


std::vector<bool> psopt_point_t::is_circular_topology() const {
    return {
            false,
            false
    };
}


void psopt_point_t::dynamics(adouble* derivatives, adouble* path, adouble* states, adouble* controls, adouble* parameters,
                           adouble& time, adouble* xad, int iphase, Workspace* workspace)
{
    derivatives[0] = controls[0] * cos(controls[1]);
    derivatives[1] = controls[0] * sin(controls[1]);
}

adouble psopt_point_t::endpoint_cost(adouble* initial_states, adouble* final_states, adouble* parameters, adouble& t0,
                                 adouble& tf, adouble* xad, int iphase, Workspace* workspace)
{
    // Since we already set endpoint constraint in events, we don't need it here
    // TODO: maybe we can set one end free, but try to reduce the cost only
    // Here we use the time as endpoint cost for minimum time control
    return tf;
}

adouble psopt_point_t::integrand_cost(adouble* states, adouble* controls, adouble* parameters, adouble& time, adouble* xad,
                      int iphase, Workspace* workspace)
{
    adouble retval = 0.0;
    return retval;
}

void psopt_point_t::events(adouble* e, adouble* initial_states, adouble* final_states, adouble* parameters, adouble& t0,
            adouble& tf, adouble* xad, int iphase, Workspace* workspace)
{
  for (unsigned i=0; i < STATE_N; i++)
  {
      e[i] = initial_states[i];
      e[STATE_N+i] = final_states[i];
  }
}

void psopt_point_t::linkages(adouble* linkages, adouble* xad, Workspace* workspace)
{
  // No linkages in this single phase problem
}
