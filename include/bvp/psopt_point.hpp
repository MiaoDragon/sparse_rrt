/**
 * @file point.hpp
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

#ifndef PSOPT_POINT_HPP
#define PSOPT_POINT_HPP
#ifndef PSOPT_H
#define PSOPT_H
#include "psopt.h"
#endif

#include "systems/point.hpp"
#include "bvp/psopt_system.hpp"

/**
 * @brief A simple system implementing a 2d point.
 * @details A simple system implementing a 2d point. It's controls include velocity and direction.
 */
class psopt_point_t : public psopt_system_t
{
public:
	psopt_point_t(int number_of_obstacles=5)
	{
		state_dimension = 2;
		control_dimension = 2;
		temp_state = new double[state_dimension];

		std::vector<Rectangle_t> available_obstacles;
		available_obstacles.push_back(Rectangle_t(   1,  -1.5,    5,  9.5));
		available_obstacles.push_back(Rectangle_t(  -8,  4.25,   -1, 5.75));
		available_obstacles.push_back(Rectangle_t(   5,   3.5,    9,  4.5));
		available_obstacles.push_back(Rectangle_t(-8.5,  -7.5, -3.5, -2.5));
		available_obstacles.push_back(Rectangle_t(   5,  -8.5,    9, -4.5));

		for (int i =0; i<number_of_obstacles; i++) {
		    obstacles.push_back(available_obstacles[i]);
		}

	}
	virtual ~psopt_point_t(){ delete temp_state;}
	std::string get_name() const override;
	double max_distance() const override;
	/**
	 * @copydoc system_t::propagate(double*, double*, int, int, double*, double& )
	 */
	virtual int propagate(
	    const double* start_state, unsigned int state_dimension,
        const double* control, unsigned int control_dimension,
	    int num_steps, double* result_state, double integration_step) override;

    static void dynamics(adouble* derivatives, adouble* path, adouble* states, adouble* controls, adouble* parameters,
             adouble& time, adouble* xad, int iphase, Workspace* workspace);
 	static adouble endpoint_cost(adouble* initial_states, adouble* final_states, adouble* parameters, adouble& t0,
             adouble& tf, adouble* xad, int iphase, Workspace* workspace);
	static adouble integrand_cost(adouble* states, adouble* controls, adouble* parameters, adouble& time, adouble* xad,
	         int iphase, Workspace* workspace);
	static void events(adouble* e, adouble* initial_states, adouble* final_states, adouble* parameters, adouble& t0,
	         adouble& tf, adouble* xad, int iphase, Workspace* workspace);
	static void linkages(adouble* linkages, adouble* xad, Workspace* workspace);

	/**
	 * @copydoc system_t::enforce_bounds()
	 */
	virtual void enforce_bounds() override;

	/**
	 * @copydoc system_t::valid_state()
	 */
	virtual bool valid_state() override;

	/**
	 * @copydoc system_t::visualize_point(double*, svg::Dimensions)
	 */
	std::tuple<double, double> visualize_point(const double* state, unsigned int state_dimension) const override;

	/**
	 * @copydoc system_t::visualize_obstacles(svg::DocumentBody&, svg::Dimensions)
	 */
    std::string visualize_obstacles(int image_width, int image_height) const override;

	/**
	 * @copydoc system_t::get_state_bounds()
	 */
    std::vector<std::pair<double, double>> get_state_bounds() const override;

    /**
	 * @copydoc system_t::get_control_bounds()
	 */
    std::vector<std::pair<double, double>> get_control_bounds() const override;

    /**
	 * @copydoc system_t::is_circular_topology()
	 */
    std::vector<bool> is_circular_topology() const override;

protected:

	std::vector<Rectangle_t> obstacles;

};


#endif
