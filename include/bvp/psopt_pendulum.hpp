/**
 * @file pendulum.hpp
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

#ifndef PSOPT_PENDULUM_HPP
#define PSOPT_PENDULUM_HPP
#ifndef PSOPT_H
#define PSOPT_H
#include "psopt.h"
#endif

#include "bvp/psopt_system.hpp"

class psopt_pendulum_t : public psopt_system_t
{
public:
	psopt_pendulum_t()
        : psopt_system_t()
	{
		state_dimension = 2;
		control_dimension = 1;
		temp_state = new double[state_dimension];
	}
	virtual ~psopt_pendulum_t(){}
    std::string get_name() const override;
	double max_distance() const override;
	/**
	 * @copydoc system_t::propagate(double*, double*, int, int, double*, double& )
	 */
	virtual int propagate(
		const double* start_state, unsigned int state_dimension,
        const double* control, unsigned int control_dimension,
	    int num_steps, double* result_state, double integration_step);

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
	virtual void enforce_bounds();

	/**
	 * @copydoc system_t::valid_state()
	 */
	virtual bool valid_state();

	/**
	 * @copydoc system_t::visualize_point(double*, svg::Dimensions)
	 */
	std::tuple<double, double> visualize_point(const double* state, unsigned int state_dimension) const override;

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
};


#endif
