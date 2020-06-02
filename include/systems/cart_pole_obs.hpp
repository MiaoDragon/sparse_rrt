/**
 * @file cart_pole_obs.hpp
 *
 * @copyright Software License Agreement (BSD License)
 * Original work Copyright (c) 2014, Rutgers the State University of New Jersey, New Brunswick
 * Modified work Copyright 2017 Oleg Y. Sinyavskiy
 * All Rights Reserved.
 * For a full description see the file named LICENSE.
 *
 * Original authors: Zakary Littlefield, Kostas Bekris
 * Modifications by: Yinglong Miao
 *
 */

#ifndef SPARSE_CART_POLE_OBS_HPP
#define SPARSE_CART_POLE_OBS_HPP


#include "systems/system.hpp"

class cart_pole_obs_t : public system_t
{
public:
	cart_pole_obs_t(std::vector<std::vector<double>>& _obs_list, double width)
	/**
	* :obs_list: list of obstacles represented by 2d array representing the middle point
	* :width: the width of the obstacle
	*/
	{
		state_dimension = 4;
		control_dimension = 1;
		temp_state = new double[state_dimension];
		deriv = new double[state_dimension];

		// copy the items from _obs_list to obs_list
		for(unsigned i=0;i<_obs_list.size();i++)
		{
			// each obstacle is represented by its middle point
			std::vector<double> obs(4*2);
			// calculate the four points representing the rectangle in the order
			// UL, UR, LR, LL
			// the obstacle points are concatenated for efficient calculation
			double x = _obs_list[i][0];
			double y = _obs_list[i][1];
			obs[0] = x - width / 2;  obs[1] = y + width / 2;
			obs[2] = x + width / 2;  obs[3] = y + width / 2;
			obs[4] = x + width / 2;  obs[5] = y - width / 2;
			obs[6] = x - width / 2;  obs[7] = y - width / 2;
			obs_list.push_back(obs);
		}

	}
	virtual ~cart_pole_obs_t(){
	    delete temp_state;
	    delete deriv;
		// clear the vector
		obs_list.clear();
	}

	/**
	 * @copydoc system_t::propagate(const double*, const double*, int, int, double*, double& )
	 */
	virtual int propagate(
	    const double* start_state, unsigned int state_dimension,
        const double* control, unsigned int control_dimension,
	    int num_steps, double* result_state, double integration_step);

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

protected:
	double* deriv;
	void update_derivative(const double* control);
	// for obstacle
	std::vector<std::vector<double>> obs_list;
	// collision checker
	// from http://www.jeffreythompson.org/collision-detection/line-rect.php
	bool lineLine(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4);
};


#endif
