/**
 * @file car.hpp
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

#ifndef SPARSE_CAR_OBS_HPP
#define SPARSE_CAR_OBS_HPP

#include "systems/system.hpp"
#include <iostream>

class car_obs_t : public system_t
{
public:
	car_obs_t(std::vector<std::vector<double>>& _obs_list, double width)
	{
		state_dimension = 3;
		control_dimension = 2;
		temp_state = new double[state_dimension];
		deriv = new double[state_dimension];

		for(unsigned i=0;i<_obs_list.size();i++)
        {
            // each obstacle is represented by its middle point
            std::vector<double> obs(4*2);
            // calculate the four points representing the rectangle in the order
            // UL, UR, LR, LL
            // the obstacle points are concatenated for efficient calculation
            double x = _obs_list[i][0];
            double y = _obs_list[i][1];
            obs[0] = x - width / 2;  obs[1] = y - width / 2;
            obs[2] = x + width / 2;  obs[3] = y - width / 2;
            obs[4] = x + width / 2;  obs[5] = y + width / 2;
            obs[6] = x - width / 2;  obs[7] = y + width / 2;
            obs_list.push_back(obs);

            std::vector<std::vector<double>> obs_axis_i(2, std::vector<double> (2, 0));
            obs_axis_i[0][0] = obs[2] - obs[0];
            obs_axis_i[1][0] = obs[6] - obs[0];
            obs_axis_i[0][1] = obs[3] - obs[1];
            obs_axis_i[1][1] = obs[7] - obs[1];

            std::vector<double> obs_length;
            obs_length.push_back(obs_axis_i[0][0]*obs_axis_i[0][0]+obs_axis_i[0][1]*obs_axis_i[0][1]);
            obs_length.push_back(obs_axis_i[1][0]*obs_axis_i[1][0]+obs_axis_i[1][1]*obs_axis_i[1][1]);
            for (unsigned i1=0; i1<2; i1++)
            {
                for (unsigned j1=0; j1<2; j1++)
                {
                    obs_axis_i[i1][j1] = obs_axis_i[i1][j1] / obs_length[j1];
                }
            }
            obs_axis.push_back(obs_axis_i);

            // not sure if below is correct
            std::vector<double> obs_ori_i;
            obs_ori_i.push_back(obs[0]*obs_axis_i[0][0]+ obs[1]*obs_axis_i[0][1]);
            obs_ori_i.push_back(obs[0]*obs_axis_i[1][0]+ obs[1]*obs_axis_i[1][1]);

            obs_ori.push_back(obs_ori_i);
        }
		std::cout << "after initialization" << std::endl;
	}
	virtual ~car_obs_t()
	{
		delete temp_state;
		delete deriv;
		obs_list.clear();
        obs_axis.clear();
        obs_ori.clear();
	}

    /**
	 * @copydoc system_t::propagate()
	 */
	virtual bool propagate(
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
	 * @copydoc system_t::visualize_point()
	 */
	std::tuple<double, double> visualize_point(const double* state, unsigned int state_dimension) const override;

	/**
	 * @copydoc system_t::get_state_bounds()
	 */
	virtual std::vector<std::pair<double, double> > get_state_bounds() const;

	/**
	 * @copydoc system_t::get_control_bounds()
	 */
    virtual std::vector<std::pair<double, double> > get_control_bounds() const;

    /**
	 * @copydoc system_t::is_circular_topology()
	 */
	std::vector<bool> is_circular_topology() const override;

	    bool overlap(std::vector<std::vector<double>>& b1corner, std::vector<std::vector<double>>& b1axis,
	                  std::vector<double>& b1orign, std::vector<double>& b2corner,
	                  std::vector<std::vector<double>>& b2axis, std::vector<double>& b2orign);

protected:
	double* deriv;
	void update_derivative(const double* control);
    std::vector<std::vector<double>> obs_list;
    std::vector<std::vector<std::vector<double>>> obs_axis;
    std::vector<std::vector<double>> obs_ori;
};


#endif
