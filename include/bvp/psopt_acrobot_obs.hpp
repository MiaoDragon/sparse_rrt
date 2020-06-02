/**
* Define the system propagate function in psopt format (adouble)
*/

#ifndef PSOPT_TWO_LINK_ACROBOT_OBS_HPP
#define PSOPT_TWO_LINK_ACROBOT_OBS_HPP

#ifndef PSOPT_H
#define PSOPT_H
#include "psopt.h"
#endif

#include "bvp/psopt_system.hpp"

class psopt_acrobot_obs_t : public psopt_system_t
{
public:
	psopt_acrobot_obs_t(std::vector<std::vector<double>>& _obs_list, double width)
        : psopt_system_t()
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
	virtual ~psopt_acrobot_obs_t(){
	    delete[] temp_state;
	    delete[] deriv;
		obs_list.clear();
	}
	static double distance(const double* point1, const double* point2, unsigned int);

	std::string get_name() const override;
	double max_distance() const override;
	/**
	 * @copydoc system_t::propagate(const double*, const double*, int, int, double*, double& )
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

protected:
	double* deriv;
	void update_derivative(const double* control);
	// for obstacle
	std::vector<std::vector<double>> obs_list;

	bool lineLine(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4);
};





#endif
