/**
* Define the system propagate function in psopt format (adouble)
*/
#include "bvp/psopt_system.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#define M 1450
#define IZ 2740
#define LF 1.3
#define LR 1.4
#define R .3
#define IF 1.8
#define IR 1.8
#define H .4
#define B 7
#define C 1.6
#define D .52

#define CRBRAKE 700
#define CRACC 0
#define CFBRAKE 700
#define CFACC 1000

#define STATE_X 0
#define STATE_Y 1
#define STATE_VX 2
#define STATE_VY 3
#define STATE_THETA 4
#define STATE_THETADOT 5
#define STATE_WF 6
#define STATE_WR 7
#define CONTROL_STA 0
#define CONTROL_TF 1
#define CONTROL_TR 2

#define MIN_X -25
#define MAX_X 25
#define MIN_Y -35
#define MAX_Y 25


void psopt_rally_car_t::dynamics(adouble* derivatives, adouble* path, adouble* states, adouble* controls, adouble* parameters,
                           adouble& time, adouble* xad, int iphase, Workspace* workspace)
{
    adouble _vx = states[2];
    adouble _vy = states[3];
    adouble _theta = states[4];
    adouble _thetadot = states[5];
    adouble _wf = states[6];
    adouble _wr = states[7];

    adouble _sta = controls[0];
    adouble _tf = controls[0];
    adouble _tr = controls[0];

    derivatives[STATE_X] = _vx;
    derivatives[STATE_Y] = _vy;
    derivatives[STATE_THETA] = _thetadot;

    adouble V = sqrt(_vx*_vx+_vy*_vy);
    adouble beta = atan2(_vy,_vx) - _theta;
    adouble V_Fx = V*cos(beta-_sta) + _thetadot*LF*sin(_sta);
    adouble V_Fy = V*sin(beta-_sta) + _thetadot*LF*cos(_sta);
    adouble V_Rx = V*cos(beta);
    adouble V_Ry = V*sin(beta) - _thetadot*LR;

    adouble s_Fx = (V_Fx - _wf*R)/(_wf*R);
    adouble s_Fy = V_Fy/(_wf*R);
    adouble s_Rx = (V_Rx - _wr*R)/(_wr*R);
    adouble s_Ry = V_Ry/(_wr*R);

    adouble s_F = sqrt(s_Fx*s_Fx+s_Fy*s_Fy);
    adouble s_R = sqrt(s_Rx*s_Rx+s_Ry*s_Ry);

    adouble mu_F = D*sin(C*atan(B*s_F));
    adouble mu_R = D*sin(C*atan(B*s_R));
    adouble mu_Fx;
    adouble mu_Fy;
    if(std::isfinite(s_Fx))
            mu_Fx = -1*(s_Fx/s_F)*mu_F;
    else
            mu_Fx = -mu_F;
    if(std::isfinite(s_Fy))
            mu_Fy = -1*(s_Fy/s_F)*mu_F;
    else
            mu_Fy = -mu_F;
    adouble mu_Rx;
    adouble mu_Ry;
    if(std::isfinite(s_Rx))
            mu_Rx = -1*(s_Rx/s_R)*mu_R;
    else
            mu_Rx = -mu_R;
    if(std::isfinite(s_Ry))
            mu_Ry = -1*(s_Ry/s_R)*mu_R;
    else
            mu_Ry = -mu_R;

    adouble fFz = (LR*M*(9.8) - H*M*9.8*mu_Rx) / (LF+LR+H*(mu_Fx*cos(_sta)-mu_Fy*sin(_sta)-mu_Rx));
    adouble fRz = M*9.8 - fFz;

    adouble fFx = mu_Fx * fFz;
    adouble fFy = mu_Fy * fFz;
    adouble fRx = mu_Rx * fRz;
    adouble fRy = mu_Ry * fRz;;

    derivatives[STATE_VX] = (fFx*cos(_theta+_sta)-fFy*sin(_theta+_sta)+fRx*cos(_theta)-fRy*sin(_theta) )/M;
    derivatives[STATE_VY] = (fFx*sin(_theta+_sta)+fFy*cos(_theta+_sta)+fRx*sin(_theta)+fRy*cos(_theta) )/M;
    derivatives[STATE_THETADOT] = ((fFy*cos(_sta)+fFx*sin(_sta))*LF - fRy*LR)/IZ;
    derivatives[STATE_WF] = (_tf-fFx*R)/IF;
    derivatives[STATE_WR] = (_tr-fRx*R)/IR;

}
