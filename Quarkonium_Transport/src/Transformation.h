#ifndef TRANSFORMATION_H        // check if TRANSFORMATION_H has been defined, if not, then define
#define TRANSFORMATION_H

#include <cmath>
#include <iostream>
#include <vector>
#include "utility.h"

// polar coordinates to cartisian coordinates
std::vector<double> polar_to_cartisian1(double length, double cos, double phi){
    std::vector<double> result(3);
    double sin = std::sqrt(1.-cos*cos);
    double c_phi = std::cos(phi);
    double s_phi = std::sin(phi);
    result[0] = length * sin * c_phi;
    result[1] = length * sin * s_phi;
    result[2] = length * cos;
    return result;
}

std::vector<double> polar_to_cartisian2(double length, double cos, double sin, double c_phi, double s_phi){
    std::vector<double> result(3);
    result[0] = length * sin * c_phi;
    result[1] = length * sin * s_phi;
    result[2] = length * cos;
    return result;
}

// add real gluon momentum to the decay products: QQbar
std::vector<double> add_real_gluon(std::vector<double> momentum_add, std::vector<double> momentum_rel){
    double q_x = 0.5*momentum_add[0];
    double q_y = 0.5*momentum_add[1];
    double q_z = 0.5*momentum_add[2];
    std::vector<double> pQpQbar(6);
    pQpQbar[0] = q_x + momentum_rel[0];
    pQpQbar[1] = q_y + momentum_rel[1];
    pQpQbar[2] = q_z + momentum_rel[2];
    pQpQbar[3] = q_x - momentum_rel[0];
    pQpQbar[4] = q_y - momentum_rel[1];
    pQpQbar[5] = q_z - momentum_rel[2];
    return pQpQbar;
}

// add virtual gluon momentum to the decay products: QQbar
std::vector<double> add_virtual_gluon(std::vector<double> momentum_1, std::vector<double> momentum_2, std::vector<double> momentum_rel){
    double q_x = 0.5*(momentum_1[0]-momentum_2[0]);
    double q_y = 0.5*(momentum_1[1]-momentum_2[1]);
    double q_z = 0.5*(momentum_1[2]-momentum_2[2]);
    std::vector<double> pQpQbar(6);
    pQpQbar[0] = q_x + momentum_rel[0];
    pQpQbar[1] = q_y + momentum_rel[1];
    pQpQbar[2] = q_z + momentum_rel[2];
    pQpQbar[3] = q_x - momentum_rel[0];
    pQpQbar[4] = q_y - momentum_rel[1];
    pQpQbar[5] = q_z - momentum_rel[2];
    return pQpQbar;
}

// subtract real gluon momentum to the reco produce: |nlm>
std::vector<double> subtract_real_gluon(std::vector<double> momentum_subtract){
    std::vector<double> p1S(3);    // 1S here represents quarkonia
    p1S[0] = -momentum_subtract[0];
    p1S[1] = -momentum_subtract[1];
    p1S[2] = -momentum_subtract[2];
    return p1S;
}

// subtract virtual gluon momentum to the reco produce: |nlm>
std::vector<double> subtract_virtual_gluon(std::vector<double> momentum_1, std::vector<double> momentum_2){
    std::vector<double> p1S(3);
    p1S[0] = momentum_1[0]-momentum_2[0];
    p1S[1] = momentum_1[1]-momentum_2[1];
    p1S[2] = momentum_1[2]-momentum_2[2];
    return p1S;
}

#endif
