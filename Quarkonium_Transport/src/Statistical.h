#ifndef STATISTICAL_H        // check if STATISTICAL_H has been defined, if not, then define
#define STATISTICAL_H

#include <cmath>
#include <iostream>
#include "DisRec.h"
#include <vector>

// static and moving frame statistical
double nB(double x){
    return 1./(std::exp(x)-1.);
}

double nBplus1(double x){
    return 1./(1. - std::exp(-x));
}

double nF(double x){
    return 1./(std::exp(x)+1.);
}

double nFminus1(double x){
    // it is 1-n_F
    return 1./(1. + std::exp(-x));
}

double fac1(double z){
    return std::log(1. - std::exp(-z));
}

double fac2(double z){
    return std::log(1. + std::exp(-z));
}

double Li2(double z){
    int n = 0;
    double result = 0.;
    double nterm;
    do{
        n += 1;
        nterm = pow(z, n)/(n*n);
        result += nterm;
    } while(std::abs(nterm/result) > accuracy);
    return result;
}


#endif
