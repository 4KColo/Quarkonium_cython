#include <iostream>
#include <cmath>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_vegas.h>
#include "DisRec.h"
#include "Statistical.h"
#include "Transformation.h"
#include <random>
#include <vector>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> dist_u(0., 4.);
std::uniform_real_distribution<double> rejection(0., 1.);
std::uniform_real_distribution<double> y_cdf(0., 1.);
std::uniform_real_distribution<double> reco_uniform(-1., 1.);
std::uniform_real_distribution<double> sample_inel(0., 1.);
std::uniform_real_distribution<double> sample_cos(-1., 1.);


// find maximum of a function
// only works for positive-function with one local maximum within [xL, xH]
double find_max_noparams(double(*f)(double x), double xL_, double xR_){
    double dfL, dfR, dfM, xM, xL = xL_, xR = xR_, dx, fM;
    dx = (xR-xL)/20.;
    dfL = f(xL+dx) - f(xL);
    dfR = f(xR) - f(xR-dx);
    do{
        xM = (xL+xR)/2.;
        dfM = f(xM+dx) - f(xM-dx);
        fM = f(xM);
        if (dfL*dfM < 0) {xR = xM; dfR = dfM;}
        if (dfM*dfR < 0) {xL = xM; dfL = dfM;}
        dx = (xR-xL)/20.;
    }while ( std::abs(dfM/dx/fM) > accuracy );
    return fM;
}

double find_max(double(*f)(double x, void * params), void * params, double xL_, double xR_){
    double dfL, dfR, dfM, xM, xL = xL_, xR = xR_, dx, fM;
    dx = (xR-xL)/20.;
    dfL = f(xL+dx, params) - f(xL, params);
    dfR = f(xR, params) - f(xR-dx, params);
    do{
        xM = (xL+xR)/2.;
        dfM = f(xM+dx, params) - f(xM-dx, params);
        fM = f(xM, params);
        if (dfL*dfM < 0) {xR = xM; dfR = dfM;}
        if (dfM*dfR < 0) {xL = xM; dfL = dfM;}
        dx = (xR-xL)/20.;
    }while ( std::abs(dfM/dx/fM) > accuracy );
    return fM;
}
// find the root of a monotonically increasing or decreasing function
double find_root(double(*f)(double x, void * params), double result, void * params, double xL_, double xR_){
    double xL = xL_, xR = xR_, xM, fL, fM, fR;
    fL = f(xL, params) - result;
    fR = f(xR, params) - result;
    if (fL*fR >= 0.0){
        if (fL >= 0.0){
            return xL;
        }
        else{
            return xR;
        }
    }
    else{
        do{
            xM = 0.5*(xR+xL);
            fL = f(xL, params) - result;
            //fR = f(xR, params) - result;
            fM = f(xM, params) - result;
            if (fL*fM < 0.){
                xR = xM;
            }
            else{
                xL = xM;
            }
        } while (std::abs((xL-xR)/xM) > accuracy );
    }
    return xM;
}


//--------------------------- Quarkonium dissociation -------------------------------
// Matrix element of |<1S|Psi(p)>|^2, (Coulomb interaction)
double Matrix1S(double p){
    double eta = Matrix1S_scale/(p+small_number);
    double x = a_B * p;
    double Numerator = Matrix1S_prefactor*eta*(rho_c_sq+x*x)*std::exp(4.0*eta*std::atan(x));
    double Denominator = pow(1.0+x*x, 6) * ( std::exp(TwoPi*eta) - 1.0 );
    return Numerator/Denominator;
}

double pMatrix1S(double p){
    return p*Matrix1S(p);
}

double p2Matrix1S(double p){
    return p*p*Matrix1S(p);
}

//---------------------- gluo-dissociation -----------------------
// Two formula (with the same reuslts) of the Cross-section of 1S -> Psi(p)
// first one
double Xsec1S(double q){
    if (q <= E1S) return 0.;
    double t1 = std::sqrt(q/E1S - 1.);
    double x = rho_c/t1;
    return Xsec1S_prefactor/pow(q, 5) * (t1*t1 + rho_c_sq)
    * std::exp(4.*x*std::atan(t1)) / (std::exp(TwoPi*x) - 1.);
}
//second one
double Xsec1S_v2(double q){
    if (q <= E1S) return 0.;
    double p = std::sqrt(M*(q-E1S));
    return 2./3.*fix_alpha_s*CF*M*q*p*Matrix1S(p);
}

// differential decay rate dR/dq, q is the momentum of incident gluon
// and its approximation form in the limit of small v
double dRdq_1S_gluon(double q, void * params_){
    double * params = static_cast<double *>(params_);
    double k1 = params[0]; //k1 is gamma*(1.+v)/T;
    double k2 = params[1]; //k2 is gamma*(1.-v)/T;
    return q*Xsec1S(q)*(fac1(q*k1) - fac1(q*k2));
}

double dRdq_1S_gluon_small_v(double q, void * params_){
    double * params = static_cast<double *>(params_);
    double T = params[0];
    return q*q*Xsec1S(q)*nB(q/T);
}

double qdRdq_1S_gluon_u(double u, void * params_){
    double * params = static_cast<double *>(params_);
    double v = params[0];
    double T = params[1];
    v = std::max(v, small_number);
    double gamma = 1./sqrt(1.-v*v);
    double new_params[2];
    new_params[0] = gamma*(1.+v)/T;
    new_params[1] = gamma*(1.-v)/T;
    double q = E1S*std::exp(u);
    return q*dRdq_1S_gluon(q, new_params);
}

// this function intergate dR/dq to get R_decay
// this is used for tabulation, not for monte-carlo simulation.
// there is a interpolation function that interpolates the table for MC-simulation.
double R1S_decay_gluon(double vabs, double T){
    if (vabs < small_number){
        double result, error, qmin=E1S, qmax=E1S*100.;
        gsl_integration_workspace *w = gsl_integration_workspace_alloc(1);
        double * params = new double[1];
        params[0] = T;
        gsl_function F;
        F.function = dRdq_1S_gluon_small_v;
        F.params = params;
        gsl_integration_qag(&F, qmin, qmax, 0, 1e-6, 1000, 6, w, &result, &error);
        delete [] params;
        gsl_integration_workspace_free(w);
        return result / (TwoPi*M_PI);
    }
    else{
        double gamma = 1./std::sqrt(1. - vabs*vabs);
        double k1 = gamma*(1.+vabs)/T;
        double k2 = gamma*(1.-vabs)/T;
        double result, error, qmin=E1S, qmax=E1S*100.;
        gsl_integration_workspace *w = gsl_integration_workspace_alloc(5000);
        double * params = new double[2];
        params[0] = k1;
        params[1] = k2;
        gsl_function F;
        F.function = dRdq_1S_gluon;
        F.params = params;
        gsl_integration_qag(&F, qmin, qmax, 0, 1e-6, 1000, 6, w, &result, &error);
        delete [] params;
        gsl_integration_workspace_free(w);
        return result * T / (TwoPi*TwoPi*vabs*gamma*gamma);
    }
}

// sample dR/dq (rejection) and dR/dq/dcostheta_q (inverse CDF)
double S1S_decay_gluon_q(double v, double T, double maximum){
    // uniformly sampling u in [0,4]
    // q = E1S*exp(u)
    // dRdu = dRdq(q(u))*dq/du = dRdq(q(u))*q(u)
    double u, result, params[2];
    params[0] = v; params[1] = T;
    do{
        u = dist_u(gen);
        result = qdRdq_1S_gluon_u(u, params)/maximum;
    } while( rejection(gen) > result );
    return E1S*std::exp(u);
}

double S1S_decay_gluon_costheta(double q, double v, double T){
    v = std::max(v, small_number);
    double coeff = q/T/std::sqrt(1. - v*v);     // parameter B = coeff
    double low = fac1(coeff*(1.-v));
    double norm = fac1(coeff*(1.+v))-low;
    double y_fac1 = y_cdf(gen)*norm + low;
    return ( -std::log(1. - std::exp(y_fac1))/coeff - 1. )/v;
}

// return the final relative momentum between Q-Qbar pair
double S1S_decay_gluon_final_p(double q){
    return std::sqrt((q-E1S)*M);
}

std::vector<double> S1S_decay_gluon(double v, double T, double maximum){
    double q = S1S_decay_gluon_q(v, T, maximum);
    double cos = S1S_decay_gluon_costheta(q, v, T);
    double phi = sample_inel(gen)*TwoPi;
    double p_rel = S1S_decay_gluon_final_p(q);
    double cos_rel = sample_cos(gen);
    double phi_rel = sample_inel(gen)*TwoPi;
    std::vector<double> momentum_gluon(3);
    std::vector<double> momentum_rel(3);
    std::vector<double> pQpQbar_final(6);
    momentum_gluon = polar_to_cartisian1(q, cos, phi);
    momentum_rel = polar_to_cartisian1(p_rel, cos_rel, phi_rel);
    pQpQbar_final = add_real_gluon(momentum_gluon, momentum_rel);
    return pQpQbar_final;
}
//---------------------- end of gluo-dissociation -----------------------


//---------------------- inelastic quark dissociation -----------------------
// first define integrand
double dRdp1dp2_1S_decay_ineq(double x[5], size_t dim, void * params_){
    double p1 = x[0];
    double c1 = x[1];
    double p2 = x[2];
    double c2 = x[3];
    double phi = x[4];
    double * params = static_cast<double *>(params_);
    double v = params[0];
    double T = params[1];
    if (p1-p2 < E1S){
        return 0.0;
    }
    else{
        double gamma = 1./std::sqrt(1.-v*v);
        double p0 = std::sqrt((p1-p2-E1S)*M);
        double s1 = std::sqrt(1.-c1*c1);
        double s2 = std::sqrt(1.-c2*c2);
        double phase1 = p1*nF(gamma*p1*(1.+v*c1)/T);
        double phase2 = p2*(1.-nF(gamma*p2*(1.+v*c2)/T));
        double part_angle = s1*s2*std::cos(phi)+c1*c2;
        double prop = p1*p2*(1.+part_angle)/( p1*p1+p2*p2-2.*p1*p2*part_angle );
        return phase1 * phase2 * p0 * Matrix1S(p0) * prop;
        // omit a prefactor and a 1/gamma, add them in the rate calculation: R1S_decay_ineq
    }
}
double R1S_decay_ineq(double v, double T){
    v = std::max(v, small_number);
    double gamma_inv = std::sqrt(1.-v*v);
    double * params = new double[2];
    params[0] = v;
    params[1] = T;
    double result, error;
    double p1up = 18.*T/std::sqrt(1.-v);
    double p2up = p1up - E1S;
    double xl[5] = { E1S, -1., 0., -1., 0. };
    double xu[5] = { p1up, 1., p2up, 1., TwoPi };
    gsl_monte_function F;
    F.f = &dRdp1dp2_1S_decay_ineq;
    F.dim = 5;
    F.params = params;
    gsl_rng * r = gsl_rng_alloc (gsl_rng_default);
    gsl_monte_vegas_state *w = gsl_monte_vegas_alloc (5); // create VEGAS workstation
    gsl_monte_vegas_integrate (&F, xl, xu, 5, 10000, r, w, &result, &error); //warm-up
    gsl_monte_vegas_integrate (&F, xl, xu, 5, 100000, r, w, &result, &error); //calculate
    delete [] params;
    gsl_monte_vegas_free(w);
    return result * dRdp1dp2_1S_decay_prefactor * gamma_inv;
}

// define some functions used in the sampling of p1 and p2
double f_p1(double p1, void * params_){
    double * params = static_cast<double *>(params_);
    double k1 = params[0];  // k1 = gamma*(1-v)/T
    double k2 = params[1];  // k2 = gamma*(1+v)/T
    return fac2(k1*p1) - fac2(k2*p1);
}
// integrated f_p1
double I_f_p1(double p1max, void * params_){
    double * params = static_cast<double *>(params_);
    double k1 = params[0];  // k1 = gamma*(1-v)/T
    double k2 = params[1];  // k2 = gamma*(1+v)/T
    double coeff1 = params[2]; // coeff1 = 1+1/v    the T^2 does not matter when taking ratios
    double coeff2 = params[3]; // coeff2 = -1+1/v
    return coeff1*Li2(-std::exp(-k1*p1max)) - coeff2*Li2(-std::exp(-k2*p1max));
}

// used in importance sampling of p1
double f_p1_decay_important(double p1, void * params_){
    double * params = static_cast<double *>(params_);
    double k1 = params[0];  // k1 = gamma*(1-v)/T
    double k2 = params[1];  // k2 = gamma*(1+v)/T
    double p2 = p1-E1S;
    if (p2 <= 0){
        return 0.0;
    }
    else{
        double p1p2 = p1/p2;
        return ( fac2(k1*p1) - fac2(k2*p1) ) * p1 /(p1p2 + 1./p1p2 -2.0);
    }
}

double S1S_decay_ineq_p1(double p1low, double p1up, void * params_){
    double * params = static_cast<double *>(params_);
    double y_try = y_cdf(gen);
    double y_result = y_try*I_f_p1(p1up, params) + (1.-y_try)*I_f_p1(p1low, params);
    double p1_try;
    p1_try = find_root(&I_f_p1, y_result, params, p1low, p1up);
    return p1_try;
}
// importance sampling of p1 works much faster than the above inverse function method, considering the overall efficiency;
// if use inverse function method, p1 sampling is easy, but the remaining integrand is a quadratic, divergent at large p1, efficiency of rejection is very low;
double S1S_decay_ineq_p1_important(double p1low, double p1up, void * params_){
    double * params = static_cast<double *>(params_);
    double result_max = find_max(&f_p1_decay_important, params, p1low, p1up);
    double result_try, p1_try;
    do{
        p1_try = sample_inel(gen)*(p1up-p1low) + p1low;
        result_try = f_p1_decay_important(p1_try, params);
    } while(rejection(gen)*result_max > result_try);
    return p1_try;
}

double S1S_decay_ineq_cos1(double p1, void * params_){
    double * params = static_cast<double *>(params_);
    double y_try = y_cdf(gen);
    double v = params[0];
    double B = params[1]*p1;    // B = gamma * p1/T
    double C = y_try * fac2(B*(1.+v)) + (1.-y_try) * fac2(B*(1.-v));
    double x_try = -(1. + std::log(std::exp(C) - 1.)/B )/v;
    return x_try;
}


std::vector<double> S1S_decay_ineq(double v, double T){
    v = std::max(v, small_number);
    double gamma = 1./std::sqrt(1.-v*v);
    double p1_try, c1_try, s1_try, p2_try, c2_try, s2_try, phi_try, c_phi, s_phi, p_rel, p1p2, part_angle, result_try, p1p2_try;
    double p1low = E1S;
    double p1up = 18.*T/std::sqrt(1.-v);
    
    double * params_p1 = new double[2];
    params_p1[0] = gamma*(1.-v)/T; //k1 = gamma*(1-v)/T
    params_p1[1] = gamma*(1.+v)/T; //k2 = gamma*(1+v)/T
    
    double * params_c1 = new double[2];
    params_c1[0] = v;
    params_c1[1] = gamma/T;
    
    do{
        do{
            p_rel = sample_inel(gen)*3.0;
        } while(rejection(gen)*max_p2Matrix > p2Matrix1S(p_rel));
        
        p1_try = S1S_decay_ineq_p1_important(p1low, p1up, params_p1);
        p2_try = p1_try - E1S - p_rel*p_rel/M;
        if (p2_try <= 0.0){
            result_try = 0.0;
        }
        else{
            c1_try = S1S_decay_ineq_cos1(p1_try, params_c1);
            c2_try = sample_cos(gen);
            s1_try = std::sqrt(1.-c1_try*c1_try);
            s2_try = std::sqrt(1.-c2_try*c2_try);
            phi_try = sample_inel(gen)*TwoPi;
            p1p2_try = p1_try/p2_try;   // p1_try/p2_try
            p1p2 = (p1_try-E1S)/p1_try;
            c_phi = std::cos(phi_try);
            s_phi = std::sin(phi_try);
            part_angle = s1_try*s2_try*c_phi + c1_try*c2_try;
            result_try = (1.+part_angle)/(p1p2 + 1./p1p2 - 2.*part_angle)/2.0*(p1p2 + 1./p1p2 - 2.0) * nFminus1(gamma*(1.+v*c2_try)*p2_try/T)/p1p2_try;
        }
    } while(rejection(gen) > result_try);
    std::vector<double> p1_final(3);
    std::vector<double> p2_final(3);
    std::vector<double> p_rel_final(3);
    std::vector<double> pQpQbar_final(6);
    double cos_rel, phi_rel;
    cos_rel = sample_cos(gen);
    phi_rel = sample_inel(gen)*TwoPi;
    p1_final = polar_to_cartisian2(p1_try, c1_try, s1_try, 1.0, 0.0);
    p2_final = polar_to_cartisian2(p2_try, c2_try, s2_try, c_phi, s_phi);
    p_rel_final = polar_to_cartisian1(p_rel, cos_rel, phi_rel);
    pQpQbar_final = add_virtual_gluon(p1_final, p2_final, p_rel_final);
    return pQpQbar_final;
}


std::vector<double> S1S_decay_ineq_test(double v, double T){
    v = std::max(v, small_number);
    double gamma = 1./std::sqrt(1.-v*v);
    double p1_try, c1_try, s1_try, p2_try, c2_try, s2_try, phi_try, c_phi, s_phi, p_rel, p1p2, part_angle, result_try, p1p2_try;
    double p1low = E1S;
    double p1up = 18.*T/std::sqrt(1.-v);

    double * params_p1 = new double[2];
    params_p1[0] = gamma*(1.-v)/T; //k1 = gamma*(1-v)/T
    params_p1[1] = gamma*(1.+v)/T; //k2 = gamma*(1+v)/T

    double * params_c1 = new double[2];
    params_c1[0] = v;
    params_c1[1] = gamma/T;

    do{
        do{
            p_rel = sample_inel(gen)*3.0;
        } while(rejection(gen)*max_p2Matrix > p2Matrix1S(p_rel));

        p1_try = S1S_decay_ineq_p1_important(p1low, p1up, params_p1);
        p2_try = p1_try - E1S - p_rel*p_rel/M;
        if (p2_try <= 0.0){
            result_try = 0.0;
        }
        else{
        c1_try = S1S_decay_ineq_cos1(p1_try, params_c1);
        c2_try = sample_cos(gen);
        s1_try = std::sqrt(1.-c1_try*c1_try);
        s2_try = std::sqrt(1.-c2_try*c2_try);
        phi_try = sample_inel(gen)*TwoPi;
        p1p2_try = p1_try/p2_try;   // p1_try/p2_try
        p1p2 = (p1_try-E1S)/p1_try;
        c_phi = std::cos(phi_try);
        s_phi = std::sin(phi_try);
        part_angle = s1_try*s2_try*c_phi + c1_try*c2_try;
        result_try = (1.+part_angle)/(p1p2_try + 1./p1p2_try - 2.*part_angle)/2.0*(p1p2 + 1./p1p2 - 2.0) * nFminus1(gamma*(1.+v*c2_try)*p2_try/T)/p1p2_try;
        }
    } while(rejection(gen) > result_try);

    std::vector<double> p1p2_test(5);
    p1p2_test[0] = p1_try;
    p1p2_test[1] = c1_try;
    p1p2_test[2] = p2_try;
    p1p2_test[3] = c2_try;
    p1p2_test[4] = phi_try;
    return p1p2_test;
}
//------------------------ end of inelastic quark dissociation -----------------------


//-------------------------- end of Quarkonium dissociation -----------------------



// ----------------------- Quark and anti-quark recombination ------------------------------
//---------------------- gluon induced recombination -----------------------
// we define rate * vol (RV) = v_rel * cross section in GeV*fm^3; p is the QQbar relative momentum
// rate * vol for a specific color, no summation over colors
double RV1S_reco_gluon(double v, double T, double p){
    double q = p*p/M + E1S;
    double reco = RV1S_prefactor * pow(q,3) * Matrix1S(p)* pow(InverseFermiToGeV,3);
    // convert GeV^-2 to GeV * fm^3
    if (v < small_number){
        double enhencement = nBplus1(q/T);
        return 2.* reco * enhencement;
    }
    else{
        double gamma = 1./std::sqrt(1.-v*v);
        double k1 = gamma*(1.+v)/T, k2 = gamma*(1.-v)/T;
        double enhencement = 2. + T/(gamma*q*v)*( fac1(q*k1) - fac1(q*k2) );
        return reco * enhencement;
    }
}

// no factor of 2 here, need to add factor 2 when determining theta function
double dist_position(double r){
    double sigma = a_B * InverseFermiToGeV;
    return std::exp( -r*r/(2.0*sigma*sigma) )/( pow(TwoPi*sigma*sigma, 1.5) );
}

// now sampling
double S1S_reco_gluon_q(double p){
    double q = p*p/M + E1S;
    return q;
}

// input the above q into the costheta sampling
double S1S_reco_gluon_costheta(double v, double T, double q){
    double gamma = 1./std::sqrt(1.-v*v);
    double y1 = q*gamma*(1.-v)/T;
    double max_value = nBplus1(y1);
    double x_try, y_try, result;
    do {
        x_try = reco_uniform(gen);
        y_try = q*gamma*(1.+ x_try*v)/T;
        result = nBplus1(y_try);
    } while (rejection(gen)*max_value > result);
    return x_try;
}

std::vector<double> S1S_reco_gluon(double v, double T, double p){
    double q = S1S_reco_gluon_q(p);
    double cos = S1S_reco_gluon_costheta(v, T, q);
    double phi = sample_inel(gen)*TwoPi;
    std::vector<double> p1S_final(3);
    p1S_final = polar_to_cartisian1(q, cos, phi);
    p1S_final = subtract_real_gluon( p1S_final );
    return p1S_final;
}
//---------------------- end of gluon induced recombination -----------------------


//---------------------- inelastic quark recombination -----------------------
// define the integrand for inelastic quark reco
double dRdp1dp2_1S_reco_ineq(double x[4], size_t dim, void * params_){
    double p1 = x[0];
    double c1 = x[1];
    double c2 = x[2];
    double phi = x[3];
    double * params = static_cast<double *>(params_);
    double v = params[0];
    double T = params[1];
    double p = params[2];
    double gamma = 1./std::sqrt(1.-v*v);
    double p2 = p1 + p*p/M + E1S;
    double s1 = std::sqrt(1.-c1*c1);
    double s2 = std::sqrt(1.-c2*c2);
    double phase1 = p1*nF(gamma*p1*(1.+v*c1)/T);
    double phase2 = p2*(1.-nF(gamma*p2*(1.+v*c2)/T));
    double part_angle = s1*s2*std::cos(phi)+c1*c2;
    double prop = p1*p2*(1.+part_angle)/( p1*p1+p2*p2-2.*p1*p2*part_angle );
    return phase1 * phase2 * Matrix1S(p) * prop;
}

// integrate the integrand to get inelastic quark reco rate
double RV1S_reco_ineq(double v, double T, double p){
    v = std::max(v, small_number);
    double * params = new double[3];
    params[0] = v;
    params[1] = T;
    params[2] = p;
    double result, error;
    double p1up = 18.*T/std::sqrt(1.-v);
    double xl[4] = { 0., -1., -1., 0. };
    double xu[4] = { p1up, 1., 1., TwoPi };
    gsl_monte_function F;
    F.f = &dRdp1dp2_1S_reco_ineq;
    F.dim = 4;
    F.params = params;
    gsl_rng * r = gsl_rng_alloc (gsl_rng_default);
    gsl_monte_vegas_state *w = gsl_monte_vegas_alloc (4); // create VEGAS workstation
    gsl_monte_vegas_integrate (&F, xl, xu, 4, 10000, r, w, &result, &error); //warm-up
    gsl_monte_vegas_integrate (&F, xl, xu, 4, 50000, r, w, &result, &error); //calculate
    delete [] params;
    gsl_monte_vegas_free(w);
    return result * dRdp1dp2_1S_reco_prefactor; // no gamma here
}

// now sampling
// used in importance sampling of p1
double f_p1_reco_important(double p1, void * params_){
    double * params = static_cast<double *>(params_);
    double k1 = params[0];  // k1 = gamma*(1-v)/T
    double k2 = params[1];  // k2 = gamma*(1+v)/T
    double E_rel = params[2];   // E_rel = p_rel^2/M
    double p2 = p1+E1S+E_rel;
    double p1p2 = p1/p2;
    return ( fac2(k1*p1) - fac2(k2*p1) ) * p2 /(p1p2 + 1./p1p2 -2.0);
}

// importance sampling of p1 works much faster than the inverse function method, considering the overall efficiency;
// if use inverse function method, p1 sampling is easy, but the remaining integrand is a quadratic, divergent at large p1, efficiency of rejection is very low;
double S1S_reco_ineq_p1_important(double p1low, double p1up, void * params_){
    double * params = static_cast<double *>(params_);
    double result_max = find_max(&f_p1_reco_important, params, p1low, p1up);
    double result_try, p1_try;
    do{
        p1_try = sample_inel(gen)*(p1up-p1low) + p1low;
        result_try = f_p1_reco_important(p1_try, params);
    } while(rejection(gen)*result_max > result_try);
    return p1_try;
}

std::vector<double> S1S_reco_ineq(double v, double T, double p){
    v = std::max(v, small_number);
    double gamma = 1./std::sqrt(1.-v*v);
    double p1_try, c1_try, s1_try, p2_try, c2_try, s2_try, phi_try, c_phi, s_phi, p1p2_try, part_angle, result_try;
    double p1low = 0.0;
    double p1up = 18.*T/std::sqrt(1.-v);
    double E_rel = p*p/M;
    
    double * params_p1 = new double[3];
    params_p1[0] = gamma*(1.-v)/T; //k1 = gamma*(1-v)/T
    params_p1[1] = gamma*(1.+v)/T; //k2 = gamma*(1+v)/T
    params_p1[2] = E_rel;   // E_rel = p^2/M
    
    double * params_c1 = new double[2];
    params_c1[0] = v;
    params_c1[1] = gamma/T;
    
    do{
        p1_try = S1S_reco_ineq_p1_important(p1low, p1up, params_p1);
        c1_try = S1S_decay_ineq_cos1(p1_try, params_c1);
        p2_try = p1_try + E_rel + E1S;
        c2_try = sample_cos(gen);
        s1_try = std::sqrt(1.-c1_try*c1_try);
        s2_try = std::sqrt(1.-c2_try*c2_try);
        phi_try = sample_inel(gen)*TwoPi;
        p1p2_try = p1_try/p2_try;   // p1/p2
        c_phi = std::cos(phi_try);
        s_phi = std::sin(phi_try);
        part_angle = s1_try*s2_try*c_phi + c1_try*c2_try;
        result_try = (1.+part_angle)/(p1p2_try + 1./p1p2_try - 2.*part_angle)/2.0 * (p1p2_try + 1./p1p2_try - 2.0) * nFminus1(gamma*(1.+v*c2_try)*p2_try/T);
    } while(rejection(gen) > result_try);
    std::vector<double> p1_final(3);
    std::vector<double> p2_final(3);
    std::vector<double> p1S_final(3);
    p1_final = polar_to_cartisian2(p1_try, c1_try, s1_try, 1.0, 0.0);
    p2_final = polar_to_cartisian2(p2_try, c2_try, s2_try, c_phi, s_phi);
    p1S_final = subtract_virtual_gluon(p1_final, p2_final);
    return p1S_final;
}

std::vector<double> S1S_reco_ineq_test(double v, double T, double p){
    v = std::max(v, small_number);
    double gamma = 1./std::sqrt(1.-v*v);
    double p1_try, c1_try, s1_try, p2_try, c2_try, s2_try, phi_try, c_phi, s_phi, p1p2_try, part_angle, result_try;
    double p1low = 0.0;
    double p1up = 18.*T/std::sqrt(1.-v);
    double E_rel = p*p/M;
    
    double * params_p1 = new double[3];
    params_p1[0] = gamma*(1.-v)/T; //k1 = gamma*(1-v)/T
    params_p1[1] = gamma*(1.+v)/T; //k2 = gamma*(1+v)/T
    params_p1[2] = E_rel;   // E_rel = p^2/M
    
    double * params_c1 = new double[2];
    params_c1[0] = v;
    params_c1[1] = gamma/T;
    
    do{
        p1_try = S1S_reco_ineq_p1_important(p1low, p1up, params_p1);
        c1_try = S1S_decay_ineq_cos1(p1_try, params_c1);
        p2_try = p1_try + E_rel + E1S;
        c2_try = sample_cos(gen);
        s1_try = std::sqrt(1.-c1_try*c1_try);
        s2_try = std::sqrt(1.-c2_try*c2_try);
        phi_try = sample_inel(gen)*TwoPi;
        p1p2_try = p1_try/p2_try;   // p1/p2
        c_phi = std::cos(phi_try);
        s_phi = std::sin(phi_try);
        part_angle = s1_try*s2_try*c_phi + c1_try*c2_try;
        result_try = (1.+part_angle)/(p1p2_try + 1./p1p2_try - 2.*part_angle)/2.0 * (p1p2_try + 1./p1p2_try - 2.0) * nFminus1(gamma*(1.+v*c2_try)*p2_try/T);
    } while(rejection(gen) > result_try);
    std::vector<double> p1p2_test(4);
    p1p2_test[0] = p1_try;
    p1p2_test[1] = c1_try;
    p1p2_test[2] = c2_try;
    p1p2_test[3] = phi_try;
    return p1p2_test;
}
//---------------------- end of inelastic quark recombination -----------------------




// ---------------------------- end of recombination ------------------------------


// convert 3-momentum to 4-momentum for Q, Qbar
std::vector<double> p3top4_Q(std::vector<double> p3){
    std::vector<double> p4(4);
    p4[0] = std::sqrt(M*M + p3[0]*p3[0] + p3[1]*p3[1] + p3[2]*p3[2]);
    p4[1] = p3[0];
    p4[2] = p3[1];
    p4[3] = p3[2];
    return p4;
}

// convert 3-momentum to 4-momentum for quarkonia
std::vector<double> p3top4_quarkonia(std::vector<double> p3){
    std::vector<double> p4(4);
    p4[0] = std::sqrt(M1S*M1S + p3[0]*p3[0] + p3[1]*p3[1] + p3[2]*p3[2]);
    p4[1] = p3[0];
    p4[2] = p3[1];
    p4[3] = p3[2];
    return p4;
}







