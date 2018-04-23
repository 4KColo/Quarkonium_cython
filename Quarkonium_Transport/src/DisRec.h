#ifndef DISREC_H        // check if DISREC_H has been defined, if not, then define
#define DISREC_H

#include "utility.h"
#include <cmath>
#include <vector>

const double rho_c_sq = pow(rho_c, 2);
const double small_number = 1e-4;
const double TwoPi = 2.*M_PI;
const double alpha_s_sqd = fix_alpha_s*fix_alpha_s;
// ------------------------- 1S: ---------------------------
const double Matrix1S_prefactor = pow(2., 9) * pow(M_PI, 2) * pow(a_B, 5) * pow(2.+rho_c, 2);

const double Matrix1S_scale = pot_alpha_s*M/(4.*Nc);

const double Xsec1S_prefactor = fix_alpha_s*CF/3.0 * pow(2.0,10)
							* pow(M_PI, 2) * rho_c*pow(2.0+rho_c, 2)
							* pow(E1S, 4) / M;

const double Xsec1S_v2_prefactor = 2./3.*fix_alpha_s*CF*M;

const double RV1S_prefactor = 2.*fix_alpha_s * TF / (3.*Nc); //gluo-recombination prefactor

const double dRdp1dp2_1S_decay_prefactor = alpha_s_sqd*4./9./pow(M_PI,4);
// above line: for inelastic quark dissociation, spin*2, color*3, antiparticle*2, flavor*2
const double dRdp1dp2_1S_reco_prefactor = alpha_s_sqd/9./M_PI/M_PI;
// above line has 1/(Nc^2-1) * 2 * 3 * 2 * 2 = 3
const double dRdq1dq2_1S_decay_prefactor = alpha_s_sqd/6./pow(M_PI,4);
// above line: gluon inelastic decay, sum over gluon color and polarizations
const double dRdq1dq2_1S_reco_prefactor = alpha_s_sqd/24./M_PI/M_PI;
// above line: gluon inelastic reco

//const double max_p2Matrix1S = 88.0; // used in rejection method, for pot_alpha_s = 0.3; 37.5 <-> 0.4
const double p_1Ssam = 3.5/a_B;    // used to find max(p2Matrix1S), sample p2Matrix1S

// ------------------------- 2S: ---------------------------
const double Matrix2S_prefactor = pow(2., 16) * pow(M_PI, 2) * pow(a_B, 5);

const double Matrix2S_term = 2. * (2.*rho_c_sq + 5.*rho_c + 3.);

//const double max_p2Matrix2S = 4163.4; // used in rejection method, for pot_alpha_s = 0.3; 1760.0 <-> 0.4
const double q_2Scrit = E2S*Matrix2S_term/(2.0+rho_c);
const double p_2Scrit = std::sqrt( M*(q_2Scrit - E2S) );    // used to find max(p2Matrix2S)
const double p_2Ssam = 1.5/a_B;    // used to find sample p2Matrix2S

double find_max(double(*f)(double x, void * params), void * params, double xL, double xR);
double find_root(double(*f)(double x, void * params), double result, void * params, double xL_, double xR_);

// --------------------- 1S -----------------------
// matrix elements
double Matrix1S(double p);
double pMatrix1S(double p);
double p2Matrix1S(double p);
double Xsec1S(double q);
double Xsec1S_v2(double q);
// real gluon dissociation
double dRdq_1S_gluon(double q, void * params_);
double dRdq_1S_gluon_small_v(double q, void * params_);
double qdRdq_1S_gluon_u(double u, void * params_);
double R1S_decay_gluon(double vabs, double T);
double S1S_decay_gluon_q(double v, double T, double maximum);
double S1S_decay_gluon_costheta(double q, double v, double T);
double S1S_decay_gluon_final_p(double q);
std::vector<double> S1S_decay_gluon(double v, double T, double maximum);
// inelastic quark dossociation
double dRdp1dp2_1S_decay_ineq(double x[5], size_t dim, void * params_);
double R1S_decay_ineq(double v, double T);
double f_p1(double p1, void * params_);
double I_f_p1(double p1max, void * params_);
double f_p1_decay1S_important(double p1, void * params_);
double S1S_decay_ineq_p1(double p1low, double p1up, void * params_);
double S1S_decay_ineq_p1_important(double p1low, double p1up, double result_max, void * params_);
double S1S_decay_ineq_cos1(double p1, void * params_);
std::vector<double> S1S_decay_ineq(double v, double T, double maximum);
std::vector<double> S1S_decay_ineq_test(double v, double T, double maximum);
// inelastic gluon dissociation
double dRdq1dq2_1S_decay_ineg(double x[5], size_t dim, void * params_);
double R1S_decay_ineg(double v, double T);
double f_q1_decay1S_important(double q1, void * params_);
double S1S_decay_ineg_q1_important(double q1low, double q1up, double result_max, void * params_);
double S1S_decay_ineg_cos1(double q1, void * params_);
std::vector<double> S1S_decay_ineg(double v, double T, double maximum);
std::vector<double> S1S_decay_ineg_test(double v, double T, double maximum);

// real gluon recombination
double RV1S_reco_gluon(double v, double T, double p);
double dist_position_1S(double r);
double S1S_reco_gluon_q(double p);
double S1S_reco_gluon_costheta(double v, double T, double q);
std::vector<double> S1S_reco_gluon(double v, double T, double p);
// inelastic quark recombination
double dRdp1dp2_1S_reco_ineq(double x[4], size_t dim, void * params_);
double RV1S_reco_ineq(double v, double T, double p);
double f_p1_reco1S_important(double p1, void * params_);
double S1S_reco_ineq_p1_important(double p1low, double p1up, double result_max, void * params_);
std::vector<double> S1S_reco_ineq(double v, double T, double p, double maximum);
std::vector<double> S1S_reco_ineq_test(double v, double T, double p, double maximum);
// inelastic gluon recombination
double dRdq1dq2_1S_reco_ineg(double x[4], size_t dim, void * params_);
double RV1S_reco_ineg(double v, double T, double p);
double f_q1_reco1S_important(double q1, void * params_);
double S1S_reco_ineg_q1_important(double q1low, double q1up, double result_max, void * params_);
std::vector<double> S1S_reco_ineg(double v, double T, double p, double maximum);
std::vector<double> S1S_reco_ineg_test(double v, double T, double p, double maximum);


// --------------------- 2S -----------------------
// matrix elements
double Matrix2S(double p);
double pMatrix2S(double p);
double p2Matrix2S(double p);
double Xsec2S(double q);
// real gluon dissociation
double dRdq_2S_gluon(double q, void * params_);
double dRdq_2S_gluon_small_v(double q, void * params_);
double qdRdq_2S_gluon_u(double u, void * params_);
double R2S_decay_gluon(double vabs, double T);
double S2S_decay_gluon_q(double v, double T, double maximum);
double S2S_decay_gluon_costheta(double q, double v, double T);
double S2S_decay_gluon_final_p(double q);
std::vector<double> S2S_decay_gluon(double v, double T, double maximum);
// inelastic quark dossociation
double dRdp1dp2_2S_decay_ineq(double x[5], size_t dim, void * params_);
double R2S_decay_ineq(double v, double T);
double f_p1_decay2S_important(double p1, void * params_);
double S2S_decay_ineq_p1(double p1low, double p1up, void * params_);
double S2S_decay_ineq_p1_important(double p1low, double p1up, double result_max, void * params_);
double S2S_decay_ineq_cos1(double p1, void * params_);
std::vector<double> S2S_decay_ineq(double v, double T, double maximum);
std::vector<double> S2S_decay_ineq_test(double v, double T, double maximum);
// inelastic gluon dissociation
double dRdq1dq2_2S_decay_ineg(double x[5], size_t dim, void * params_);
double R2S_decay_ineg(double v, double T);
double f_q1_decay2S_important(double q1, void * params_);
double S2S_decay_ineg_q1_important(double q1low, double q1up, double result_max, void * params_);
double S2S_decay_ineg_cos1(double q1, void * params_);
std::vector<double> S2S_decay_ineg(double v, double T, double maximum);
std::vector<double> S2S_decay_ineg_test(double v, double T, double maximum);

// real gluon recombination
double RV2S_reco_gluon(double v, double T, double p);
double dist_position_2S(double r);
double S2S_reco_gluon_q(double p);
double S2S_reco_gluon_costheta(double v, double T, double q);
std::vector<double> S2S_reco_gluon(double v, double T, double p);
// inelastic quark recombination
double dRdp1dp2_2S_reco_ineq(double x[4], size_t dim, void * params_);
double RV2S_reco_ineq(double v, double T, double p);
double f_p1_reco2S_important(double p1, void * params_);
double S2S_reco_ineq_p1_important(double p1low, double p1up, double result_max, void * params_);
std::vector<double> S2S_reco_ineq(double v, double T, double p, double maximum);
std::vector<double> S2S_reco_ineq_test(double v, double T, double p, double maximum);
// inelastic gluon dissociation
double dRdq1dq2_2S_reco_ineg(double x[4], size_t dim, void * params_);
double RV2S_reco_ineg(double v, double T, double p);
double f_q1_reco2S_important(double q1, void * params_);
double S2S_reco_ineg_q1_important(double q1low, double q1up, double result_max, void * params_);
std::vector<double> S2S_reco_ineg(double v, double T, double p, double maximum);
std::vector<double> S2S_reco_ineg_test(double v, double T, double p, double maximum);


// change polar coordinates to cartisian coordinates
std::vector<double> p3top4_Q(std::vector<double> p3);
std::vector<double> p3top4_quarkonia_1S(std::vector<double> p3);
std::vector<double> p3top4_quarkonia_2S(std::vector<double> p3);

// define these functions in .h file so that cython can call; for what these functions really do, the compiler will refer to cpp(cxx) file
#endif
