#ifndef UTILITY_H
#define UTILITY_H

const double fix_alpha_s = 0.3;
const double pot_alpha_s = 0.3;
const double Nc = 3., TF = 0.5, CF = 4./3.;
const double rho_c = 1./(Nc*Nc - 1.);
const double M = 4.6500; //[GeV]
const double a_B = 2./pot_alpha_s/CF/M;
const double E1S = pot_alpha_s*CF/2./a_B;
const double M1S = 2.*M - E1S;
const double E2S = E1S/4.;
const double M2S = 2.*M - E2S;
const double InverseFermiToGeV = 0.197327;
const double li2_minus1 = -0.822467;
const double accuracy = 0.00001;

#endif
