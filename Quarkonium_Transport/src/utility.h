#ifndef UTILITY_H
#define UTILITY_H

const double fix_alpha_s = 0.3;
const double pot_alpha_s = 0.36;
const double Nc = 3., TF = 0.5, CF = 4./3.;
const double rho_c = 1./(Nc*Nc - 1.);
const double M = 4.65; //[GeV]
const double a_B = 2./pot_alpha_s/CF/M;
const double size_2S = 2.*a_B;
const double size_1P = 2.*a_B;
const double E1S = pot_alpha_s*CF/2./a_B;
const double M1S = 2.*M - E1S;
const double E2S = E1S/4.;
const double M2S = 2.*M - E2S;
const double E1P = E1S/4.;
const double M1P = 2.*M - E1P;
const double InverseFermiToGeV = 0.197327;
const double li2_minus1 = -0.822467;
const double accuracy = 0.0001;
#endif
