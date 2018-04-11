#include <iostream>
#include <cmath>
#include "TransLorentzRotate.h"
#include <vector>


// ---------------------------- lorentz transformation ----------------------------
std::vector<double> lorentz_transform(std::vector<double> momentum_4, std::vector<double> velocity_3){
    double vx = velocity_3[0];
    double vy = velocity_3[1];
    double vz = velocity_3[2];
    double v_sq = vx*vx + vy*vy + vz*vz;
    
    if (v_sq == 0.0){
        return momentum_4;
    }
    else{
        double E = momentum_4[0];
        double px = momentum_4[1];
        double py = momentum_4[2];
        double pz = momentum_4[3];
        
        double gamma = 1./std::sqrt(1.-v_sq);
        double vdotp = vx*px + vy*py + vz*pz;
        
        std::vector<double> p4_new(4);
        p4_new[0] = gamma*(E - vdotp);
        p4_new[1] = -gamma*vx*E + px + (gamma-1.)*vx*vdotp/v_sq;
        p4_new[2] = -gamma*vy*E + py + (gamma-1.)*vy*vdotp/v_sq;
        p4_new[3] = -gamma*vz*E + pz + (gamma-1.)*vz*vdotp/v_sq;
        return p4_new;
    }
}

// ------------------------ end of lorentz transformation ----------------------------



// ---------------------------- rotation transformation ------------------------------
std::vector<double> rotation_transform3(std::vector<double> vector_3, double theta, double phi){
    double vx = vector_3[0];
    double vy = vector_3[1];
    double vz = vector_3[2];
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    double cos_phi = std::cos(phi);
    double sin_phi = std::sin(phi);
    std::vector<double> v3_new(3);
    v3_new[0] = cos_theta*cos_phi*vx - sin_phi*vy + sin_theta*cos_phi*vz;
    v3_new[1] = cos_theta*sin_phi*vx + cos_phi*vy + sin_theta*sin_phi*vz;
    v3_new[2] = -sin_theta*vx + cos_theta*vz;
    return v3_new;
}

std::vector<double> rotation_transform4(std::vector<double> vector_4, double theta, double phi){
    double vx = vector_4[1];
    double vy = vector_4[2];
    double vz = vector_4[3];
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    double cos_phi = std::cos(phi);
    double sin_phi = std::sin(phi);
    std::vector<double> v4_new(4);
    v4_new[0] = vector_4[0];
    v4_new[1] = cos_theta*cos_phi*vx - sin_phi*vy + sin_theta*cos_phi*vz;
    v4_new[2] = cos_theta*sin_phi*vx + cos_phi*vy + sin_theta*sin_phi*vz;
    v4_new[3] = -sin_theta*vx + cos_theta*vz;
    return v4_new;
}

// only works for 4-vector A, rotate back from D as z axis to original D
std::vector<double> rotate_by_Dinv(std::vector<double> A, double Dx, double Dy, double Dz){
    std::vector<double> Ap(4);
    double Dperp = std::sqrt(Dx*Dx + Dy*Dy);
    double D = std::sqrt(Dperp*Dperp + Dz*Dz);
    if (Dperp/D < 1e-10){
        Ap[0] = A[0];
        Ap[1] = A[1];
        Ap[2] = A[2];
        Ap[3] = A[3];
    }
    else{
        double c2 = Dz/D, s2 = Dperp/D;
        double c3 = Dx/Dperp, s3 = Dy/Dperp;
        Ap[0] = A[0];
        Ap[1] = -s3*A[2] + c3*(c2*A[1] + s2*A[3]);
        Ap[2] = c3*A[2]  + s3*(c2*A[1] + s2*A[3]);
        Ap[3] =          - s2*A[1]     + c2*A[3];
    }
    return Ap;
}
// ------------------------ end of rotation transformation ---------------------------


// ------------------------ find Prel in CM frame ---------------------------
std::vector<double> find_vCM_prel(std::vector<double> pQ, std::vector<double> pQbar, double mass){
    std::vector<double> p3_CM(3), v3_CM(3), pQ_CM(4), pQbar_CM(4), p_rel(3);
    double v3_CM_abs, p_sqd, E_CM, p_rel_abs;
    p3_CM[0] = pQ[1] + pQbar[1];
    p3_CM[1] = pQ[2] + pQbar[2];
    p3_CM[2] = pQ[3] + pQbar[3];
    p_sqd = p3_CM[0]*p3_CM[0] + p3_CM[1]*p3_CM[1] + p3_CM[2]*p3_CM[2];
    E_CM = std::sqrt(p_sqd + mass * mass);
    v3_CM[0] = p3_CM[0]/E_CM;
    v3_CM[1] = p3_CM[1]/E_CM;
    v3_CM[2] = p3_CM[2]/E_CM;
    v3_CM_abs = std::sqrt(p_sqd)/E_CM;      // CM velocity
    pQ_CM = lorentz_transform(pQ, v3_CM);   // pQ in CM frame
    pQbar_CM = lorentz_transform(pQbar, v3_CM); // pQbar in CM frame
    p_rel[0] = 0.5*(pQ_CM[1] - pQbar_CM[1]);
    p_rel[1] = 0.5*(pQ_CM[2] - pQbar_CM[2]);
    p_rel[2] = 0.5*(pQ_CM[3] - pQbar_CM[3]);
    p_rel_abs = std::sqrt(p_rel[0]*p_rel[0] + p_rel[1]*p_rel[1] + p_rel[2]*p_rel[2]);
    std::vector<double> output(5);
    output[0] = v3_CM[0];
    output[1] = v3_CM[1];
    output[2] = v3_CM[2];
    output[3] = v3_CM_abs;
    output[4] = p_rel_abs;
    return output;
}



// ------------------------ find theta and phi of a vector ---------------------------
std::vector<double> angle_find(std::vector<double> vector_3){
    double vx = vector_3[0];
    double vy = vector_3[1];
    double vz = vector_3[2];
    double vabs = std::sqrt(vx*vx + vy*vy + vz*vz);
    
    std::vector<double> theta_phi(2);
    if (vabs==0.){
        theta_phi[0] = 0.;
    }
    else{
        theta_phi[0] = std::acos(vz/vabs);
    }
    if (vx > 0.){
        theta_phi[1] = std::atan(vy/vx);
    }
    else if (vx == 0.){
        if (vy > 0.){
            theta_phi[1] = 0.5*M_PI;
        }
        else if (vy == 0.){
            theta_phi[1] = 0.;
        }
        else{
            theta_phi[1] = -0.5*M_PI;
        }
    }
    else{
        theta_phi[1] = std::atan(vy/vx) + M_PI;
    }
    return theta_phi;
}

// ---------------------- end of find theta and phi of a vector ----------------------

