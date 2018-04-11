# cython: c_string_type=str, c_string_encoding=ascii
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.math cimport *
import cython, os, h5py
import numpy as np
cimport numpy as np

#### ----------- Import C++ fucntions and class for Xsection and rates------------------
cdef extern from "../src/TransLorentzRotate.h":
	# lorentz_transform
	cdef vector[double] lorentz_transform(vector[double] momentum_4, vector[double] velocity_3)
	cdef vector[double] rotation_transform3(vector[double] vector_3, double theta, double phi)
	cdef vector[double] rotation_transform4(vector[double] vector_4, double theta, double phi)	
	cdef vector[double] angle_find(vector[double] vector_3)
	cdef vector[double] find_vCM_prel(vector[double] pQ, vector[double] pQbar, double mass)
	cdef vector[double] rotate_by_Dinv(vector[double] A, double Dx, double Dy, double Dz)

#### ----------- Provide a direct python wrapper for these functions for tests

## ----------------- Lorentz transformation ---------------- ##
def lorentz(np.ndarray[double, ndim=1] p4, 
			np.ndarray[double, ndim=1] v3):
	return np.array(lorentz_transform(p4, v3))
	
## --------------- end of Lorentz transformation ------------ ##


## --------------- Rotation ---------------- ##
def rotation3(np.ndarray[double, ndim=1] v3, double theta, double phi):
	return np.array(rotation_transform3(v3, theta, phi))

def rotation4(np.ndarray[double, ndim=1] v4, double theta, double phi):
	return np.array(rotation_transform4(v4, theta, phi))

def rotate_back_from_D(np.ndarray[double, ndim=1] A, double Dx, double Dy, double Dz):
	return np.array(rotate_by_Dinv(A, Dx, Dy, Dz))
## ------------ end of rotation ------------ ##


## --------------- find theta and phi of a 3-vector ------------------ ##
def angle(np.ndarray[double, ndim=1] v3):
	return np.array(angle_find(v3))

## ----------------- end of finding theta and phi -------------------- ##


## ---------------- find p_rel from pQ and pQbar --------------------- ##
def vCM_prel(np.ndarray[double, ndim=1] pQ, np.ndarray[double, ndim=1] pQbar, double mass):
	result = np.array(find_vCM_prel(pQ, pQbar, mass))
	return np.array(result[0:3]), result[3], result[4]	# v3_CM, v3_CM_abs, Prel_abs


## ------------- end of finding p_rel from pQ and pQbar -------------- ##