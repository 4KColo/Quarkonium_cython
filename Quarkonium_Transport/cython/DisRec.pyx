# cython: c_string_type=str, c_string_encoding=ascii
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport *
import cython, os, h5py
import numpy as np
cimport numpy as np

#------------------Import C++ fucntions and class for Xsection and rates------------------
cdef extern from "../src/DisRec.h":
	# function related to real gluon decay rate
	cdef double R1S_decay_gluon(double vabs, double T)
	cdef double R2S_decay_gluon(double vabs, double T)

	# function related to real gluon decay sampling
	cdef double dRdq_1S_gluon(double q, void * params_)
	cdef double dRdq_1S_gluon_small_v(double q, void * params_)
	cdef double qdRdq_1S_gluon_u(double u, void * params_)
	cdef double find_max(double(*f)(double x, void * params), void * params, double xL_, double xR_)
	cdef vector[double] S1S_decay_gluon(double v, double T, double maximum)
	cdef double dRdq_2S_gluon(double q, void * params_)
	cdef double dRdq_2S_gluon_small_v(double q, void * params_)
	cdef double qdRdq_2S_gluon_u(double u, void * params_)
	cdef vector[double] S2S_decay_gluon(double v, double T, double maximum)
	
	# function related to inelastic quark decay rate
	cdef double R1S_decay_ineq(double v, double T)
	cdef double R2S_decay_ineq(double v, double T)
	
	# function related to inelastic quark decay sampling
	cdef double S1S_decay_ineq_p1(double p1low, double p1up, void * params_)	# test
	cdef double S1S_decay_ineq_cos1(double p1, void * params_)	# test
	cdef double f_p1_decay1S_important(double p1, void * params_)
	cdef double f_p1_decay2S_important(double p1, void * params_)
	cdef vector[double] S1S_decay_ineq(double v, double T, double maximum)
	cdef vector[double] S1S_decay_ineq_test(double v, double T, double maximum)
	cdef vector[double] S2S_decay_ineq(double v, double T, double maximum)
	cdef vector[double] S2S_decay_ineq_test(double v, double T, double maximum)
		
	# function related to inelastic gluon decay rate
	cdef double R1S_decay_ineg(double v, double T)
	cdef double R2S_decay_ineg(double v, double T)
	
	# function related to inelastic gluon decay sampling
	cdef double f_q1_decay1S_important(double q1, void * params_)
	cdef double f_q1_decay2S_important(double q1, void * params_)
	cdef vector[double] S1S_decay_ineg(double v, double T, double maximum)
	cdef vector[double] S1S_decay_ineg_test(double v, double T, double maximum)
	cdef vector[double] S2S_decay_ineg(double v, double T, double maximum)
	cdef vector[double] S2S_decay_ineg_test(double v, double T, double maximum)
			
	# function related to real gluon recombine rate
	cdef double RV1S_reco_gluon(double v, double T, double p)
	cdef double RV2S_reco_gluon(double v, double T, double p)
	cdef double dist_position_1S(double r)
	cdef double dist_position_2S(double r)
	
	# function related to real gluon recombine sampling
	cdef vector[double] S1S_reco_gluon(double v, double T, double p)
	cdef vector[double] S2S_reco_gluon(double v, double T, double p)
	
	# function related to inelastic quark recombine rate
	cdef double RV1S_reco_ineq(double v, double T, double p)
	cdef double RV2S_reco_ineq(double v, double T, double p)
	
	# function related to inelastic quark recombine sampling
	cdef double f_p1_reco1S_important(double p1, void * params_)
	cdef double f_p1_reco2S_important(double p1, void * params_)
	cdef vector[double] S1S_reco_ineq(double v, double T, double p, double maximum)
	cdef vector[double] S1S_reco_ineq_test(double v, double T, double p, double maximum)
	cdef vector[double] S2S_reco_ineq(double v, double T, double p, double maximum)
	cdef vector[double] S2S_reco_ineq_test(double v, double T, double p, double maximum)
		
	# function related to inelastic gluon recombine rate
	cdef double RV1S_reco_ineg(double v, double T, double p)
	cdef double RV2S_reco_ineg(double v, double T, double p)
	
	# function related to inelastic gluon recombine sampling
	cdef double f_q1_reco1S_important(double q1, void * params_)
	cdef double f_q1_reco2S_important(double q1, void * params_)
	cdef vector[double] S1S_reco_ineg(double v, double T, double p, double maximum)
	cdef vector[double] S1S_reco_ineg_test(double v, double T, double p, double maximum)
	cdef vector[double] S2S_reco_ineg(double v, double T, double p, double maximum)
	cdef vector[double] S2S_reco_ineg_test(double v, double T, double p, double maximum)
	
	# small v cut
	cdef double small_number
	
	# convert p3 to p4
	cdef vector[double] p3top4_Q(vector[double] p3)
	cdef vector[double] p3top4_quarkonia_1S(vector[double] p3)
	cdef vector[double] p3top4_quarkonia_2S(vector[double] p3)

	# matrix elements |<nS|r|Psi_p>|^2
	cpdef double Matrix1S(double p)
	cpdef double Matrix2S(double p)
	
cdef extern from "../src/utility.h":
	cdef double E1S
	cdef double E2S
	cdef double M
		
# Provide a direct python wrapper for these functions for tests
def pyE1S():
	return E1S
def pyE2S():
	return E2S
#------------------------------- rates -----------------------------
def pyR1S_decay_gluon(double vabs, double T):
	return R1S_decay_gluon(vabs, T)

def pydRdq_1S_gluon(double q, double v, double T):
	cdef double * params = <double*>malloc(2*sizeof(double))
	v = np.max([small_number, v])
	cdef double gamma = 1./sqrt(1. - v*v)
	params[0] = gamma*(1.+v)/T
	params[1] = gamma*(1.-v)/T
	return dRdq_1S_gluon(q, params)

def pyR1S_decay_ineq(double vabs, double T):
	return R1S_decay_ineq(vabs, T)

def pyR1S_decay_ineg(double vabs, double T):
	return R1S_decay_ineg(vabs, T)
		
def pyRV1S_reco_gluon(double vabs, double T, double p_rel):
	return RV1S_reco_gluon(vabs, T, p_rel)

def pyRV1S_reco_ineq(double vabs, double T, double p_rel):
	return RV1S_reco_ineq(vabs, T, p_rel)
	
def pyRV1S_reco_ineg(double vabs, double T, double p_rel):
	return RV1S_reco_ineg(vabs, T, p_rel)

def pyR2S_decay_gluon(double vabs, double T):
	return R2S_decay_gluon(vabs, T)

def pyR2S_decay_ineq(double vabs, double T):
	return R2S_decay_ineq(vabs, T)

def pyR2S_decay_ineg(double vabs, double T):
	return R2S_decay_ineg(vabs, T)
		
def pyRV2S_reco_gluon(double vabs, double T, double p_rel):
	return RV2S_reco_gluon(vabs, T, p_rel)

def pyRV2S_reco_ineq(double vabs, double T, double p_rel):
	return RV2S_reco_ineq(vabs, T, p_rel)
	
def pyRV2S_reco_ineg(double vabs, double T, double p_rel):
	return RV2S_reco_ineg(vabs, T, p_rel)
		
#---------------------------- sampling -----------------------------
def pyS1S_reco_gluon(double v, double T, double p_rel):
	p1S = np.array(S1S_reco_gluon(v, T, p_rel))
	return p1S

def pyS1S_decay_ineq_p1(double p1low, double p1up, double v, double T):
	cdef double * params = <double*>malloc(4*sizeof(double))
	v = np.max([small_number, v])
	cdef double gamma = 1./sqrt(1. - v*v)
	params[0] = gamma*(1.-v)/T
	params[1] = gamma*(1.+v)/T
	params[2] = 1.0+1.0/v
	params[3] = -1.0+1.0/v
	return S1S_decay_ineq_p1(p1low, p1up, params)

def pyS1S_decay_ineq_cos1(double p1, double v, double T):
	cdef double * params = <double*>malloc(2*sizeof(double))
	v = np.max([small_number, v])
	cdef double gamma = 1./sqrt(1. - v*v)
	params[0] = v
	params[1] = gamma/T
	return S1S_decay_ineq_cos1(p1, params)

def pyS1S_decay_ineq_test(double v, double T, double maximum):
	return S1S_decay_ineq_test(v, T, maximum)

def pyS1S_decay_ineg_test(double v, double T, double maximum):
	return S1S_decay_ineg_test(v, T, maximum)
	
def pyS1S_reco_ineq_test(double v, double T, double p, double maximum):
	return S1S_reco_ineq_test(v, T, p, maximum)

def pyS1S_reco_ineg_test(double v, double T, double p, double maximum):
	return S1S_reco_ineg_test(v, T, p, maximum)
	
pyR1S_decay_gluon = np.vectorize(pyR1S_decay_gluon)
pydRdq_1S_gluon = np.vectorize(pydRdq_1S_gluon)
pyR1S_decay_ineq = np.vectorize(pyR1S_decay_ineq)
pyR1S_decay_ineg = np.vectorize(pyR1S_decay_ineg)
pyRV1S_reco_gluon = np.vectorize(pyRV1S_reco_gluon)
pyRV1S_reco_ineq = np.vectorize(pyRV1S_reco_ineq)
pyRV1S_reco_ineg = np.vectorize(pyRV1S_reco_ineg)
pyR2S_decay_gluon = np.vectorize(pyR2S_decay_gluon)
pyR2S_decay_ineq = np.vectorize(pyR2S_decay_ineq)
pyR2S_decay_ineg = np.vectorize(pyR2S_decay_ineg)
pyRV2S_reco_gluon = np.vectorize(pyRV2S_reco_gluon)
pyRV2S_reco_ineq = np.vectorize(pyRV2S_reco_ineq)
pyRV2S_reco_ineg = np.vectorize(pyRV2S_reco_ineg)

@cython.boundscheck(False)
@cython.wraparound(False)



## --------------------- 2-d interpolation ----------------- ##
cdef double interp2d(np.ndarray[np.double_t, ndim=2] grid,
					 double x, double y, 
					 double xmin, double xmax, double dx, Nx,
					 double ymin, double ymax, double dy, Ny):
	x = np.min([np.max([x, xmin]), xmax])
	y = np.min([np.max([y, ymin]), ymax])
	cdef double rx, ry, res=0.
	cdef size_t ix, iy, i, j
	rx, ix = np.modf((x - xmin)/dx)
	ry, iy = np.modf((y - ymin)/dy)
	cdef double wx[2]
	cdef double wy[2]
	wx[0] = 1.-rx; wx[1] = rx
	wy[0] = 1.-ry; wy[1] = ry
	for i in range(2):
		for j in range(2):
			res += grid[(ix+i)%Nx, (iy+j)%Ny]*wx[i]*wy[j]
	return res


## --------------------- 3-d interpolation ----------------- ##
cdef double interp3d(np.ndarray[np.double_t, ndim=3] grid,
					 double x, double y, double z,
					 double xmin, double xmax, double dx, Nx,
					 double ymin, double ymax, double dy, Ny,
					 double zmin, double zmax, double dz, Nz):
	x = np.min([np.max([x, xmin]), xmax])
	y = np.min([np.max([y, ymin]), ymax])
	z = np.min([np.max([z, zmin]), zmax])
	cdef double rx, ry, rz, res=0.
	cdef size_t ix, iy, iz, i, j, k
	rx, ix = np.modf((x - xmin)/dx) 
	ry, iy = np.modf((y - ymin)/dy)
	rz, iz = np.modf((z - zmin)/dz)
	cdef double wx[2]
	cdef double wy[2]
	cdef double wz[2]
	wx[0] = 1.-rx; wx[1] = rx
	wy[0] = 1.-ry; wy[1] = ry
	wz[0] = 1.-rz; wz[1] = rz
	for i in range(2):
		for j in range(2):
			for k in range(2):
				res += grid[(ix+i)%Nx, (iy+j)%Ny, (iz+k)%Nz] *wx[i]*wy[j]*wz[k]
	return res



cdef class DisRec(object):
	cdef double vmin, vmax, dv
	cdef double Tmin, Tmax, dT
	cdef p_rel_log_min, p_rel_log_max, dp_rel
	cdef size_t N_v, N_T, N_p_rel
	cdef np.ndarray T_R1S_decay_gluon
	cdef np.ndarray T_R1S_decay_ineq
	cdef np.ndarray T_R1S_decay_ineg
	cdef np.ndarray T_qdRdq_1S_gluon_max
	cdef np.ndarray T_RV1S_reco_gluon
	cdef np.ndarray T_RV1S_reco_ineq
	cdef np.ndarray T_RV1S_reco_ineg
	cdef np.ndarray T_S1S_decay_ineq_max
	cdef np.ndarray T_S1S_decay_ineg_max
	cdef np.ndarray T_S1S_reco_ineq_max
	cdef np.ndarray T_S1S_reco_ineg_max
	cdef np.ndarray T_R2S_decay_gluon
	cdef np.ndarray T_R2S_decay_ineq
	cdef np.ndarray T_R2S_decay_ineg
	cdef np.ndarray T_qdRdq_2S_gluon_max
	cdef np.ndarray T_RV2S_reco_gluon
	cdef np.ndarray T_RV2S_reco_ineq
	cdef np.ndarray T_RV2S_reco_ineg
	cdef np.ndarray T_S2S_decay_ineq_max
	cdef np.ndarray T_S2S_decay_ineg_max
	cdef np.ndarray T_S2S_reco_ineq_max
	cdef np.ndarray T_S2S_reco_ineg_max
		
	def __cinit__(self, table_folder='tables', overwrite=False):
		if not os.path.exists(table_folder):
			os.mkdir(table_folder)
		fname = table_folder+'/Quarkonium-table.hdf5'
		if not os.path.exists(fname):
			f = h5py.File(fname, 'w')
		else:
			f = h5py.File(fname, 'a')
		
		gpname = 'b-system'
		cdef double params[2]
		cdef double params_decay[2]
		cdef double params_reco[3]
		if (gpname in f) and (not overwrite):
			gp = f[gpname]
			self.vmin, self.vmax, self.N_v = gp.attrs['v-min-max-N']
			self.Tmin, self.Tmax, self.N_T = gp.attrs['T-min-max-N']
			self.p_rel_log_min, self.p_rel_log_max, self.N_p_rel = gp.attrs['p_rel_log-min-max-N']
			
			print ("loading Upsilon(1S)+g <-> b+bbar rate table")
			self.T_R1S_decay_gluon = gp['R1S_decay_gluon'].value
			self.T_qdRdq_1S_gluon_max = gp['qdRdq_1S_gluon_max'].value
			self.T_RV1S_reco_gluon = gp['RV1S_reco_gluon'].value
			print ("loading Upsilon(2S)+g <-> b+bbar rate table")
			self.T_R2S_decay_gluon = gp['R2S_decay_gluon'].value
			self.T_qdRdq_2S_gluon_max = gp['qdRdq_2S_gluon_max'].value
			self.T_RV2S_reco_gluon = gp['RV2S_reco_gluon'].value
						
			print ("loading Upsilon(1S)+q <-> b+bbar+q rate table")
			self.T_R1S_decay_ineq = gp['R1S_decay_ineq'].value
			self.T_RV1S_reco_ineq = gp['RV1S_reco_ineq'].value
			self.T_S1S_decay_ineq_max = gp['S1S_decay_ineq_max'].value
			self.T_S1S_reco_ineq_max = gp['S1S_reco_ineq_max'].value
			print ("loading Upsilon(2S)+q <-> b+bbar+q rate table")
			self.T_R2S_decay_ineq = gp['R2S_decay_ineq'].value
			self.T_RV2S_reco_ineq = gp['RV2S_reco_ineq'].value
			self.T_S2S_decay_ineq_max = gp['S2S_decay_ineq_max'].value
			self.T_S2S_reco_ineq_max = gp['S2S_reco_ineq_max'].value
						
			print ("loading Upsilon(1S)+g <-> b+bbar+g rate table")
			self.T_R1S_decay_ineg = gp['R1S_decay_ineg'].value
			self.T_RV1S_reco_ineg = gp['RV1S_reco_ineg'].value
			self.T_S1S_decay_ineg_max = gp['S1S_decay_ineg_max'].value
			self.T_S1S_reco_ineg_max = gp['S1S_reco_ineg_max'].value
			print ("loading Upsilon(2S)+g <-> b+bbar+g rate table")
			self.T_R2S_decay_ineg = gp['R2S_decay_ineg'].value
			self.T_RV2S_reco_ineg = gp['RV2S_reco_ineg'].value
			self.T_S2S_decay_ineg_max = gp['S2S_decay_ineg_max'].value
			self.T_S2S_reco_ineg_max = gp['S2S_reco_ineg_max'].value
			
		else:
			if gpname in f:
				del f[gpname]
			gp = f.create_group(gpname)
			self.vmin = 0.01; self.vmax = 0.999; self.N_v = 50
			self.Tmin = 0.15; self.Tmax = 0.55; self.N_T = 30
			# here we use p_rel in GeV, the 4.0 and 8.6 are for MeV; ln(1000)=6.9
			self.p_rel_log_min = 4.0-6.9; self.p_rel_log_max = 8.6-6.9; self.N_p_rel = 80

			## Initialize dissociation rate table
			# gluon-dissociation
			print ("generating Upsilon(1S)+g -> b+bbar rate table")
			varray = np.linspace(self.vmin, self.vmax, self.N_v)
			Tarray = np.linspace(self.Tmin, self.Tmax, self.N_T)
			grid_v, grid_T = np.meshgrid(varray, Tarray)
			self.T_R1S_decay_gluon = np.transpose(pyR1S_decay_gluon(grid_v, grid_T))
			self.T_qdRdq_1S_gluon_max = np.zeros_like(self.T_R1S_decay_gluon)
			print ("generating Upsilon(2S)+g -> b+bbar rate table")
			self.T_R2S_decay_gluon = np.transpose(pyR2S_decay_gluon(grid_v, grid_T))
			self.T_qdRdq_2S_gluon_max = np.zeros_like(self.T_R2S_decay_gluon)

			# inelastic quark dissociation
			print ("generating Upsilon(1S)+q -> b+bbar+q rate table")
			self.T_R1S_decay_ineq = np.transpose(pyR1S_decay_ineq(grid_v, grid_T))
			self.T_S1S_decay_ineq_max = np.zeros_like(self.T_R1S_decay_ineq)
			print ("generating Upsilon(2S)+q -> b+bbar+q rate table")
			self.T_R2S_decay_ineq = np.transpose(pyR2S_decay_ineq(grid_v, grid_T))
			self.T_S2S_decay_ineq_max = np.zeros_like(self.T_R2S_decay_ineq)

			# inelastic gluon dissociation
			print ("generating Upsilon(1S)+g -> b+bbar+g rate table")
			self.T_R1S_decay_ineg = np.transpose(pyR1S_decay_ineg(grid_v, grid_T))
			self.T_S1S_decay_ineg_max = np.zeros_like(self.T_R1S_decay_ineg)
			print ("generating Upsilon(2S)+g -> b+bbar+g rate table")
			self.T_R2S_decay_ineg = np.transpose(pyR2S_decay_ineg(grid_v, grid_T))
			self.T_S2S_decay_ineg_max = np.zeros_like(self.T_R2S_decay_ineg)
			
			# generate the maximums for sampling
			print ("generating decay sampling")
			for iv, v in enumerate(varray):
				params[0] = v
				gamma = np.sqrt(1.-v*v)
				for iT, T in enumerate(Tarray):
					params[1] = T
					params_decay[0] = gamma*(1.-v)/T
					params_decay[1] = gamma*(1.+v)/T
					uplim = 15.0*T/np.sqrt(1.-v)
					self.T_qdRdq_1S_gluon_max[iv, iT] = find_max(&qdRdq_1S_gluon_u, params, 0., 4.)
					self.T_qdRdq_2S_gluon_max[iv, iT] = find_max(&qdRdq_2S_gluon_u, params, 0., 1.2354)
					self.T_S1S_decay_ineq_max[iv, iT] = find_max(&f_p1_decay1S_important, params_decay, E1S, uplim)
					self.T_S2S_decay_ineq_max[iv, iT] = find_max(&f_p1_decay2S_important, params_decay, E2S, uplim)
					self.T_S1S_decay_ineg_max[iv, iT] = find_max(&f_q1_decay1S_important, params_decay, E1S, uplim)
					self.T_S2S_decay_ineg_max[iv, iT] = find_max(&f_q1_decay2S_important, params_decay, E1S, uplim)
						
			## Initialize recombination rate*vol table
			# gluon-recombination
			print ("generating b+bbar -> Upsilon(1S)+g rate*vol table, vol in fm^3")
			p_relarray = np.exp( np.linspace(self.p_rel_log_min, self.p_rel_log_max, self.N_p_rel) )
			grd_v, grd_T, grd_p_rel = np.meshgrid(varray, Tarray, p_relarray)
			self.T_RV1S_reco_gluon = pyRV1S_reco_gluon(grd_v, grd_T, grd_p_rel).transpose(1,0,2)
			print ("generating b+bbar -> Upsilon(2S)+g rate*vol table, vol in fm^3")
			self.T_RV2S_reco_gluon = pyRV2S_reco_gluon(grd_v, grd_T, grd_p_rel).transpose(1,0,2)
			
			# inelastic quark dissociation
			print ("generating b+bbar+q -> Upsilon(1S)+q rate*vol table, vol in fm^3")
			self.T_RV1S_reco_ineq = pyRV1S_reco_ineq(grd_v, grd_T, grd_p_rel).transpose(1,0,2)
			self.T_S1S_reco_ineq_max = np.zeros_like(self.T_RV1S_reco_ineq)
			print ("generating b+bbar+q -> Upsilon(2S)+q rate*vol table, vol in fm^3")
			self.T_RV2S_reco_ineq = pyRV2S_reco_ineq(grd_v, grd_T, grd_p_rel).transpose(1,0,2)
			self.T_S2S_reco_ineq_max = np.zeros_like(self.T_RV2S_reco_ineq)
			
			# inelastic gluon dissociation
			print ("generating b+bbar+g -> Upsilon(1S)+g rate*vol table, vol in fm^3")
			self.T_RV1S_reco_ineg = pyRV1S_reco_ineg(grd_v, grd_T, grd_p_rel).transpose(1,0,2)
			self.T_S1S_reco_ineg_max = np.zeros_like(self.T_RV1S_reco_ineg)
			print ("generating b+bbar+g -> Upsilon(2S)+g rate*vol table, vol in fm^3")
			self.T_RV2S_reco_ineg = pyRV2S_reco_ineg(grd_v, grd_T, grd_p_rel).transpose(1,0,2)
			self.T_S2S_reco_ineg_max = np.zeros_like(self.T_RV2S_reco_ineg)

			# generate the maximums for sampling
			print ("generating reco sampling")
			for iv, v in enumerate(varray):
				gamma = np.sqrt(1.-v*v)
				for iT, T in enumerate(Tarray):
					params_reco[0] = gamma*(1.-v)/T
					params_reco[1] = gamma*(1.+v)/T
					uplim = 15.0*T/np.sqrt(1.-v)
					for iPrel, Prel in enumerate(p_relarray):
						params_reco[2] = Prel*Prel/M
						self.T_S1S_reco_ineq_max[iv, iT, iPrel] = find_max(&f_p1_reco1S_important, params, 0.0, uplim)
						self.T_S2S_reco_ineq_max[iv, iT, iPrel] = find_max(&f_p1_reco2S_important, params, 0.0, uplim)
						self.T_S1S_reco_ineg_max[iv, iT, iPrel] = find_max(&f_q1_reco1S_important, params, 0.0, uplim)
						self.T_S2S_reco_ineg_max[iv, iT, iPrel] = find_max(&f_q1_reco2S_important, params, 0.0, uplim)
										
			## store the disso and reco rates in datasets
			gp.create_dataset('R1S_decay_gluon', data=self.T_R1S_decay_gluon)
			gp.create_dataset('R1S_decay_ineq', data=self.T_R1S_decay_ineq)
			gp.create_dataset('R1S_decay_ineg', data=self.T_R1S_decay_ineg)
			gp.create_dataset('qdRdq_1S_gluon_max', data=self.T_qdRdq_1S_gluon_max)
			gp.create_dataset('RV1S_reco_gluon', data = self.T_RV1S_reco_gluon)
			gp.create_dataset('RV1S_reco_ineq', data = self.T_RV1S_reco_ineq)
			gp.create_dataset('RV1S_reco_ineg', data = self.T_RV1S_reco_ineg)
			gp.create_dataset('R2S_decay_gluon', data=self.T_R2S_decay_gluon)
			gp.create_dataset('R2S_decay_ineq', data=self.T_R2S_decay_ineq)
			gp.create_dataset('R2S_decay_ineg', data=self.T_R2S_decay_ineg)
			gp.create_dataset('qdRdq_2S_gluon_max', data=self.T_qdRdq_2S_gluon_max)
			gp.create_dataset('RV2S_reco_gluon', data = self.T_RV2S_reco_gluon)
			gp.create_dataset('RV2S_reco_ineq', data = self.T_RV2S_reco_ineq)
			gp.create_dataset('RV2S_reco_ineg', data = self.T_RV2S_reco_ineg)
			
			gp.create_dataset('S1S_decay_ineq_max', data = self.T_S1S_decay_ineq_max)
			gp.create_dataset('S2S_decay_ineq_max', data = self.T_S2S_decay_ineq_max)
			gp.create_dataset('S1S_decay_ineg_max', data = self.T_S1S_decay_ineg_max)
			gp.create_dataset('S2S_decay_ineg_max', data = self.T_S2S_decay_ineg_max)
			gp.create_dataset('S1S_reco_ineq_max', data = self.T_S1S_reco_ineq_max)
			gp.create_dataset('S2S_reco_ineq_max', data = self.T_S2S_reco_ineq_max)
			gp.create_dataset('S1S_reco_ineg_max', data = self.T_S1S_reco_ineg_max)
			gp.create_dataset('S2S_reco_ineg_max', data = self.T_S2S_reco_ineg_max)
			
			gp.attrs.create('v-min-max-N', 
							np.array([self.vmin, self.vmax, self.N_v]))
			gp.attrs.create('T-min-max-N', 
							np.array([self.Tmin, self.Tmax, self.N_T]))
			gp.attrs.create('p_rel_log-min-max-N', 
							np.array([self.p_rel_log_min, self.p_rel_log_max, self.N_p_rel]))
		
		self.dv = (self.vmax-self.vmin)/(self.N_v - 1.)
		self.dT = (self.Tmax-self.Tmin)/(self.N_T - 1.)
		self.dp_rel = (self.p_rel_log_max - self.p_rel_log_min)/(self.N_p_rel - 1.)

		print ("done")
	
	##----------- define functions that can be called and give rates -------------##
	# 1S real gluon dissociation
	cpdef get_R1S_decay_gluon(self, double v, double T):
		return interp2d(self.T_R1S_decay_gluon, v, T, 
						self.vmin, self.vmax, self.dv, self.N_v,
						self.Tmin, self.Tmax, self.dT, self.N_T)
						
	# 1S real gluon dissociation max integrand
	cpdef get_qdRdq_1S_gluon_max(self, double v, double T):
		return interp2d(self.T_qdRdq_1S_gluon_max, v, T, 
						self.vmin, self.vmax, self.dv, self.N_v,
						self.Tmin, self.Tmax, self.dT, self.N_T)
						
	# 1S inelastic quark dissociation
	cpdef get_R1S_decay_ineq(self, double v, double T):
		return interp2d(self.T_R1S_decay_ineq, v, T, 
						self.vmin, self.vmax, self.dv, self.N_v,
						self.Tmin, self.Tmax, self.dT, self.N_T)

	# 1S inelastic gluon dissociation
	cpdef get_R1S_decay_ineg(self, double v, double T):
		return interp2d(self.T_R1S_decay_ineg, v, T, 
						self.vmin, self.vmax, self.dv, self.N_v,
						self.Tmin, self.Tmax, self.dT, self.N_T)
							
	# 1S gluon recombination
	cpdef get_R1S_reco_gluon(self, double v, double T, double p_rel, double r):
		small_number = 0.000001		# add this small number to p_rel in case p_rel = 0
		cdef double Rvol = interp3d(self.T_RV1S_reco_gluon, v, T, np.log(p_rel+small_number),
									self.vmin, self.vmax, self.dv, self.N_v,
									self.Tmin, self.Tmax, self.dT, self.N_T,
									self.p_rel_log_min, self.p_rel_log_max, self.dp_rel, self.N_p_rel)
		return Rvol * dist_position_1S(r)
		# no factor of 2 in dist_position, add that when judging the theta function
	
	# 1S inelastic quark recombination
	cpdef get_R1S_reco_ineq(self, double v, double T, double p_rel, double r):
		small_number = 0.000001		# add this small number to p_rel in case p_rel = 0
		cdef double Rvol = interp3d(self.T_RV1S_reco_ineq, v, T, np.log(p_rel+small_number),
									self.vmin, self.vmax, self.dv, self.N_v,
									self.Tmin, self.Tmax, self.dT, self.N_T,
									self.p_rel_log_min, self.p_rel_log_max, self.dp_rel, self.N_p_rel)
		return Rvol * dist_position_1S(r)	

	# 1S inelastic gluon recombination
	cpdef get_R1S_reco_ineg(self, double v, double T, double p_rel, double r):
		small_number = 0.000001		# add this small number to p_rel in case p_rel = 0
		cdef double Rvol = interp3d(self.T_RV1S_reco_ineg, v, T, np.log(p_rel+small_number),
									self.vmin, self.vmax, self.dv, self.N_v,
									self.Tmin, self.Tmax, self.dT, self.N_T,
									self.p_rel_log_min, self.p_rel_log_max, self.dp_rel, self.N_p_rel)
		return Rvol * dist_position_1S(r)	
	
	# 2S real gluon dissociation
	cpdef get_R2S_decay_gluon(self, double v, double T):
		return interp2d(self.T_R2S_decay_gluon, v, T, 
						self.vmin, self.vmax, self.dv, self.N_v,
						self.Tmin, self.Tmax, self.dT, self.N_T)
						
	# 2S real gluon dissociation max integrand
	cpdef get_qdRdq_2S_gluon_max(self, double v, double T):
		return interp2d(self.T_qdRdq_2S_gluon_max, v, T, 
						self.vmin, self.vmax, self.dv, self.N_v,
						self.Tmin, self.Tmax, self.dT, self.N_T)
						
	# 2S inelastic quark dissociation
	cpdef get_R2S_decay_ineq(self, double v, double T):
		return interp2d(self.T_R2S_decay_ineq, v, T, 
						self.vmin, self.vmax, self.dv, self.N_v,
						self.Tmin, self.Tmax, self.dT, self.N_T)

	# 2S inelastic gluon dissociation
	cpdef get_R2S_decay_ineg(self, double v, double T):
		return interp2d(self.T_R2S_decay_ineg, v, T, 
						self.vmin, self.vmax, self.dv, self.N_v,
						self.Tmin, self.Tmax, self.dT, self.N_T)
							
	# 2S gluon recombination
	cpdef get_R2S_reco_gluon(self, double v, double T, double p_rel, double r):
		small_number = 0.000001		# add this small number to p_rel in case p_rel = 0
		cdef double Rvol = interp3d(self.T_RV2S_reco_gluon, v, T, np.log(p_rel+small_number),
									self.vmin, self.vmax, self.dv, self.N_v,
									self.Tmin, self.Tmax, self.dT, self.N_T,
									self.p_rel_log_min, self.p_rel_log_max, self.dp_rel, self.N_p_rel)
		return Rvol * dist_position_2S(r)
		# no factor of 2 in dist_position, add that when judging the theta function
	
	# 2S inelastic quark recombination
	cpdef get_R2S_reco_ineq(self, double v, double T, double p_rel, double r):
		small_number = 0.000001		# add this small number to p_rel in case p_rel = 0
		cdef double Rvol = interp3d(self.T_RV2S_reco_ineq, v, T, np.log(p_rel+small_number),
									self.vmin, self.vmax, self.dv, self.N_v,
									self.Tmin, self.Tmax, self.dT, self.N_T,
									self.p_rel_log_min, self.p_rel_log_max, self.dp_rel, self.N_p_rel)
		return Rvol * dist_position_2S(r)	

	# 2S inelastic gluon recombination
	cpdef get_R2S_reco_ineg(self, double v, double T, double p_rel, double r):
		small_number = 0.000001		# add this small number to p_rel in case p_rel = 0
		cdef double Rvol = interp3d(self.T_RV2S_reco_ineg, v, T, np.log(p_rel+small_number),
									self.vmin, self.vmax, self.dv, self.N_v,
									self.Tmin, self.Tmax, self.dT, self.N_T,
									self.p_rel_log_min, self.p_rel_log_max, self.dp_rel, self.N_p_rel)
		return Rvol * dist_position_2S(r)	
		
						
	##----------------- define function that can be called to sample ------------------##
	# 1S gluon dissociation		
	cpdef vector[double] sample_S1S_decay_gluon(self, double v, double T):
		cdef vector[double] pQpQbar, pQ, pQbar
		pQpQbar = S1S_decay_gluon(v, T, self.get_qdRdq_1S_gluon_max(v, T))
		pQ = np.array(p3top4_Q(pQpQbar[0:3]))
		pQbar = np.array(p3top4_Q(pQpQbar[3:6]))
		return np.concatenate((pQ, pQbar), axis=0)
		
	# 1S inelastic quark dissociation
	# max
	cpdef get_S1S_decay_ineq_max(self, double v, double T):
		return interp2d(self.T_S1S_decay_ineq_max, v, T, 
						self.vmin, self.vmax, self.dv, self.N_v,
						self.Tmin, self.Tmax, self.dT, self.N_T)
						
	cpdef vector[double] sample_S1S_decay_ineq(self, double v, double T):
		cdef vector[double] pQpQbar, pQ, pQbar
		if v < 0.999:
			pQpQbar = S1S_decay_ineq(v, T, self.get_S1S_decay_ineq_max(v, T))
		else:
			pQpQbar = S1S_decay_gluon(v, T, self.get_qdRdq_1S_gluon_max(v, T))
		pQ = np.array(p3top4_Q(pQpQbar[0:3]))
		pQbar = np.array(p3top4_Q(pQpQbar[3:6]))
		return np.concatenate((pQ, pQbar), axis=0)

	# 1S inelastic gluon dissociation
	# max
	cpdef get_S1S_decay_ineg_max(self, double v, double T):
		return interp2d(self.T_S1S_decay_ineg_max, v, T, 
						self.vmin, self.vmax, self.dv, self.N_v,
						self.Tmin, self.Tmax, self.dT, self.N_T)
						
	cpdef vector[double] sample_S1S_decay_ineg(self, double v, double T):
		cdef vector[double] pQpQbar, pQ, pQbar
		if v < 0.999:
			pQpQbar = S1S_decay_ineg(v, T, self.get_S1S_decay_ineg_max(v, T))
		else:
			pQpQbar = S1S_decay_gluon(v, T, self.get_qdRdq_1S_gluon_max(v, T))
		pQ = np.array(p3top4_Q(pQpQbar[0:3]))
		pQbar = np.array(p3top4_Q(pQpQbar[3:6]))
		return np.concatenate((pQ, pQbar), axis=0)
					
	# 1S gluon recombination
	cpdef vector[double] sample_S1S_reco_gluon(self, double v, double T, double p_rel):
		cdef vector[double] p1S
		p1S = S1S_reco_gluon(v, T, p_rel)
		return np.array(p3top4_quarkonia_1S(p1S))	# convert p3 to p4
		
	# 1S inelastic quark recombination
	# max
	cpdef get_S1S_reco_ineq_max(self, double v, double T, double p_rel):
		return interp3d(self.T_S1S_reco_ineq_max, v, T, np.log(p_rel+small_number),
									self.vmin, self.vmax, self.dv, self.N_v,
									self.Tmin, self.Tmax, self.dT, self.N_T,
									self.p_rel_log_min, self.p_rel_log_max, self.dp_rel, self.N_p_rel)
						
	cpdef vector[double] sample_S1S_reco_ineq(self, double v, double T, double p_rel):
		cdef vector[double] p1S
		if v < 0.999:
			p1S = S1S_reco_ineq(v, T, p_rel, self.get_S1S_reco_ineq_max(v, T, p_rel))
		else:
			p1S = S1S_reco_gluon(v, T, p_rel)
		return np.array(p3top4_quarkonia_1S(p1S))	# convert p3 to p4

	# 1S inelastic gluon recombination
	# max
	cpdef get_S1S_reco_ineg_max(self, double v, double T, double p_rel):
		return interp3d(self.T_S1S_reco_ineg_max, v, T, np.log(p_rel+small_number),
									self.vmin, self.vmax, self.dv, self.N_v,
									self.Tmin, self.Tmax, self.dT, self.N_T,
									self.p_rel_log_min, self.p_rel_log_max, self.dp_rel, self.N_p_rel)
									
	cpdef vector[double] sample_S1S_reco_ineg(self, double v, double T, double p_rel):
		cdef vector[double] p1S
		if v < 0.999:
			p1S = S1S_reco_ineg(v, T, p_rel, self.get_S1S_reco_ineg_max(v, T, p_rel))
		else:
			p1S = S1S_reco_gluon(v, T, p_rel)
		return np.array(p3top4_quarkonia_1S(p1S))	# convert p3 to p4

	# 2S gluon dissociation		
	cpdef vector[double] sample_S2S_decay_gluon(self, double v, double T):
		cdef vector[double] pQpQbar, pQ, pQbar
		pQpQbar = S2S_decay_gluon(v, T, self.get_qdRdq_2S_gluon_max(v, T))
		pQ = np.array(p3top4_Q(pQpQbar[0:3]))
		pQbar = np.array(p3top4_Q(pQpQbar[3:6]))
		return np.concatenate((pQ, pQbar), axis=0)
		
	# 2S inelastic quark dissociation
	# max
	cpdef get_S2S_decay_ineq_max(self, double v, double T):
		return interp2d(self.T_S2S_decay_ineq_max, v, T, 
						self.vmin, self.vmax, self.dv, self.N_v,
						self.Tmin, self.Tmax, self.dT, self.N_T)
						
	cpdef vector[double] sample_S2S_decay_ineq(self, double v, double T):
		cdef vector[double] pQpQbar, pQ, pQbar
		if v < 0.999:
			pQpQbar = S2S_decay_ineq(v, T, self.get_S2S_decay_ineq_max(v, T))
		else:
			pQpQbar = S2S_decay_gluon(v, T, self.get_qdRdq_2S_gluon_max(v, T))
		pQ = np.array(p3top4_Q(pQpQbar[0:3]))
		pQbar = np.array(p3top4_Q(pQpQbar[3:6]))
		return np.concatenate((pQ, pQbar), axis=0)

	# 2S inelastic gluon dissociation
	# max
	cpdef get_S2S_decay_ineg_max(self, double v, double T):
		return interp2d(self.T_S2S_decay_ineg_max, v, T, 
						self.vmin, self.vmax, self.dv, self.N_v,
						self.Tmin, self.Tmax, self.dT, self.N_T)
						
	cpdef vector[double] sample_S2S_decay_ineg(self, double v, double T):
		cdef vector[double] pQpQbar, pQ, pQbar
		if v < 0.999:
			pQpQbar = S2S_decay_ineg(v, T, self.get_S2S_decay_ineg_max(v, T))
		else:
			pQpQbar = S2S_decay_gluon(v, T, self.get_qdRdq_2S_gluon_max(v, T))
		pQ = np.array(p3top4_Q(pQpQbar[0:3]))
		pQbar = np.array(p3top4_Q(pQpQbar[3:6]))
		return np.concatenate((pQ, pQbar), axis=0)
					
	# 2S gluon recombination
	cpdef vector[double] sample_S2S_reco_gluon(self, double v, double T, double p_rel):
		cdef vector[double] p2S
		p2S = S2S_reco_gluon(v, T, p_rel)
		return np.array(p3top4_quarkonia_2S(p2S))	# convert p3 to p4
		
	# 2S inelastic quark recombination
	# max
	cpdef get_S2S_reco_ineq_max(self, double v, double T, double p_rel):
		return interp3d(self.T_S2S_reco_ineq_max, v, T, np.log(p_rel+small_number),
									self.vmin, self.vmax, self.dv, self.N_v,
									self.Tmin, self.Tmax, self.dT, self.N_T,
									self.p_rel_log_min, self.p_rel_log_max, self.dp_rel, self.N_p_rel)
									
	cpdef vector[double] sample_S2S_reco_ineq(self, double v, double T, double p_rel):
		cdef vector[double] p2S
		if v < 0.999:
			p2S = S2S_reco_ineq(v, T, p_rel, self.get_S2S_reco_ineq_max(v, T, p_rel))
		else:
			p2S = S2S_reco_gluon(v, T, p_rel)
		return np.array(p3top4_quarkonia_2S(p2S))	# convert p3 to p4

	# 2S inelastic gluon recombination
	# max
	cpdef get_S2S_reco_ineg_max(self, double v, double T, double p_rel):
		return interp3d(self.T_S2S_reco_ineg_max, v, T, np.log(p_rel+small_number),
									self.vmin, self.vmax, self.dv, self.N_v,
									self.Tmin, self.Tmax, self.dT, self.N_T,
									self.p_rel_log_min, self.p_rel_log_max, self.dp_rel, self.N_p_rel)
									
	cpdef vector[double] sample_S2S_reco_ineg(self, double v, double T, double p_rel):
		cdef vector[double] p2S
		if v < 0.999:
			p2S = S2S_reco_ineg(v, T, p_rel, self.get_S2S_reco_ineg_max(v, T, p_rel))
		else:
			p2S = S2S_reco_gluon(v, T, p_rel)
		return np.array(p3top4_quarkonia_2S(p2S))	# convert p3 to p4
		