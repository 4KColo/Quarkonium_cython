#!/usr/bin/env python
import sys
sys.path.append('/Users/Colo4K/Desktop/Thesis/cython_quarkonium')
import numpy as np
import scipy.integrate as si
from Static_quarkonium_evolution import QQbar_evol
import h5py


alpha_s = 0.3	# for bottomonium
N_C = 3.0
T_F = 0.5
M = 4650.0		# MeV b-quark
rho_c = 1.0/(N_C**2-1.0)
C_F = 4.0/3.0
a_B = 2.0/(alpha_s*C_F*M)
E_1S = alpha_s*C_F/(2.0*a_B)  # here is magnitude, true value is its negative
M_1S = M*2-E_1S
C1 = 0.197327                 # 0.197 GeV*fm = 1


#### ------------ multiple runs averaged and compare ---------------- ####
N_ave = 10		# #of parallel runnings
T = 0.3		
N_step = 25
dt = 0.04
tmax = N_step*dt
t = np.linspace(0.0, tmax, N_step+1)
Nb0 = 50			# initial number of Q or Qbar
N1s0 = 0			# initial number of U1s
N1s_t = []			# to store number of U1s in each time step
Nb_t = []			# to store number of Q or Qbar in each time step
P_sample = 10.0		# GeV, initial uniform sampling


## initialize N_ave number of events
events = []
for i in range(N_ave):
	events.append(QQbar_evol('static', temp_init = T, HQ_scat = True))
	events[i].initialize(N_Q = Nb0, N_U1S = N1s0, thermal_dist = True)


## next store the N(t), px, py, pz into arrays
for i in range(N_step+1):
	len_u1s_tot = 0.0
	
	for j in range(N_ave):
		len_u1s_tot += len(events[j].U1Slist['4-momentum'])		# length of U1S list in one event
		events[j].run()	
	N1s_t.append(len_u1s_tot/N_ave)		# store the average Num. of U1S	


N1s_t = np.array(N1s_t)			# time-sequenced particle number
Nb_t = Nb0 + N1s0 - N1s_t		# time-sequenced particle number

N1s_r = N1s_t/(Nb0+N1s0+0.0)	# ratio
Nb_r = 1.0 - N1s_r				# ratio


#### ------------ end of multiple runs averaged and compare ---------- ####





#### ------------ save the data in a h5py file ------------- ####
Ep = []
for i in range(0, N_ave):
	if len(events[i].U1Slist['4-momentum']) != 0:
		if len(Ep) == 0:
			Ep = events[i].U1Slist['4-momentum']
		else:
			Ep = np.append(Ep, events[i].U1Slist['4-momentum'], axis=0)


file1 = h5py.File('ThermalT='+str(T)+'N_event='+str(N_ave)+'N_step='+str(N_step)+'Nb0='+str(Nb0)+'N1s0='+str(N1s0)+'.hdf5', 'w')
file1.create_dataset('percentage', data = N1s_r)
file1.create_dataset('time', data = t)
file1.create_dataset('4momentum', data = Ep)
file1.close()

#### ------------ end of saving the file ------------- ####


















