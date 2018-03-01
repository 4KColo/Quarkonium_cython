#!/usr/bin/env python
import numpy as np
import scipy.integrate as si
from Static_quarkonium_evolution import QQbar_evol
import h5py


#### ------------ multiple runs averaged and compare ---------------- ####
N_ave = 5000		# #of parallel runnings
T = 0.3		
N_step = 2500
dt = 0.04
tmax = N_step*dt
t = np.linspace(0.0, tmax, N_step+1)
Nb0 = 50			# initial number of Q or Qbar
N1s0 = 0			# initial number of U1s
N1s_t = []			# to store number of U1s in each time step
Nb_t = []			# to store number of Q or Qbar in each time step
P_sample = 5.0		# GeV, initial uniform sampling
Process_chosen = 'ineq'

# define the event generator
event_gen = QQbar_evol('static', temp_init = T, HQ_scat = False, process = Process_chosen)

for i in range(N_ave):
	# initialize N_ave number of events
	event_gen.initialize(N_Q = Nb0, N_Qbar = Nb0, N_U1S = N1s0, thermal_dist = True )
	#event_gen.initialize(N_Q = Nb0, N_Qbar = Nb0, N_U1S = N1s0, uniform_dist = True, Pmax = P_sample)
	N1s_t.append([])
	for j in range(N_step+1):
		N1s_t[i].append(len(event_gen.U1Slist['4-momentum']))	# store N_1S(t) for each event
		event_gen.run()
	event_gen.dict_clear()	## clear data from last simulation

	
N1s_t = np.array(N1s_t)
N1s_t_ave = np.sum(N1s_t, axis = 0)/(N_ave + 0.0)	# averaged number of 1S state (time-sequenced)

Nb_t_ave = Nb0 + N1s0 - N1s_t_ave		# time-sequenced bottom quark number

R1s_t = N1s_t_ave/(Nb0+N1s0)			# ratio of number of Q in the U1S
Rb_t = 1.0 - R1s_t						# ratio
#### ------------ end of multiple runs averaged and compare ---------- ####




#### ------------ save the data in a h5py file ------------- ####

file1 = h5py.File('ThermalnoHQT='+str(T)+'N_event='+str(N_ave)+'N_step='+str(N_step)+'Nb0='+str(Nb0)+'N1s0='+str(N1s0)+str(Process_chosen)+'.hdf5', 'w')
#file1 = h5py.File('UniformTPmax='+str(P_sample)+'HQT='+str(T)+'N_event='+str(N_ave)+'N_step='+str(N_step)+'Nc0='+str(Nc0)+'N1s0='+str(N1s0)+str(Process_chosen)+'.hdf5', 'w')
file1.create_dataset('percentage', data = R1s_t)
file1.create_dataset('time', data = t)
file1.close()

#### ------------ end of saving the file ------------- ####
