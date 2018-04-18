#!/usr/bin/env python
import numpy as np
import scipy.integrate as si
from Dynam_quarkonium_evolution import QQbar_evol
import h5py


#### ------------ multiple runs averaged and compare ---------------- ####
centrality = '0-10'
energy = 2760

N_ave = 2		# No. of parallel runnings
N_step = 1268		# total step required is t_total / dt, t_total = 11.4/(2/cosh(y_max)), 11.4 fm/c is the lifetime of QGP in 0-10% 2.76 GeV
dt_run = 0.05
tmax = N_step*dt_run
t = np.linspace(0.0, tmax, N_step+1)
p4i_1S = []			# initial p4
p4f_1S = []			# final p4
p4f_2S = []
tForm_1S = []		# formation time of 1S
tForm_2S = []
N1S_t = []			# time sequence of No. of 1S state
N2S_t = []			# time sequence of No. of 2S state

# define the event generator
event_gen = QQbar_evol(centrality_str_given = centrality, energy_GeV = energy, recombine = True, HQ_scat = True)

for i in range(N_ave):
	# initialize N_ave number of events
	event_gen.initialize(sample_method = '1S')
	N1S_t.append([])
	N2S_t.append([])
	
	# store initial momenta
	leni_1S = len(event_gen.U1Slist['4-momentum'])	# initial No. of 1S
	for k in range(leni_1S):
		p4i_1S.append(event_gen.U1Slist['4-momentum'][k])
		
	# run the event
	for j in range(N_step+1):
		N1S_t[i].append(len(event_gen.U1Slist['4-momentum']))	# store N_1S(t) for each event
		N2S_t[i].append(len(event_gen.U2Slist['4-momentum']))
		event_gen.run(dt = dt_run)
	
	# store final momenta
	lenf_1S = len(event_gen.U1Slist['4-momentum'])	# final No. of 1S
	lenf_2S = len(event_gen.U2Slist['4-momentum'])	# final No. of 2S
	for k in range(lenf_1S):
		p4f_1S.append(event_gen.U1Slist['4-momentum'][k])
		tForm_1S.append(event_gen.U1Slist['last_form_time'][k])
	for k in range(lenf_2S):
		p4f_2S.append(event_gen.U2Slist['4-momentum'][k])
		tForm_2S.append(event_gen.U2Slist['last_form_time'][k])
		
	event_gen.dict_clear()	## clear data from last simulation

	
N1S_t = np.array(N1S_t)
N2S_t = np.array(N2S_t)
N1S_t_ave = np.sum(N1S_t, axis = 0)/(N_ave + 0.0)	# averaged number of 1S state (time-sequenced)
N2S_t_ave = np.sum(N2S_t, axis = 0)/(N_ave + 0.0)
#### ------------ end of multiple runs averaged and compare ---------- ####




#### ------------ save the data in a h5py file ------------- ####

file1 = h5py.File('energy='+str(energy)+'GeVcentrality='+str(centrality)+'N_event='+str(N_ave)+'_1S.hdf5', 'w')
file1.create_dataset('1Sp4initial', data = p4i_1S)
file1.create_dataset('1Sp4final', data = p4f_1S)
file1.create_dataset('1Snumber', data = N1S_t_ave)
file1.create_dataset('1Sformtime', data = tForm_1S)
file1.create_dataset('2Sp4final', data = p4f_2S)
file1.create_dataset('2Snumber', data = N2S_t_ave)
file1.create_dataset('2Sformtime', data = tForm_2S)
file1.create_dataset('time', data = t)
file1.close()

#### ------------ end of saving the file ------------- ####
