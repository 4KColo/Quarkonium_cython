#!/usr/bin/env python
import numpy as np
import random as rd
import h5py
import LorRot

# Parameters: 	norm=125, nucleon-width=0.5fm, fluctuation=1,
#				(for norm=?, see 1605.03954)
# 				grid-max=15.05fm, (grid value is at the center of grid) 
#				grid-step=0.1fm, b-min=0fm, b-max=5.0fm
# EoS: HotQCD, e(T=Tc) = 0.329 Gev/fm^3, eta/s = 0.08+1*(T-Tc), bulk_norm=0.01, bulk_width=0.01


Taa = {'2760': {'0-10': 23.0, '0-5': 26.32, '5-10': 20.56, '10-20': 14.39, '20-30': 8.6975, '30-40': 5.001, '40-50': 2.675, '50-100': 0.458}}		# mb^-1
Xsect_bbbar = {'2760': 0.06453}		# mb
Xsect_1S = {'2760': 0.0001788}
Xsect_2S = {'2760': 0.00004456}

class Dynam_Initial_Sample:
	def __init__(self, energy_GeV = 2760, centrality_str = '0-10', channel_2S = False):
	
		### -------- store the position information -------- ###
		#file_TAB = h5py.File('ic-'+centrality_str+'-avg.hdf5','r')
		#Tab = np.array(file_TAB['TAB_0'].value)
		#Nx, Ny = Tab.shape
		#Tab_flat = Tab.reshape(-1)
		file_TAB = open(str(energy_GeV)+'initial_averaged_sd_cen'+centrality_str+'.dat','r')
		Tab = file_TAB.read().split()
		Tab_flat = np.array([float(each) for each in Tab])
		dx = 0.1 	# fm
		dy = 0.1	# fm
		Nx = Ny = np.sqrt(len(Tab_flat))
		T_tot = np.sum(Tab_flat)
		T_AA_mb = T_AA_mb = Taa[str(energy_GeV)][centrality_str]
		T_norm = Tab_flat/T_tot
		T_accum = np.zeros_like(T_norm, dtype=np.double)
		for index, value in enumerate(T_norm):
			T_accum[index] = T_accum[index-1] + value
		file_TAB.close()
		
		
		### -------- store the momentum information -------- ###
		filename_bbbar = str(energy_GeV) + "bbbar.dat"
		filename_1S = str(energy_GeV) + "1S.dat"
		filename_2S = str(energy_GeV) + "2S.dat"
		
		p4_bbbar = np.fromfile(filename_bbbar, dtype=float, sep=" ")
		p4_1S = np.fromfile(filename_1S, dtype=float, sep=" ")
		p4_2S = np.fromfile(filename_2S, dtype=float, sep=" ")
		len_bbbar = int(len(p4_bbbar)/8.0)
		len_1S = int(len(p4_1S)/4.0)
		len_2S = int(len(p4_2S)/4.0)
		
		N_bbbar = Xsect_bbbar[str(energy_GeV)] * T_AA_mb
		N_1S = Xsect_1S[str(energy_GeV)] * T_AA_mb
		N_2S = Xsect_2S[str(energy_GeV)] * T_AA_mb
		
		Nsam_bbbar = int(N_bbbar) + 1
		
		### ---------- sample momenta and postions --------- ###
		p4_Q = []
		p4_Qbar = []
		p4_U1S = []
		p4_U2S = []
		x3_Q = []
		x3_Qbar = []
		x3_1S = []
		x3_2S = []
		
		for i in range(Nsam_bbbar):
			r_bbbar = rd.uniform(0.0, Nsam_bbbar+0.0)
			if r_bbbar <= N_bbbar:
				## momenta
				row_bbbar = 8*rd.randrange(0, len_bbbar-1, 1)
				p4_Q.append( [p4_bbbar[row_bbbar], p4_bbbar[row_bbbar+1], p4_bbbar[row_bbbar+2], p4_bbbar[row_bbbar+3]] )
				p4_Qbar.append( [p4_bbbar[row_bbbar+4], p4_bbbar[row_bbbar+5], p4_bbbar[row_bbbar+6], p4_bbbar[row_bbbar+7]] )
				## positions
				r_xy = rd.uniform(0.0, 1.0)
				i_bbbar = np.searchsorted(T_accum, r_xy)
				i_x = np.floor((i_bbbar+0.0)/Ny)
				i_y = i_bbbar - i_x*Ny
				i_x += np.random.rand()
				i_y += np.random.rand()
				x = (i_x - Nx/2.)*dx
				y = (i_y - Ny/2.)*dy
				x3_Q.append(np.array([x,y,0.0]))
				x3_Qbar.append(np.array([x,y,0.0]))

		## since the bottomonia cross sections are too small
		## we only sample cases where bottomonia are produced			
		r_nS = rd.uniform(0.0, 1.0)*(N_1S + N_2S * channel_2S)
		if r_nS <= N_1S:
			## momenta
			row_1S = 4*rd.randrange(0, len_1S-1, 1)
			p4_U1S.append( [p4_1S[row_1S], p4_1S[row_1S+1], p4_1S[row_1S+2], p4_1S[row_1S+3]] )
			## positions
			r_xy = rd.uniform(0.0, 1.0)
			i_1S = np.searchsorted(T_accum, r_xy)
			i_x = np.floor((i_1S+0.0)/Ny)
			i_y = i_1S - i_x*Ny
			i_x += np.random.rand()
			i_y += np.random.rand()
			x = (i_x - Nx/2.)*dx
			y = (i_y - Ny/2.)*dy
			x3_1S.append(np.array([x,y,0.0]))
		
		elif r_nS <= N_1S + N_2S:
			## momenta
			row_2S = 4*rd.randrange(0, len_2S-1, 1)
			p4_U2S.append( [p4_2S[row_2S], p4_2S[row_2S+1], p4_2S[row_2S+2], p4_2S[row_2S+3]] )
			## positions
			r_xy = rd.uniform(0.0, 1.0)
			i_2S = np.searchsorted(T_accum, r_xy)
			i_x = np.floor((i_2S+0.0)/Ny)
			i_y = i_2S - i_x*Ny
			i_x += np.random.rand()
			i_y += np.random.rand()
			x = (i_x - Nx/2.)*dx
			y = (i_y - Ny/2.)*dy
			x3_2S.append(np.array([x,y,0.0]))
					
		self.x3_Q = np.array(x3_Q)
		self.x3_Qbar = np.array(x3_Qbar)
		self.x3_1S = np.array(x3_1S)
		self.x3_2S = np.array(x3_2S)
		self.p4_Q = np.array(p4_Q)
		self.p4_Qbar = np.array(p4_Qbar)
		self.p4_U1S = np.array(p4_U1S)
		self.p4_U2S = np.array(p4_U2S)

		
	def Qinit_p(self):
		return self.p4_Q
	
	def Qbarinit_p(self):
		return self.p4_Qbar
	
	def U1Sinit_p(self):
		return self.p4_U1S	

	def U2Sinit_p(self):
		return self.p4_U2S	
		
	def Qinit_x(self):
		return self.x3_Q
		
	def Qbarinit_x(self):
		return self.x3_Qbar
	
	def U1Sinit_x(self):
		return self.x3_1S	

	def U2Sinit_x(self):
		return self.x3_2S	
		
		#unbound pair production sampling:
			#N_co = 8*rd.randrange(0, len_corr-1, 1)
			#p4_Q.append( [p4_corr[N_co], p4_corr[N_co+1], p4_corr[N_co+2], p4_corr[N_co+3]] )
			#p4_Qbar.append( [p4_corr[N_co+4], p4_corr[N_co+5], p4_corr[N_co+6], p4_corr[N_co+7]] )
			#N_co = 4*rd.randrange(0, len_corr-1, 1)
			#p_x = p4_corr[N_co+1]*0.5
			#p_y = p4_corr[N_co+2]*0.5
			#p_z = p4_corr[N_co+3]*0.5
			#E_e = np.sqrt(M**2 + p_x**2 + p_y**2 + p_z**2)
			#p4_Q.append( [E_e, p_x, p_y, p_z] )
			#p4_Qbar.append( [E_e, p_x, p_y, p_z] )
			#v3 = p4_corr[N_co+1:N_co+4]/p4_corr[N_co]
			#costheta = rd.uniform(0.0, 1.0)
			#sintheta = np.sqrt(1.0 - costheta**2)
			#phi = rd.uniform(0.0, 2.0*np.pi)
			#cosphi = np.cos(phi)
			#sinphi = np.sin(phi)
			#E_rest = np.sqrt(M**2 + Prel**2)
			#px_rest = Prel * sintheta * cosphi
			#py_rest = Prel * sintheta * sinphi
			#pz_rest = Prel * costheta
			#p4Q_rest = np.array([E_rest, px_rest, py_rest, pz_rest])
			#p4Qbar_rest = np.array([E_rest, -px_rest, -py_rest, -pz_rest])
			#p4_Q.append( LorRot.lorentz(p4Q_rest, -v3) )
			#p4_Qbar.append( LorRot.lorentz(p4Qbar_rest, -v3) )		