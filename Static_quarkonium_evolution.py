#!/usr/bin/env python
import numpy as np
import h5py
import csv
import random as rd
from scipy.spatial import cKDTree
from Static_Initial_Sample import Static_Initial_Sample
from Static_HQ_evo import HQ_p_update
import DisRec
import LorRot


#### ---------------------- some constants -----------------------------
alpha_s = 0.4 				  # for bottomonium, choose the same value as alpha_s_pot in the cython library
N_C = 3.0
T_F = 0.5
M = 4.65 					  # GeV b-quark
rho_c = 1.0/(N_C**2-1.0)
C_F = 4.0/3.0
a_B = 2.0/(alpha_s*C_F*M)
E_1S = alpha_s*C_F/(2.0*a_B)  # Upsilon(1S), here is magnitude, true value is its negative
M_1S = M*2.0 - E_1S  		  # mass of Upsilon(1S)
C1 = 0.197327                 # 0.197 GeV*fm = 1
R_search = 1.0				  # (fm), pair-search radius in the recombination
T_1S = 1.5					  # melting temperature of Upsilon_1S in GeV


####--------- initial sample of heavy Q and Qbar using thermal distribution -------- ####
def thermal_dist(temp, mass, momentum):
    return momentum**2*np.exp(-np.sqrt(mass**2+momentum**2)/temp)


# sample according to the thermal distribution function
def thermal_sample(temp, mass):
    p_max = np.sqrt(2.0*temp**2 + 2.0*temp*np.sqrt(temp**2+mass**2))
    # most probably momentum
    p_uplim = 10.0*p_max
    y_uplim = thermal_dist(temp, mass, p_max)
    while True:
        p_try = rd.uniform(0.0, p_uplim)
        y_try = rd.uniform(0.0, y_uplim)
        if y_try < thermal_dist(temp, mass, p_try):
            break
        
    E = np.sqrt(p_try**2+mass**2)
    cos_theta = rd.uniform(-1.0, 1.0)
    sin_theta = np.sqrt(1.0-cos_theta**2)
    phi = rd.uniform(0.0, 2.0*np.pi)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    return np.array([ E, p_try*sin_theta*cos_phi, p_try*sin_theta*sin_phi, p_try*cos_theta ])

####---- end of initial sample of heavy Q and Qbar using thermal distribution ---- ####




####--------- initial sample of heavy Q and Qbar using uniform distribution ------ ####
def uniform_sample(Pmax, mass):
    px, py, pz = (np.random.rand(3)-0.5)*2*Pmax
    E = np.sqrt(mass**2 + px**2 + py**2 + pz**2)
    return np.array([E, px, py, pz])

####----- end of initial sample of heavy Q and Qbar using uniform distribution ---- ####



class QQbar_evol:
####---- input the medium_type when calling the class ----####
	def __init__(self, medium_type = 'static', process = 'all', temp_init = 0.3, recombine = True, HQ_scat = False):
		self.type = medium_type
		self.process = process	# gluon: real-gluon, ineq: inelastic quark, ineg: inelastic gluon
		self.T = temp_init		# initial temperature in GeV	
		self.recombine = recombine
		self.HQ_scat = HQ_scat
		if self.HQ_scat == True:
			self.HQ_diff = HQ_p_update()
		## ---------- create the rates reader --------- ##
		self.event = DisRec.DisRec()
		
####---- initialize Q, Qbar, Quarkonium -- currently we only study Upsilon(1S) ----####
	def initialize(self, N_Q = 100, N_Qbar = 100, N_U1S = 10, Lmax = 10.0, thermal_dist = True,
	fonll_dist = False, Fonll_path = False, uniform_dist = False, Pmax = 10.0, decaytestmode = False, P_decaytest = [0.0, 0.0, 5.0]):
		# initial momentum: thermal; Fonll (give the fonll file path), uniform in x,y,z (give the Pmax in GeV)
        # if decaytestmode: give initial Px,Py,Pz in GeV
		if self.type == 'static':
			## ----- initialize the clock to keep track of time
			self.t = 0.0
			## ----- store the box side length
			self.Lmax = Lmax
			## ----- create dictionaries to store momenta, positions, id ----- ##
			self.Qlist = {'4-momentum': [], '3-position': [], 'id': 5}
			self.Qbarlist = {'4-momentum': [], '3-position': [], 'id': -5}
			self.U1Slist = {'4-momentum': [], '3-position': [], 'id': 533, 'last_form_time': []}
			
			## --------- sample initial momenta and positions -------- ##
			if thermal_dist == True:
				for i in range(N_Q):
					self.Qlist['4-momentum'].append( thermal_sample(self.T, M) )
					self.Qlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_Qbar):
					self.Qbarlist['4-momentum'].append( thermal_sample(self.T, M) )
					self.Qbarlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_U1S):
					self.U1Slist['4-momentum'].append( thermal_sample(self.T, M_1S) )
					self.U1Slist['3-position'].append( np.random.rand(3)*Lmax )
					self.U1Slist['last_form_time'].append( self.t )
			
			if fonll_dist == True:
				p_generator = Static_Initial_Sample(Fonll_path, rapidity = 0.)
				
				for i in range(N_Q):
					self.Qlist['4-momentum'].append( p_generator.p_HQ_sample(M) )
					self.Qlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_Qbar):
					self.Qbarlist['4-momentum'].append( p_generator.p_HQ_sample(M) )
					self.Qbarlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_U1S):
					self.U1Slist['4-momentum'].append( p_generator.p_U1S_sample(M, M_1S) )
					self.U1Slist['3-position'].append( np.random.rand(3)*Lmax )
					self.U1Slist['last_form_time'].append( self.t )
			
			if uniform_dist == True:
				for i in range(N_Q):
					self.Qlist['4-momentum'].append( uniform_sample(Pmax, M) )
					self.Qlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_Qbar):
					self.Qbarlist['4-momentum'].append( uniform_sample(Pmax, M) )
					self.Qbarlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_U1S):
					self.U1Slist['4-momentum'].append( uniform_sample(Pmax, M_1S) )
					self.U1Slist['3-position'].append( np.random.rand(3)*Lmax )
					self.U1Slist['last_form_time'].append( self.t )
			
			if decaytestmode == True:
				E_decaytest = np.sqrt(M_1S**2 + P_decaytest[0]**2 + P_decaytest[1]**2 + P_decaytest[2]**2)
				for i in range(N_Q):
					self.Qlist['4-momentum'].append( [M, 0.0, 0.0, 0.0] )
					self.Qlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_Qbar):
					self.Qbarlist['4-momentum'].append( [M, 0.0, 0.0, 0.0] )
					self.Qbarlist['3-position'].append( np.random.rand(3)*Lmax )
				for i in range(N_U1S):
					self.U1Slist['4-momentum'].append( np.append(E_decaytest, P_decaytest) )
					self.U1Slist['3-position'].append( np.random.rand(3)*Lmax )
					self.U1Slist['last_form_time'].append( self.t )
			                
			self.Qlist['4-momentum'] = np.array(self.Qlist['4-momentum'])
			self.Qlist['3-position'] = np.array(self.Qlist['3-position'])
			self.Qbarlist['4-momentum'] = np.array(self.Qbarlist['4-momentum'])
			self.Qbarlist['3-position'] = np.array(self.Qbarlist['3-position'])
			self.U1Slist['4-momentum'] = np.array(self.U1Slist['4-momentum'])
			self.U1Slist['3-position'] = np.array(self.U1Slist['3-position'])
			self.U1Slist['last_form_time'] = np.array(self.U1Slist['last_form_time'])
			

#### ---------------- event evolution function ------------------ ####
	def run(self, dt = 0.04, temp_run = -1.0):			# universal time to consider recombination
		len_Q = len(self.Qlist['4-momentum'])
		len_Qbar = len(self.Qbarlist['4-momentum'])
		len_U1S = len(self.U1Slist['4-momentum'])
		
		if temp_run != -1.0:
			self.T = temp_run
		
		### ------------- heavy quark diffusion --------------- ###
		if self.HQ_scat == True:
			for i in range(len_Q):
				self.Qlist['4-momentum'][i] = self.HQ_diff.update(HQ_Ep = self.Qlist['4-momentum'][i], Temp = self.T, time_step = dt)
			for i in range(len_Qbar):
				self.Qbarlist['4-momentum'][i] = self.HQ_diff.update(HQ_Ep = self.Qbarlist['4-momentum'][i], Temp = self.T, time_step = dt)
		
		### ----------- end of heavy quark diffusion ---------- ###




		### ----------- free stream these particles ------------###
		for i in range(len_Q):
			v3_Q = self.Qlist['4-momentum'][i][1:]/self.Qlist['4-momentum'][i][0]
			self.Qlist['3-position'][i] = (self.Qlist['3-position'][i] + dt * v3_Q)%self.Lmax
		for i in range(len_Qbar):
			v3_Qbar = self.Qbarlist['4-momentum'][i][1:]/self.Qbarlist['4-momentum'][i][0]
			self.Qbarlist['3-position'][i] = (self.Qbarlist['3-position'][i] + dt * v3_Qbar)%self.Lmax	
		for i in range(len_U1S):
			v3_U1S = self.U1Slist['4-momentum'][i][1:]/self.U1Slist['4-momentum'][i][0]
			self.U1Slist['3-position'][i] = (self.U1Slist['3-position'][i] + dt * v3_U1S)%self.Lmax
		### ----------- end of free stream particles -----------###




		### -------------------- decay ------------------------ ###
		delete_U1S = []
		add_pQ = []
		add_pQbar = []
		add_xQ = []
		#add_xQbar = [] the positions of Q and Qbar are the same
		
		for i in range(len_U1S):
			p4_in_box = self.U1Slist['4-momentum'][i]
			v3_in_box = p4_in_box[1:]/p4_in_box[0]
			v_in_box = np.sqrt(np.sum(v3_in_box**2))
			if self.process == 'all':
				rate_decay_gluon = self.event.get_R1S_decay_gluon( v_in_box, self.T )	# GeV
				rate_decay_ineq = self.event.get_R1S_decay_ineq( v_in_box, self.T )		# GeV
				rate_decay_ineg = self.event.get_R1S_decay_ineg( v_in_box, self.T )
			if self.process == 'gluon':
				rate_decay_gluon = self.event.get_R1S_decay_gluon( v_in_box, self.T )	# GeV
				rate_decay_ineq = 0.0
				rate_decay_ineg = 0.0
			if self.process == 'ineq':
				rate_decay_gluon = 0.0
				rate_decay_ineq = self.event.get_R1S_decay_ineq( v_in_box, self.T )		# GeV
				rate_decay_ineg = 0.0
			if self.process == 'ineg':
				rate_decay_gluon = 0.0
				rate_decay_ineq = 0.0
				rate_decay_ineg = self.event.get_R1S_decay_ineg( v_in_box, self.T )							
			prob_decay_gluon = rate_decay_gluon * dt/C1
			prob_decay_ineq = rate_decay_ineq * dt/C1
			prob_decay_ineg = rate_decay_ineg * dt/C1
			rej_mc = np.random.rand(1)
			
			if rej_mc <= prob_decay_gluon + prob_decay_ineq + prob_decay_ineg:
				delete_U1S.append(i)
				# outgoing momenta of Q Qbar in the rest frame of quarkonia, depends on decay channel
				if rej_mc <= prob_decay_gluon:
					recoil_pQpQbar = self.event.sample_S1S_decay_gluon( v_in_box, self.T )
				elif rej_mc <= prob_decay_gluon + prob_decay_ineq:
					recoil_pQpQbar = self.event.sample_S1S_decay_ineq( v_in_box, self.T )
				else:
					recoil_pQpQbar = self.event.sample_S1S_decay_ineg( v_in_box, self.T )
				
				recoil_pQ = np.array(recoil_pQpQbar[0:4])
				recoil_pQbar = np.array(recoil_pQpQbar[4:8])
				# Q, Qbar momenta need to be rotated from the v = z axis to what it is in the box frame
				# first get the rotation matrix angles
				theta_rot, phi_rot = LorRot.angle( v3_in_box )
				# then do the rotation in the spatial components
				rotmomentum_Q = LorRot.rotation4(recoil_pQ, theta_rot, phi_rot)
				rotmomentum_Qbar = LorRot.rotation4(recoil_pQbar, theta_rot, phi_rot)
				
				# we now transform them back to the box frame
				momentum_Q = LorRot.lorentz(rotmomentum_Q, -v3_in_box)			# final momentum of Q
				momentum_Qbar = LorRot.lorentz(rotmomentum_Qbar, -v3_in_box)	# final momentum of Qbar
				
				# positions of Q and Qbar
				position_Q = self.U1Slist['3-position'][i]
				#position_Qbar = position_Q
		
				# add x and p for the QQbar to the temporary list
				add_pQ.append(momentum_Q)
				add_pQbar.append(momentum_Qbar)
				add_xQ.append(position_Q)
				#add_xQbar.append(position_Qbar)
				
		### ------------------ end of decay ------------------- ###
		
		
		
		### ------------------ recombination ------------------ ###
		if self.recombine == True and len_Q != 0 and len_Qbar !=0:
			delete_Q = []
			delete_Qbar = []
			
			# make the periodic box 26 times bigger!
			Qbar_x_list = np.concatenate((self.Qbarlist['3-position'], 
			self.Qbarlist['3-position']+[0.0, 0.0, self.Lmax], self.Qbarlist['3-position']+[0.0, 0.0, -self.Lmax],
			self.Qbarlist['3-position']+[0.0, self.Lmax, 0.0], self.Qbarlist['3-position']+[0.0, -self.Lmax, 0.0],
			self.Qbarlist['3-position']+[self.Lmax, 0.0, 0.0], self.Qbarlist['3-position']+[-self.Lmax, 0.0, 0.0],
			self.Qbarlist['3-position']+[0.0, self.Lmax, self.Lmax], self.Qbarlist['3-position']+[0.0, self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[0.0, -self.Lmax, self.Lmax], self.Qbarlist['3-position']+[0.0, -self.Lmax, -self.Lmax],
			self.Qbarlist['3-position']+[self.Lmax, 0.0, self.Lmax], self.Qbarlist['3-position']+[self.Lmax, 0.0, -self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, 0.0, self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, 0.0, -self.Lmax],
			self.Qbarlist['3-position']+[self.Lmax, self.Lmax, 0.0], self.Qbarlist['3-position']+[self.Lmax, -self.Lmax, 0.0], self.Qbarlist['3-position']+[-self.Lmax, self.Lmax, 0.0], self.Qbarlist['3-position']+[-self.Lmax, -self.Lmax, 0.0],
			self.Qbarlist['3-position']+[self.Lmax, self.Lmax, self.Lmax], self.Qbarlist['3-position']+[self.Lmax, self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[self.Lmax, -self.Lmax, self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, self.Lmax, self.Lmax],
			self.Qbarlist['3-position']+[-self.Lmax, -self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, -self.Lmax, self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[self.Lmax, -self.Lmax, -self.Lmax]),
			axis = 0 )
			
			pair_search = cKDTree(Qbar_x_list)
			# for each Q, obtain the Qbar indexes within R_search
			pair_list = pair_search.query_ball_point(self.Qlist['3-position'], r = R_search)
			
			for i in range(len_Q):		# loop over Q
				len_recoQbar = len(pair_list[i])
				rate_reco_gluon = []
				rate_reco_ineq = []
				rate_reco_ineg = []
				for j in range(len_recoQbar):		# loop over Qbar within R_search
					# positions in lab frame
					xQ = self.Qlist['3-position'][i]
					xQbar = Qbar_x_list[pair_list[i][j]]
					x_rel = xQ - xQbar
					i_Qbar_mod = pair_list[i][j]%len_Qbar	# use for momentum and delete_index
					#rdotp = np.sum( x_rel* (self.Qlist['4-momentum'][i][1:] - self.Qbarlist['4-momentum'][i_Qbar_mod][1:]) )
					
					if  i_Qbar_mod not in delete_Qbar:
						r_rel = np.sqrt(np.sum(x_rel**2))
						#x_CM = 0.5*( xQ + xQbar )
						# momenta in hydro cell frame
						pQ = self.Qlist['4-momentum'][i]
						pQbar = self.Qbarlist['4-momentum'][i_Qbar_mod]
						# CM momentum and velocity
						v_CM, v_CM_abs, p_rel_abs = LorRot.vCM_prel(pQ, pQbar, 2.0*M)
						
						rate_reco_gluon.append(self.event.get_R1S_reco_gluon(v_CM_abs, self.T, p_rel_abs, r_rel))
						rate_reco_ineq.append(self.event.get_R1S_reco_ineq(v_CM_abs, self.T, p_rel_abs, r_rel))
						rate_reco_ineg.append(self.event.get_R1S_reco_ineg(v_CM_abs, self.T, p_rel_abs, r_rel))
						
					else:	# the Qbar has been taken by other Q's
						rate_reco_gluon.append(0.)
						rate_reco_ineq.append(0.)
						rate_reco_ineg.append(0.)
				
				# get the recombine probability
				# 3/4 factor for Upsilon(1S) v.s. eta_b
				if self.process == 'all':
					prob_reco_gluon = 0.75*8./9.*np.array(rate_reco_gluon)*dt/C1
					prob_reco_ineq = 0.75*8./9.*np.array(rate_reco_ineq)*dt/C1
					prob_reco_ineg = 0.75*8./9.*np.array(rate_reco_ineg)*dt/C1
				if self.process == 'gluon':
					prob_reco_gluon = 0.75*8./9.*np.array(rate_reco_gluon)*dt/C1
					prob_reco_ineq = 0.0*np.array(rate_reco_ineq)
					prob_reco_ineg = 0.0*np.array(rate_reco_ineg)
				if self.process == 'ineq':
					prob_reco_gluon = 0.0*np.array(rate_reco_gluon)
					prob_reco_ineq = 0.75*8./9.*np.array(rate_reco_ineq)*dt/C1
					prob_reco_ineg = 0.0*np.array(rate_reco_ineg)
				if self.process == 'ineg':
					prob_reco_gluon = 0.0*np.array(rate_reco_gluon)
					prob_reco_ineq = 0.0*np.array(rate_reco_ineq)
					prob_reco_ineg = 0.75*8./9.*np.array(rate_reco_ineg)*dt/C1
												
				#total_prob_reco_gluon = np.sum(prob_reco_gluon)
				#total_prob_reco_ineq = np.sum(prob_reco_ineq)
				#total_prob_reco_ineg = np.sum(prob_reco_ineg)
				prob_reco_all = prob_reco_gluon + prob_reco_ineq + prob_reco_ineg
				total_prob_reco_all = np.sum(prob_reco_all)
				#print total_prob_reco_all/dt*C1
				rej_mc = np.random.rand(1)
				
				if rej_mc <= total_prob_reco_all:
					delete_Q.append(i)		# remove this Q later
					# find the Qbar we need to remove
					a = 0.0
					for j in range(len_recoQbar):
						if a <= rej_mc <= a + prob_reco_all[j]:
							k = j
							if rej_mc <= a + prob_reco_gluon[j]:
								channel_reco = 'gluon'
							elif rej_mc <= a + prob_reco_gluon[j] + prob_reco_ineq[j]:
								channel_reco = 'ineq'
							else:
								channel_reco = 'ineg'
							break
						a += prob_reco_all[j]
					delete_Qbar.append(pair_list[i][k]%len_Qbar)
					
					# re-construct the reco event and sample initial and final states
					# positions and local temperature
					xQ = self.Qlist['3-position'][i]
					xQbar = Qbar_x_list[pair_list[i][k]]
					x_CM = 0.5*( xQ + xQbar )
					
					# momenta
					pQ = self.Qlist['4-momentum'][i]
					pQbar = self.Qbarlist['4-momentum'][pair_list[i][k]%len_Qbar]
					
					# CM momentum and velocity
					v_CM, v_CM_abs, p_rel_abs = LorRot.vCM_prel(pQ, pQbar, 2.0*M)

					# calculate the final quarkonium momenta in the CM frame of QQbar, depends on channel_reco
					if channel_reco == 'gluon':
						tempmomentum_U1S = np.array(self.event.sample_S1S_reco_gluon(v_CM_abs, self.T, p_rel_abs))
					elif channel_reco == 'ineq':
						tempmomentum_U1S = np.array(self.event.sample_S1S_reco_ineq(v_CM_abs, self.T, p_rel_abs))
					else:	# 'ineg'
						tempmomentum_U1S = np.array(self.event.sample_S1S_reco_ineg(v_CM_abs, self.T, p_rel_abs))
					
					# need to rotate the vector, v is not the z axis in box frame
					theta_rot, phi_rot = LorRot.angle(v_CM)
					rotmomentum_U1S = LorRot.rotation4(tempmomentum_U1S, theta_rot, phi_rot)
					# lorentz back to the box frame
					momentum_U1S = LorRot.lorentz( rotmomentum_U1S, -v_CM )
					# positions of the quarkonium
					position_U1S = x_CM%self.Lmax
					
					# update the quarkonium list
					if len(self.U1Slist['4-momentum']) == 0:
						self.U1Slist['4-momentum'] = np.array([momentum_U1S])
						self.U1Slist['3-position'] = np.array([position_U1S])
						self.U1Slist['last_form_time'] = np.array([self.t])
					else:
						self.U1Slist['4-momentum'] = np.append(self.U1Slist['4-momentum'], [momentum_U1S], axis=0)
						self.U1Slist['3-position'] = np.append(self.U1Slist['3-position'], [position_U1S], axis=0)
						self.U1Slist['last_form_time'] = np.append(self.U1Slist['last_form_time'], self.t)

			## now update Q and Qbar lists
			self.Qlist['4-momentum'] = np.delete(self.Qlist['4-momentum'], delete_Q, axis=0)
			self.Qlist['3-position'] = np.delete(self.Qlist['3-position'], delete_Q, axis=0)
			self.Qbarlist['4-momentum'] = np.delete(self.Qbarlist['4-momentum'], delete_Qbar, axis=0)
			self.Qbarlist['3-position'] = np.delete(self.Qbarlist['3-position'], delete_Qbar, axis=0)
		
		### -------------- end of recombination --------------- ###
		
		
		### -------------- update lists due to decay ---------- ###
		add_pQ = np.array(add_pQ)
		add_pQbar = np.array(add_pQbar)
		add_xQ = np.array(add_xQ)
		#add_xQbar = np.array(add_xQbar)
		if len(add_pQ):	
			# if there is at least quarkonium decays, we need to update all the three lists
			self.U1Slist['3-position'] = np.delete(self.U1Slist['3-position'], delete_U1S, axis=0) # delete along the axis = 0
			self.U1Slist['4-momentum'] = np.delete(self.U1Slist['4-momentum'], delete_U1S, axis=0)
			self.U1Slist['last_form_time'] = np.delete(self.U1Slist['last_form_time'], delete_U1S)
			
			if len(self.Qlist['4-momentum']) == 0:
				self.Qlist['3-position'] = np.array(add_xQ)
				self.Qlist['4-momentum'] = np.array(add_pQ)
			else:
				self.Qlist['3-position'] = np.append(self.Qlist['3-position'], add_xQ, axis=0)
				self.Qlist['4-momentum'] = np.append(self.Qlist['4-momentum'], add_pQ, axis=0)
			if len(self.Qbarlist['4-momentum']) == 0:
				self.Qbarlist['3-position'] = np.array(add_xQ)
				self.Qbarlist['4-momentum'] = np.array(add_pQbar)
			else:
				self.Qbarlist['3-position'] = np.append(self.Qbarlist['3-position'], add_xQ, axis=0)
				self.Qbarlist['4-momentum'] = np.append(self.Qbarlist['4-momentum'], add_pQbar, axis=0)
		
		### ---------- end of update lists due to decay ------- ###

		self.t += dt
#### ----------------- end of evolution function ----------------- ####


#### ----------------- define clear function ------------------ ####
	def dict_clear(self):
		self.Qlist.clear()
		self.Qbarlist.clear()
		self.U1Slist.clear()
#### ----------------- end of clear function ------------------ ####


#### -------------------- test gluon recombination -------------------- ####
	def test_reco_gluon(self):
		len_Q = len(self.Qlist['4-momentum'])
		len_Qbar = len(self.Qbarlist['4-momentum'])
		
		total_rate_reco_gluon = 0.0
		
		# make the periodic box 26 times bigger!
		Qbar_x_list = np.concatenate((self.Qbarlist['3-position'], 
		self.Qbarlist['3-position']+[0.0, 0.0, self.Lmax], self.Qbarlist['3-position']+[0.0, 0.0, -self.Lmax],
		self.Qbarlist['3-position']+[0.0, self.Lmax, 0.0], self.Qbarlist['3-position']+[0.0, -self.Lmax, 0.0],
		self.Qbarlist['3-position']+[self.Lmax, 0.0, 0.0], self.Qbarlist['3-position']+[-self.Lmax, 0.0, 0.0],
		self.Qbarlist['3-position']+[0.0, self.Lmax, self.Lmax], self.Qbarlist['3-position']+[0.0, self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[0.0, -self.Lmax, self.Lmax], self.Qbarlist['3-position']+[0.0, -self.Lmax, -self.Lmax],
		self.Qbarlist['3-position']+[self.Lmax, 0.0, self.Lmax], self.Qbarlist['3-position']+[self.Lmax, 0.0, -self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, 0.0, self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, 0.0, -self.Lmax],
		self.Qbarlist['3-position']+[self.Lmax, self.Lmax, 0.0], self.Qbarlist['3-position']+[self.Lmax, -self.Lmax, 0.0], self.Qbarlist['3-position']+[-self.Lmax, self.Lmax, 0.0], self.Qbarlist['3-position']+[-self.Lmax, -self.Lmax, 0.0],
		self.Qbarlist['3-position']+[self.Lmax, self.Lmax, self.Lmax], self.Qbarlist['3-position']+[self.Lmax, self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[self.Lmax, -self.Lmax, self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, self.Lmax, self.Lmax],
		self.Qbarlist['3-position']+[-self.Lmax, -self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, -self.Lmax, self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[self.Lmax, -self.Lmax, -self.Lmax]),
		axis = 0 )
			
		pair_search = cKDTree(Qbar_x_list)
		# for each Q, obtain the Qbar indexes within R_search
		pair_list = pair_search.query_ball_point(self.Qlist['3-position'], r = R_search)
		
		for i in range(len_Q):
			len_recoQbar = len(pair_list[i])
			rate_reco_gluon = []
			for j in range(len_recoQbar):		# loop over Qbar within R_search
				xQ = self.Qlist['3-position'][i]
				xQbar = Qbar_x_list[pair_list[i][j]]
				x_rel = xQ - xQbar
				i_Qbar_mod = pair_list[i][j]%len_Qbar	# use for momentum and delete_index
				#rdotp = np.sum( x_rel* (self.Qlist['4-momentum'][i][1:] - self.Qbarlist['4-momentum'][i_Qbar_mod][1:]) )
					
				#if  rdotp < 0.0:
				r_rel = np.sqrt(np.sum(x_rel**2))
				x_CM = 0.5*( xQ + xQbar )
					
				# momenta in the static medium frame
				pQ = self.Qlist['4-momentum'][i]
				pQbar = self.Qbarlist['4-momentum'][i_Qbar_mod]
						
				# CM momentum and velocity
				v_CM, v_CM_abs, p_rel_abs = LorRot.vCM_prel(pQ, pQbar, 2.0*M)
				# the factor of 2 is to account the theta function renormalization					
				rate_reco_gluon.append(self.event.get_R1S_reco_gluon(v_CM_abs, self.T, p_rel_abs, r_rel))
			
			rate_reco_gluon = np.array(rate_reco_gluon)
			total_rate_reco_gluon += np.sum(rate_reco_gluon)
		
		return 0.75*8./9.*total_rate_reco_gluon/len_Q
			
			
			
#### -------------------- test ineq recombination -------------------- ####
	def test_reco_ineq(self):
		len_Q = len(self.Qlist['4-momentum'])
		len_Qbar = len(self.Qbarlist['4-momentum'])
		
		total_rate_reco_ineq = 0.0
		
		# make the periodic box 26 times bigger!
		Qbar_x_list = np.concatenate((self.Qbarlist['3-position'], 
		self.Qbarlist['3-position']+[0.0, 0.0, self.Lmax], self.Qbarlist['3-position']+[0.0, 0.0, -self.Lmax],
		self.Qbarlist['3-position']+[0.0, self.Lmax, 0.0], self.Qbarlist['3-position']+[0.0, -self.Lmax, 0.0],
		self.Qbarlist['3-position']+[self.Lmax, 0.0, 0.0], self.Qbarlist['3-position']+[-self.Lmax, 0.0, 0.0],
		self.Qbarlist['3-position']+[0.0, self.Lmax, self.Lmax], self.Qbarlist['3-position']+[0.0, self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[0.0, -self.Lmax, self.Lmax], self.Qbarlist['3-position']+[0.0, -self.Lmax, -self.Lmax],
		self.Qbarlist['3-position']+[self.Lmax, 0.0, self.Lmax], self.Qbarlist['3-position']+[self.Lmax, 0.0, -self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, 0.0, self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, 0.0, -self.Lmax],
		self.Qbarlist['3-position']+[self.Lmax, self.Lmax, 0.0], self.Qbarlist['3-position']+[self.Lmax, -self.Lmax, 0.0], self.Qbarlist['3-position']+[-self.Lmax, self.Lmax, 0.0], self.Qbarlist['3-position']+[-self.Lmax, -self.Lmax, 0.0],
		self.Qbarlist['3-position']+[self.Lmax, self.Lmax, self.Lmax], self.Qbarlist['3-position']+[self.Lmax, self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[self.Lmax, -self.Lmax, self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, self.Lmax, self.Lmax],
		self.Qbarlist['3-position']+[-self.Lmax, -self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, -self.Lmax, self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[self.Lmax, -self.Lmax, -self.Lmax]),
		axis = 0 )
			
		pair_search = cKDTree(Qbar_x_list)
		# for each Q, obtain the Qbar indexes within R_search
		pair_list = pair_search.query_ball_point(self.Qlist['3-position'], r = R_search)
		
		for i in range(len_Q):
			len_recoQbar = len(pair_list[i])
			rate_reco_ineq = []
			for j in range(len_recoQbar):		# loop over Qbar within R_search
				xQ = self.Qlist['3-position'][i]
				xQbar = Qbar_x_list[pair_list[i][j]]
				x_rel = xQ - xQbar
				i_Qbar_mod = pair_list[i][j]%len_Qbar	# use for momentum and delete_index
				#rdotp = np.sum( x_rel* (self.Qlist['4-momentum'][i][1:] - self.Qbarlist['4-momentum'][i_Qbar_mod][1:]) )
					
				r_rel = np.sqrt(np.sum(x_rel**2))
				x_CM = 0.5*( xQ + xQbar )
					
				# momenta in the static medium frame
				pQ = self.Qlist['4-momentum'][i]
				pQbar = self.Qbarlist['4-momentum'][i_Qbar_mod]
						
				# CM momentum and velocity
				v_CM, v_CM_abs, p_rel_abs = LorRot.vCM_prel(pQ, pQbar, 2.0*M)
				
				# the factor of 2 is to account the theta function renormalization					
				rate_reco_ineq.append(self.event.get_R1S_reco_ineq(v_CM_abs, self.T, p_rel_abs, r_rel))
			
			rate_reco_ineq = np.array(rate_reco_ineq)
			total_rate_reco_ineq += np.sum(rate_reco_ineq)
		
		return 0.75*8./9.*total_rate_reco_ineq/len_Q		
			
			
#### -------------------- test ineg recombination -------------------- ####
	def test_reco_ineg(self):
		len_Q = len(self.Qlist['4-momentum'])
		len_Qbar = len(self.Qbarlist['4-momentum'])
		
		total_rate_reco_ineg = 0.0
		
		# make the periodic box 26 times bigger!
		Qbar_x_list = np.concatenate((self.Qbarlist['3-position'], 
		self.Qbarlist['3-position']+[0.0, 0.0, self.Lmax], self.Qbarlist['3-position']+[0.0, 0.0, -self.Lmax],
		self.Qbarlist['3-position']+[0.0, self.Lmax, 0.0], self.Qbarlist['3-position']+[0.0, -self.Lmax, 0.0],
		self.Qbarlist['3-position']+[self.Lmax, 0.0, 0.0], self.Qbarlist['3-position']+[-self.Lmax, 0.0, 0.0],
		self.Qbarlist['3-position']+[0.0, self.Lmax, self.Lmax], self.Qbarlist['3-position']+[0.0, self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[0.0, -self.Lmax, self.Lmax], self.Qbarlist['3-position']+[0.0, -self.Lmax, -self.Lmax],
		self.Qbarlist['3-position']+[self.Lmax, 0.0, self.Lmax], self.Qbarlist['3-position']+[self.Lmax, 0.0, -self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, 0.0, self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, 0.0, -self.Lmax],
		self.Qbarlist['3-position']+[self.Lmax, self.Lmax, 0.0], self.Qbarlist['3-position']+[self.Lmax, -self.Lmax, 0.0], self.Qbarlist['3-position']+[-self.Lmax, self.Lmax, 0.0], self.Qbarlist['3-position']+[-self.Lmax, -self.Lmax, 0.0],
		self.Qbarlist['3-position']+[self.Lmax, self.Lmax, self.Lmax], self.Qbarlist['3-position']+[self.Lmax, self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[self.Lmax, -self.Lmax, self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, self.Lmax, self.Lmax],
		self.Qbarlist['3-position']+[-self.Lmax, -self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, -self.Lmax, self.Lmax], self.Qbarlist['3-position']+[-self.Lmax, self.Lmax, -self.Lmax], self.Qbarlist['3-position']+[self.Lmax, -self.Lmax, -self.Lmax]),
		axis = 0 )
			
		pair_search = cKDTree(Qbar_x_list)
		# for each Q, obtain the Qbar indexes within R_search
		pair_list = pair_search.query_ball_point(self.Qlist['3-position'], r = R_search)
		
		for i in range(len_Q):
			len_recoQbar = len(pair_list[i])
			rate_reco_ineg = []
			for j in range(len_recoQbar):		# loop over Qbar within R_search
				xQ = self.Qlist['3-position'][i]
				xQbar = Qbar_x_list[pair_list[i][j]]
				x_rel = xQ - xQbar
				i_Qbar_mod = pair_list[i][j]%len_Qbar	# use for momentum and delete_index
				#rdotp = np.sum( x_rel* (self.Qlist['4-momentum'][i][1:] - self.Qbarlist['4-momentum'][i_Qbar_mod][1:]) )
					
				r_rel = np.sqrt(np.sum(x_rel**2))
				x_CM = 0.5*( xQ + xQbar )
					
				# momenta in the static medium frame
				pQ = self.Qlist['4-momentum'][i]
				pQbar = self.Qbarlist['4-momentum'][i_Qbar_mod]
						
				# CM momentum and velocity
				v_CM, v_CM_abs, p_rel_abs = LorRot.vCM_prel(pQ, pQbar, 2.0*M)
				
				# the factor of 2 is to account the theta function renormalization					
				rate_reco_ineg.append(self.event.get_R1S_reco_ineg(v_CM_abs, self.T, p_rel_abs, r_rel))
			
			rate_reco_ineg = np.array(rate_reco_ineg)
			total_rate_reco_ineg += np.sum(rate_reco_ineg)
		
		return 0.75*8./9.*total_rate_reco_ineg/len_Q				
		