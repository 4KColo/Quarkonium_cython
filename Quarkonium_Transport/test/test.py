#!/usr/bin/env python3
import DisRec
import h5py, numpy as np, matplotlib.pyplot as plt

A = DisRec.DisRec()
"""
for T in np.linspace(0.15, 0.6, 4):
	v = np.linspace(0,0.999,100)
	R = np.array([A.get_R_dis(vv, T) for vv in v])
	plt.plot(v, R, label=r'$T = {:1.2f}$ [GeV]'.format(T))
plt.legend()
plt.show()
"""
v = 0.99
T = 0.5

q, costheta_q, phi = np.array([A.pysample_dRdq(v, T) for i in range(100000)]).T
q /= DisRec.pyE1S()
plt.subplot(3,3,1)
plt.hist(q, bins=100, range=[1,8], normed=True, histtype='step')
plt.xticks([])
plt.xlim(1., 7.9)
plt.title(r'$q/E_{1S}$')

plt.subplot(3,3,4)
plt.hist2d(q, costheta_q, range=[[1,8],[-1,1]], bins=100, normed=True)
plt.xticks([])
plt.xlim(1., 7.9)

plt.subplot(3,3,5)
plt.hist(costheta_q, range = [-1,1], bins=100, normed=True, histtype='step')
plt.xticks([])
plt.yticks([])
plt.xlim(-1., 0.99)
plt.title(r'$\cos(\theta_q)$')

plt.subplot(3,3,7)
plt.hist2d(q, phi, range=[[1,8],[1,2.*np.pi]], bins=100, normed=True)
plt.xlim(1., 7.9)

plt.subplot(3,3,8)
plt.yticks([])
plt.hist2d(costheta_q, phi, range=[[-1,1],[0,2.*np.pi]], bins=100, normed=True)
plt.xlim(-1., 0.99)

plt.subplot(3,3,9)
plt.hist(phi, bins=100, range=[0,2.*np.pi], normed=True, histtype='step')
plt.yticks([])
plt.xlim(0.0,2.*np.pi)
plt.title(r'$\phi_q$')

plt.suptitle('$v = 0.4, T = 0.3$ [GeV]')
plt.subplots_adjust(wspace=0., hspace=0.)

plt.show()
		
