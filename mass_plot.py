#!/usr/bin/env python3.5

import numpy as np
import argparse
from matplotlib import pyplot as plt
from astropy import constants as con
from astropy import units as u
import matplotlib.image as image


def plot(m1, m2, pb_dot, pb_dot_tot, om_dot, gamma, r, s, req_val, flag):

	if flag==0:

		plt_pb = plt.contour(m1, m2, pb_dot, [req_val[0]- req_val[5], req_val[0], req_val[0]+req_val[5]], origin='lower', colors='red', linewidths=0.5)
		plt_pb_pn = plt.contour(m1, m2, pb_dot_tot, [req_val[0]], origin='lower', colors='black', linewidths=1.0, linestyles='dashed')
		plt_om = plt.contour(m1, m2, om_dot, [req_val[1]], origin='lower', colors='blue', linewidths=0.5)
		plt_gam = plt.contour(m1, m2, gamma, [req_val[2]], origin='lower', colors='brown', linewidths=0.5)
		plt_r = plt.contour(m1, m2, r, [req_val[3]], origin='lower', colors='green', linewidths=0.5)
		plt_s = plt.contour(m1, m2, s, [req_val[4]], origin='lower', colors='purple', linewidths=0.5)

	if flag==1:

		plt_pb = plt.contour(m1, m2, pb_dot, [req_val[0]- req_val[5], req_val[0]+req_val[5]], origin='lower', colors='red', linewidths=0.5)
		plt_pb_pn = plt.contour(m1, m2, pb_dot_tot, [req_val[0]], origin='lower', colors='black', linewidths=1.0, linestyles='dashed')
		plt_om = plt.contour(m1, m2, om_dot, [0.99*req_val[1], req_val[1]], origin='lower', colors='blue', linewidths=0.5)
		plt_gam = plt.contour(m1, m2, gamma, [0.99*req_val[2], req_val[2]], origin='lower', colors='brown', linewidths=0.5)
		plt_r = plt.contour(m1, m2, r, [0.99*req_val[3], req_val[3]], origin='lower', colors='green', linewidths=0.5)
		plt_s = plt.contour(m1, m2, s, [0.99*req_val[4], req_val[4]], origin='lower', colors='purple', linewidths=0.5)
		
	leg1,_ = plt_pb.legend_elements()
	leg2,_ = plt_pb_pn.legend_elements()
	leg3,_ = plt_om.legend_elements()
	leg4,_ = plt_gam.legend_elements()
	leg5,_ = plt_r.legend_elements()
	leg6,_ = plt_s.legend_elements()

	xl = plt.xlabel(r'Mass A ($M_{\odot}$)')

	xl.set_color('red')

	yl = plt.ylabel(r'Mass B ($M_{\odot}$)')

	yl.set_color('red')

	plt.legend([leg1[0], leg2[0], leg3[0], leg4[0], leg5[0], leg6[0]],[r'$\dot{P_{b, lead}}$', r'$\dot{P_{b, 1PN}}$', r'$\dot{\omega}$', r'$\gamma$', r'$r$', r'$s$'])

	tit = plt.title('Mass- Mass plot')

	tit.set_color('red')

	#sub_axes = plt.axes([0.6,0.2,0.25,0.25])

	#dummy_plot(m1, m2, pb_dot, pb_dot_tot, om_dot, gamma, r, s, req_val, 0, sub_axes)

	#plt.savefig('mass_plot_50.pdf')
	
	plt.show()


def dummy_plot(m1, m2, pb_dot, pb_dot_tot, om_dot, gamma, r, s, req_val, flag, ax):

	if flag==0:

		plt_pb = ax.contour(m1, m2, pb_dot, [req_val[0]], origin='lower', colors='red', linewidths=0.5)
		plt_pb_pn = ax.contour(m1, m2, pb_dot_tot, [req_val[0]], origin='lower', colors='black', linewidths=1.0, linestyles='dashed')
		plt_om = ax.contour(m1, m2, om_dot, [req_val[1]], origin='lower', colors='blue', linewidths=0.5)
		plt_gam = ax.contour(m1, m2, gamma, [req_val[2]], origin='lower', colors='brown', linewidths=0.5)
		plt_r = ax.contour(m1, m2, r, [req_val[3]], origin='lower', colors='green', linewidths=0.5)
		plt_s = ax.contour(m1, m2, s, [req_val[4]], origin='lower', colors='purple', linewidths=0.5)

	if flag==1:

		plt_pb = ax.contour(m1, m2, pb_dot, [0.99*req_val[0], req_val[0]], origin='lower', colors='red', linewidths=0.5)
		plt_pb_pn = ax.contour(m1, m2, pb_dot_tot, [0.99*req_val[0], req_val[0]], origin='lower', colors='black', linewidths=1.0, linestyles='dashed')
		plt_om = ax.contour(m1, m2, om_dot, [0.99*req_val[1], req_val[1]], origin='lower', colors='blue', linewidths=0.5)
		plt_gam = ax.contour(m1, m2, gamma, [0.99*req_val[2], req_val[2]], origin='lower', colors='brown', linewidths=0.5)
		plt_r = ax.contour(m1, m2, r, [0.99*req_val[3], req_val[3]], origin='lower', colors='green', linewidths=0.5)
		plt_s = ax.contour(m1, m2, s, [0.99*req_val[4], req_val[4]], origin='lower', colors='purple', linewidths=0.5)
		
	ax.set_xlim(1.3, 1.4)
	ax.set_ylim(1.2, 1.3)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Code use to plot m1-m2 plot')

	parser.add_argument('pb', help='Period of binary in days', type=float)
	parser.add_argument('ecc', help='Eccentricity of binary', type=float)
	parser.add_argument('a1', help='Projected Semi major axis of binary in lt-sec', type=float)
	parser.add_argument('pbdot', help='Period derivative of binary', type=float)
	parser.add_argument('om', help='Periastron derivative of binary', type=float)
	parser.add_argument('gamm', help='Gravitational redshift of binary', type=float)
	parser.add_argument('ran', help='Sapiro range parameter', type=float)
	parser.add_argument('shap', help='Sapiro shape parameter', type=float)
	parser.add_argument('sig', help='1 sigma error on pb', type=float)
	args = parser.parse_args()

	c_lt = con.c.value

	G = c = con.G.value

	m_sun = con.M_sun.value

	T_sun = con.GM_sun.value/(c_lt**3)


	mp = np.linspace(1.0,2.0,1000)
	mc = np.linspace(0.5,2.0,1000)
	#mc = np.linspace(49.0,51.0,10000)
	m1, m2  = np.meshgrid(mp,mc)
	print(m1,m2)


	n = 2*np.pi/(args.pb*u.day.to(u.second))
	ecc = args.ecc
	x = args.a1*c_lt

	om_dot = 3*n*((u.radian.to(u.degree))/(u.second.to(u.year)))*(((m1 + m2)*T_sun*n)**(2.0/3))/(1-(ecc)**2)

	pb_dot = 2*np.pi*(m1*m2/(m1+m2)**2)*(((m1 + m2)*T_sun*n)**(5.0/3))*(96.0 + 292*(ecc**2) + 37*(ecc**4))/(5*((1-ecc**2)**(3.5)))

	eta = m1*m2/((m1+m2)**2.0)

	pb_dot_1PN = 2*np.pi*(m1*m2/(m1+m2)**2)*(((m1 + m2)*T_sun*n)**(5.0/3))*((-1.0/280)*(((m1 + m2)*T_sun*n)**(2.0/3))*(8828*(ecc**6)*eta -11717*(ecc**6) + 141708*(ecc**4)*eta - 197022*(ecc**4) - 219880*(ecc**2) + 159600*(ecc**2)*eta + 14784*eta - 20368))/(((1-ecc**2)**(4.5)))

	pb_dot_tot = (pb_dot + pb_dot_1PN)

	gamma = (ecc/n)*(((m1 + m2)*T_sun*n)**(2.0/3))*(m2*(m1+ 2*m2))/((m1 + m2)**2)

	r = m2

	s = x*((m1 + m2)/m2)*((G*(m1 + m2)*m_sun/(n**2))**(-1/3))

	print (pb_dot, om_dot, gamma, r, s)

	req_val = [args.pbdot, args.om, args.gamm, args.ran, args.shap, 3*args.sig]

	#dummy_plot(m1, m2, pb_dot, pb_dot_tot, om_dot, gamma, r, s, req_val, 0)

	plot(m1, m2, pb_dot, pb_dot_tot, om_dot, gamma, r, s, req_val, 0)

	