
__author__ = "Brunel Odegard"
__version__ = "0.0"


import sys
import os
import math
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines  as lines
import scipy.optimize    as opt

import utils.fileio  as fileio
import utils.display as display
import utils.model   as model




sources = {
	0:"None",
	1:"Cs137",
	2:"Ba133",
	3:"Na22",
	4:"Mn54",
	5:"Cd109",
	6:"Co57",
	7:"Co60",
	8:"Pb210",
	9:"Am241",
}

class peak(object):
	"""attribute holder for peak information"""
	def __init__(self, **kwargs):
		for key,value in kwargs.items():
			self.__setattr__(key,value)
		if "sf" in kwargs.keys():
			self.e_err = 0.5 * (10**(-self.sf))
		else:
			self.e_err = 0.0

		if 'name' not in kwargs.keys():
			self.name = "{} {}".format(sources[self.src], self.e)

		if 'desc' not in kwargs.keys():
			self.desc = ""
peaks = {
	# -1: unidentified
	-1: peak(i=-1,src=0,e=0,),
	
	# 000 - 099 : no associated source

	# 0: SPE peak
	0:peak(i=0, src=0, e=0.0),
	
	# 100 - 199 : cs137
	100: peak(i=100, e=32.19  , sf=2, src=1, name="137Cs Ka", desc="Ka conversion line for Cs137"),
	101: peak(i=101, e=661.657, sf=2, src=1, name="137Cs gamma", desc=""),

	# 200 - 299 : Ba133
	200:peak(i = 200, e =  30.8510, sf=4, src=2),
	201:peak(i = 201, e =  53.1622, sf=4, src=2),
	202:peak(i = 202, e =  79.6142, sf=4, src=2),
	203:peak(i = 203, e =  80.9979, sf=4, src=2),
	204:peak(i = 204, e = 160.6121, sf=4, src=2),
	205:peak(i = 205, e = 223.237 , sf=3, src=2),
	206:peak(i = 206, e = 276.3992, sf=4, src=2),
	207:peak(i = 207, e = 302.8512, sf=4, src=2),
	208:peak(i = 208, e = 356.0134, sf=4, src=2),
	209:peak(i = 209, e = 383.8491, sf=4, src=2),

	# 300 - 399 : Na22
	300:peak(i=300, e=511.0, sf=1, src=3),

	# 400 - 499 : Mn54
	400:peak(i=400, e=239.   , sf=0, src=4),
	401:peak(i=401, e=834.848, sf=3, src=4),

	# 500 - 599 : Cd109
	500:peak(i=500, e= 22.1 , sf=1, src=5),
	501:peak(i=501, e= 88.04, sf=2, src=5),

	# 600 - 699 : Co57
	600:peak(i=600, e=  14.41300, sf=5, src=6),
	601:peak(i=601, e= 122.0614 , sf=4, src=6),
	602:peak(i=602, e= 136.4743 , sf=4, src=6),
	603:peak(i=603, e= 230.29   , sf=2, src=6),
	604:peak(i=604, e= 339.54   , sf=2, src=6),
	605:peak(i=605, e= 352.36   , sf=2, src=6),
	606:peak(i=606, e= 366.75   , sf=2, src=6),
	607:peak(i=607, e= 569.92   , sf=2, src=6),
	608:peak(i=608, e= 692.03   , sf=2, src=6),
	609:peak(i=609, e= 706.40   , sf=2, src=6),

	# 700 - 799 : Co60
	700:peak(i=700, e=346.93, sf=2, src=7),
	701:peak(i=701, e=826.06, sf=2, src=7),

	# 800 - 899 : Pb210
	800:peak(i=800, e=46.539, sf=3, src=8),

	# 900 - 999 : Am241
	900:peak(i=900, e=59.5412, sf=4, src=9),
	901:peak(i=901, e=26.3448, sf=4, src=9),
}




# bias voltage per run, in volts
bias = {
	4134:1460,
	4135:1450,
	4136:1440,
	4137:1430,
	4138:1420,
	4139:1410,
	4140:1400,
	4141:1390,
	4142:1380,
	4143:1370,
	4144:1360,
	4145:1350,
	4146:1340,
	4147:1330,
	4148:1320,
	4149:1310,
	4150:1300,
	4151:1200,
	4152:1150,
	4153:1150,
	4154:1200,
	4155:1150,
	4156:1150,
	4157:1100,
	4158:1100,
	4159:1050,
	4160:1050,
	4161:1000,
	4162:1000,
	4163:950,
	4164:950,
	4165:950,
	4166:900,
	4167:900,
	4168:900,
	4169:850,
	4170:850,
	4171:850,
	4172:800,
	4173:800,
	4174:800,
	4175:800,
	4176:800,
	4177:750,
	4178:750,
	4179:750,
	4180:750,
	4181:750,
	4182:750,
	4183:700,
	4184:700,
	4185:700,
	4186:700,
	4187:700,
	4188:700,
	4189:650,
	4190:650,
	4191:650,
	4192:650,
	4193:600,
	4194:600,
	4195:600,
}

color_by_ch = {
	"area_3046_1":"darkred", 1:"darkred", 
	"area_3046_2":"g"      , 2:"g"      , 
	"area_3046_3":"b"      , 3:"b"      , 
}

alt_color_by_ch = {
	"area_3046_1":"lightcoral"    , 1:"lightcoral"    ,
	"area_3046_2":"lightgreen"    , 2:"lightgreen"    ,
	"area_3046_3":"cornflowerblue", 3:"cornflowerblue",
}

device_by_ch = {
	"area_3046_1":"BigNaI, LG",
	"area_3046_2":"SmallNaI 1, HG",
	"area_3046_3":"SmallNaI 2, HG",
}

marker_by_src = {
	"Cd109":"d", 5:"d",
	"Ba133":"s", 2:"s",
	"Co57" :"v", 6:"v",
	"Na22" :"+", 3:"+",
	"Mn54" :"*", 4:"*",
	"Cs137":"2", 1:"2",
	"SPE"  :".", 0:".",
}

SPE_FILES = [
	"./data/fits/2022_03_22_spe.csv"
]
SRC_FILES = [
	"./data/fits/2022_03_22_src_cd109.csv",
	"./data/fits/2022_03_22_src_ba133.csv",
	"./data/fits/2022_03_22_src_co57.csv",
	"./data/fits/2022_03_22_src_na22.csv",
	"./data/fits/2022_03_22_src_mn54.csv",
	"./data/fits/2022_03_22_src_cs137.csv",
]




def model_area(nc=3, vref=1450, peak_ids_vary_e=[], ids_needed=None, constants=False, eref=None, emod=None):

			# jank but it makes it easier since we can index this with arrays
			if eref is None:
				eref = np.zeros(1 + max(peaks.keys()))

			# array to be updated and indexed in the function
			if emod is None:
				emod = eref.copy()

			for p in peaks.values():
				if p.i>=0:
					eref[p.i] = p.e
					emod[p.i] = p.e

			# compose lists of relevant and free/fixed energies
			ids_needed = sorted(ids_needed if ids_needed else [_ for _ in peaks.keys() if _>=0])
			free_e_ids = sorted([_ for _ in ids_needed if _ in peak_ids_vary_e])

			def f(x, *free_params):

				# x is a (3,npeaks) array of integers
				# (1,:) is bias voltage, in volts
				# (2,:) is peak ID (-1 for no ID, 0 for SPE, n>=100 for peak-associated)
				# (3,:) is channel ID (1, 2, or 3 here.)
				bias    = x[0]
				peak_id = x[1].astype(int)
				ch      = x[2].astype(int)

				# extract parameters
				if constants:
					fpa = np.array(free_params[3:])
					con = np.array(free_params[:3])
				else:
					fpa = np.array(free_params)
				coeff   = fpa[0*nc:1*nc] # power law coefficient
				ape_ref = fpa[1*nc:2*nc] # area of SPE event at bias voltage VREF
				epe     = fpa[2*nc:3*nc] # average energy needed to make one photoelectron
				e_adjust = fpa[3*nc:] # fractional changes to energies. E' = E*(1+this)

				# update modified energy array
				for iadj,adj in enumerate(e_adjust):
					emod[free_e_ids[iadj]] = eref[free_e_ids[iadj]] * (adj+1)
					# emod[free_e_ids[iadj]] = adj

				# result array. starts as zeros, since we're going
				# to populate it based on a criterion (source vs SPE)
				area = np.zeros(bias.shape)

				# calculate SPE area for each entry
				# for SPE events, this is the final result
				# for source events, it will be scaled by E/Espe
				# area_spe = ape_ref[ch-1] * np.exp((bias - vref) / vd[ch-1])
				area_spe = ape_ref[ch-1] * np.power((bias / vref), coeff[ch-1])

				# peak_id = 0 -> SPE
				ftr_spe = (peak_id == 0)
				area[ftr_spe] = area_spe[ftr_spe]

				# peak_id >= 100 -> source
				ftr_peak = (peak_id >= 100)
				area[ftr_peak] = area_spe[ftr_peak] * (emod[peak_id[ftr_peak]] / epe[ch[ftr_peak]-1])

				# add on constant offsets
				if constants:
					area += con[ch-1]

				# leave the area of any other peaks as zero
				return area

			return f




class routine(object):
	

	def __init__(self, args=None, **kwargs):
		for key,value in kwargs.items():
			self.__setattr__(key, value)


	def get_peak_data(self, spe_files, src_files):

		if type(spe_files) is str:
			spe_files = [spe_files]
		if type(src_files) is str:
			src_files = [src_files]

		self.spe_files = spe_files
		self.src_files = src_files
		self.spe_fits = sum([fileio.load_fits(_) for _ in spe_files], [])
		self.src_fits = sum([fileio.load_fits(_) for _ in src_files], [])

		list_v   = []
		list_id  = []
		list_ch  = []
		list_c   = []
		list_ce  = []
		list_mu  = []
		list_mue = []
		list_s   = []
		list_se  = []

		# extract peaks from SPE data
		for fit in self.spe_fits:

			if fit.ngaus != 1:
				print("Warning: entry in SPE file does not have exactly one gaussian, and will be ignored.")
				continue

			this_v   = bias[fit.run]
			this_id  = 0
			this_ch  = int(fit.fit_branch[-1])

			list_v.append(  this_v  )
			list_id.append( this_id )
			list_ch.append( this_ch )

			list_c.append(   fit.popt_gaus[0] )
			list_ce.append(  fit.perr_gaus[0] )
			list_mu.append(  fit.popt_gaus[1] )
			list_mue.append( fit.perr_gaus[1] )
			list_s.append(   fit.popt_gaus[2] )
			list_se.append(  fit.perr_gaus[2] )

		# extract peaks from source data
		for fit in self.src_fits:
			for ig,g in enumerate(fit.gaus_names):

				if not g.startswith('x'):
					continue

				this_v   = bias[fit.run]
				this_id  = int(g[1:])
				this_ch  = int(fit.fit_branch[-1])

				list_v.append(  this_v  )
				list_id.append( this_id )
				list_ch.append( this_ch )

				list_c.append(   fit.popt_gaus[0+3*ig] )
				list_ce.append(  fit.perr_gaus[0+3*ig] )
				list_mu.append(  fit.popt_gaus[1+3*ig] )
				list_mue.append( fit.perr_gaus[1+3*ig] )
				list_s.append(   fit.popt_gaus[2+3*ig] )
				list_se.append(  fit.perr_gaus[2+3*ig] )

		# convert to arrays and store
		self.v   = np.array(list_v  )
		self.id  = np.array(list_id ).astype(int)
		self.ch  = np.array(list_ch ).astype(int)
		self.c   = np.array(list_c  )
		self.ce  = np.array(list_ce )
		self.mu  = np.array(list_mu )
		self.mue = np.array(list_mue)
		self.s   = np.array(list_s  )
		self.se  = np.array(list_se )

		# set some conventience properties
		self.ids = set(self.id)


	def perform_fit(
		self,
		v_lim_by_id={},
		a_lim_by_ch={},
		exclude_ids=set(),
		peaks_vary=[],
		area_err_fudge=0.0,
		vref=1450,
		constants=False,
			):

		# tag peaks to be excluded from fit
		self.include = np.ones(self.v.shape, dtype=bool)
		for ip,p in enumerate(self.id):
		
			# peak ID is excluded
			if p in exclude_ids:
				self.include[ip] = False
				continue

			# voltage out of bounds given ID
			this_v_lim = v_lim_by_id.get(p, False)
			if this_v_lim:
				if (self.v[ip]<this_v_lim[0]) or (self.v[ip]>this_v_lim[1]):
					self.include[ip] = False
					continue

			# area out of bounds given channel
			this_a_lim = a_lim_by_ch.get(self.ch[ip], False)
			if this_a_lim:
				if (self.mu[ip]<this_a_lim[0]) or (self.mu[ip]>this_a_lim[1]):
					self.include[ip] = False
					continue


		# filter arrays
		f_v   = self.v[   self.include ]
		f_id  = self.id[  self.include ]
		f_ch  = self.ch[  self.include ]
		f_c   = self.c[   self.include ]
		f_ce  = self.ce[  self.include ]
		f_mu  = self.mu[  self.include ]
		f_mue = self.mue[ self.include ]
		f_s   = self.s[   self.include ]
		f_se  = self.se[  self.include ]

		# Adjust effective error for fitting using a supplied value for
		# systematic in measurement ability.
		self.mue_adj = np.sqrt(self.mue**2 + (area_err_fudge * self.mu)**2)
		f_mue_adj = self.mue_adj[self.include]


		# compose list of peaks to be varied.
		# no bounds on variation, currently.
		self.peaks_vary = sorted([_ for _ in peaks_vary if _ in self.ids])
		self.vary_bounds = []

		# compose fit function
		self.vref = vref
		self.eref = np.zeros(1 + max(peaks.keys()))
		self.emod = np.zeros(1 + max(peaks.keys()))
		self.fit_function = model_area(3, self.vref, self.peaks_vary, set(self.ids), eref=self.eref, emod=self.emod)

		# Stack data into format expected by fit function.
		# Note that this converts all data into floats, since vias
		# voltage is a float value. To index with id and channel, we
		# will need to un-stack and covert back to integer arrays.
		xdata   = np.stack((self.v, self.id, self.ch), axis=0)
		f_xdata = np.stack((f_v, f_id, f_ch), axis=0)


		# compose initial parameter guess
		p0_list = []

		# constants
		# just start with zeros
		if constants:
			p0_const = [0.0] * 3
			p0_list += p0_const

		# Power law coefficients:
		# based on previous fit results, this is pretty close.
		p0_coeff = [6.7, 6.7, 6.7]
		p0_list += p0_coeff

		# SPE area at voltage defined by vref:
		# values read off SPE curves at vref of 1450V.
		p0_ape_ref = [3628.0, 5475.0, 6040.0]
		p0_list += p0_ape_ref
		
		# SPE energy (average energy per PE produced:)
		# rough calculation using Co57 at 900V and SPE at 1450V.
		p0_epe = [0.0723, 0.1040, 0.1105]
		p0_list += p0_epe
		
		# Peak energy adjustments for peaks allowed to vary:
		# start with unchanged peak energy
		p0_adj = [0.0] * len(peaks_vary)
		# p0_adj = [peaks[_].e for _ in peaks_vary]
		p0_list += p0_adj
		
		# combined parameter guess
		p0 = np.array(p0_list)


		# perform the fit
		self.popt, self.pcov = opt.curve_fit(
			f = self.fit_function, 
			xdata = f_xdata,
			ydata = f_mu,
			p0 = p0,
			sigma = f_mue_adj,
			absolute_sigma = True,
		)
		print(self.emod[self.emod>0])
		print(self.eref[self.eref>0])
		print("\n")

		# print fit results nicely
		self.perr = np.sqrt(np.diag(self.pcov))
		pnames = [""]*9 + ["{:>3} ({:>5.1f} KeV)".format(_,peaks[_].e) for _ in self.peaks_vary]
		print("fit results: parameter \xb1 error")
		for ipar,par in enumerate(self.popt):
			print("{:>16} {:>11.4f} \xb1 {:>9.6f}".format(pnames[ipar], par, self.perr[ipar]))
		print("\n")

		# store parameters in convienent form for access
		i0 = 3 if constants else 0
		self.k_opt   = self.popt[i0+0:i0+3]
		self.k_err   = self.perr[i0+0:i0+3]
		self.apr_opt = self.popt[i0+3:i0+6]
		self.apr_err = self.perr[i0+3:i0+6]
		self.epe_opt = self.popt[i0+6:i0+9]
		self.epe_err = self.perr[i0+6:i0+9]
		self.adj_opt = self.popt[i0+9:]
		self.adj_err = self.perr[i0+9:]

	
		# calculate modeled values for peak locations (mu)
		# note that this includes peaks excluded from fit routine,
		# so in order to calculate chi2 and ndof for the fit, the same
		# filter must be applied to these arrays as for the fit routine.
		self.mu_opt = self.fit_function(xdata, *self.popt)
		self.mu_res = self.mu - self.mu_opt
		f_mu_opt = self.mu_opt[self.include]
		f_mu_res = self.mu_res[self.include]

		# calculate error on optimal mu
		# TODO: for now, this is set to zero
		self.mue_opt = np.zeros(self.mu_opt.shape)

		# # print model results per peak
		# lines = {False:[],True:[]}
		# for ip,p in enumerate(self.id):
		# 	lines[self.include[ip]].append(
		# 		"{:>3} {:>3} {:>12.2f} {:>12.2f} {:>10.2f} {:>8.2f}".format(p, self.include[ip], self.mu_opt[ip], self.mu[ip], self.mu_res[ip], self.mue[ip])
		# 	)
		# print('\n'.join(lines[True]))
		# print("")
		# print('\n'.join(lines[False]))
		# print("")

		# create energy and adjusted energy arrays
		# todo: calculate error on energy using source peak energy errors
		# todo: calculate error on adjusted energy using error propagaion of adjustment parameters
		self.e      = np.zeros(self.id.shape) # energy
		self.ee     = np.zeros(self.id.shape) # error on energy
		self.e_adj  = np.zeros(self.id.shape) # adjusted energy based on fit
		self.ee_adj = np.zeros(self.id.shape) # error on adjusted energy

		ftr_src = (self.id>0)
		ftr_spe = np.logical_not(ftr_src)

		self.e[  ftr_src ] = self.eref[    self.id[ftr_src] ]
		self.e[  ftr_spe ] = self.epe_opt[ self.ch[ftr_spe]-1 ]
		self.ee[ ftr_src ] = 0.0 # self.eref[    self.id[ftr_src] ]
		self.ee[ ftr_spe ] = self.epe_err[ self.ch[ftr_spe]-1 ]

		self.e_adj[  ftr_src ] = self.emod[    self.id[ftr_src] ]
		self.e_adj[  ftr_spe ] = self.epe_opt[ self.ch[ftr_spe]-1 ]
		self.ee_adj[ ftr_src ] = 0.0 # self.eref[    self.id[ftr_src] ]
		self.ee_adj[ ftr_spe ] = self.epe_err[ self.ch[ftr_spe]-1 ]

		# calculate and print chi2, ndof
		self.chi2 = ((f_mu_res / f_mue_adj)**2).sum()
		self.ndof = f_mu_res.size - p0.size
		self.gof_string = "chi2/ndof = {:>4.2f} / {:<4} = {:<2.4f}".format(self.chi2,self.ndof,self.chi2/self.ndof)
		print(self.gof_string)
		print("")


	def display_results(
		self,
		plot_channels = [1,2,3],
		plot_separate = True,
		residuals = False,
		draw_data = True,
		draw_fits = True,
		label_sources = True,
		hlines=[],
		vlines=[],
		data_ls="",
		suptitle="{gof}",
			):


		# create figure and axes
		nrows = 1
		ncols = len(plot_channels) if plot_separate else 1
		fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(ncols*8-1,nrows*8-1))
		if nrows*ncols == 1:
			ax = [ax]

		ch_ax = {
			1:ax[min([ncols, 1]) - 1],
			2:ax[min([ncols, 2]) - 1],
			3:ax[min([ncols, 3]) - 1],
		}

		id_vsep = 0.15
		pad_model_x = 25.0
		res_lines = [('k',0.0,'-','model'),('b',0.01,'--','\xb1{}%'), ('c',0.03,'--','\xb1{}%')]

		# residual plots
		if residuals:

			offsets = {n:(i+1)*id_vsep for i,n in enumerate(sorted(self.ids, key=lambda i:peaks[i].e))}

			frac_res     = 1 - self.mu_opt/self.mu
			frac_res_err = abs(self.mu_opt / self.mu) * np.sqrt( (self.mue_adj / self.mu)**2 + (self.mue_opt / self.mu_opt)**2 )

			for ich,ch in enumerate(plot_channels):
				this_ax = ch_ax[ch]
				
				for _id in self.ids:

					ftr_this = (self.ch == ch) & (self.id == _id)
					
					if draw_data:

						this_incl = self.include[ftr_this]
						this_excl = np.logical_not(this_incl)

						if any(this_incl):
							this_ax.errorbar(
								self.v[ftr_this][this_incl],
								frac_res[ftr_this][this_incl] + offsets[_id],
								frac_res_err[ftr_this][this_incl],
								color=color_by_ch[ch],
								marker=marker_by_src[_id//100],
								ls='',
							)
						if any(this_excl):
							this_ax.errorbar(
								self.v[ftr_this][this_excl],
								frac_res[ftr_this][this_excl] + offsets[_id],
								frac_res_err[ftr_this][this_excl],
								color='k',
								marker=marker_by_src[_id//100],
								ls='',
							)

					if draw_fits:

						this_v = self.v[ftr_this]
						this_xaxis = np.linspace(this_v.min()-pad_model_x, this_v.max()+pad_model_x, 10)
						this_ones  = np.ones(this_xaxis.shape)

						# this_ax.plot(this_xaxis, this_ones * offsets[_id], 'k-')
						for col,val,ls,fmt in res_lines:
							this_ax.plot(this_xaxis, this_ones * (offsets[_id] + val), color=col, ls=ls)
							if abs(val)>0:
								this_ax.plot(this_xaxis, this_ones * (offsets[_id] - val), color=col, ls=ls)


		# absolute plots
		else:

			for ich,ch in enumerate(plot_channels):
				this_ax = ch_ax[ch]
				
				for _id in self.ids:

					ftr_this = (self.ch == ch) & (self.id == _id)
					
					if draw_data:

						this_incl = self.include[ftr_this]
						this_excl = np.logical_not(this_incl)
						
						if any(this_incl):
							this_ax.errorbar(
								self.v[ftr_this][this_incl],
								self.mu[ftr_this][this_incl],
								self.mue_adj[ftr_this][this_incl],
								color=color_by_ch[ch],
								marker=marker_by_src[_id//100],
								ls='',
							)

						if any(this_excl):
							this_ax.errorbar(
								self.v[ftr_this][this_excl],
								self.mu[ftr_this][this_excl],
								self.mue_adj[ftr_this][this_excl],
								color=alt_color_by_ch[ch],
								marker=marker_by_src[_id//100],
								ls='',
							)

					if draw_fits:

						this_v = self.v[ftr_this]
						
						this_bias = np.linspace(this_v.min()-pad_model_x, this_v.max()+pad_model_x, 10)
						this_ones  = np.ones(this_bias.shape)
						this_xdata = np.stack([this_bias, this_ones * _id, this_ones * ch], axis=0)

						this_ax.plot(
							this_bias,
							self.fit_function(this_xdata, *self.popt),
							color=color_by_ch[ch],
							marker='',
							ls='--',
						)



		devices = ["BigNaI", "SmallNaI 1", "SmallNaI 2", "big LYSO"]
		for ch in plot_channels:

			ax = ch_ax[ch]

			if plot_separate:
				ax.set_title("channel {} - {}".format(ch, devices[ch-1]))

			legend_items = []

			if not plot_separate:
				legend_items += [
					lines.Line2D([],[],color=color_by_ch[1],ls='',marker='o',label='ch1 (BigNaI, LG)'),
					lines.Line2D([],[],color=color_by_ch[2],ls='',marker='o',label='ch2 (SmallNaI 1, HG)'),
					lines.Line2D([],[],color=color_by_ch[3],ls='',marker='o',label='ch3 (SmallNaI 2, HG)'),
				]

			if label_sources:
				# custom labels
				# legend_items.append(lines.Line2D([],[],color='k',ls='',marker='.',label='SPE'))
				legend_items += [lines.Line2D([],[],color='k',ls='',marker=mkr,label=src) for src,mkr in marker_by_src.items() if type(src) is str]

			if residuals:
				legend_items += [lines.Line2D([],[],color=c,ls=l,marker='',label=f.format(int(v*100))) for c,v,l,f in res_lines]

			for values,c,l,label in hlines:
				# single value -> same for each channel
				if type(values) in (int,float):
					values=[values]*3
				ax.axhline(values[ch-1],color=c,ls=l)
				legend_items.append(lines.Line2D([],[],color=c,ls=l,marker='',label=label.format(values[ch-1])))

			if legend_items:
				ax.legend(handles=legend_items)

			# markup and show plot
			ax.set_xlabel("bias (volts)")
			ax.set_ylabel("fractional residuals" if residuals else "area (pVs)")
			ax.set_xscale("log")
			if not residuals:
				ax.set_yscale("log")
		
		fig.subplots_adjust(left=0.04, right=0.96, bottom=0.1, top=0.9, hspace=0.1, wspace=0.1)
		if suptitle:
			plt.suptitle(suptitle.format(gof=self.gof_string))
		plt.show()


	def display_energy_adjustments(self):
		...

	def analyze_mu_vs_sigma(
		self,
		plot_channels = [1,2,3],
		plot_separate=True,
		label_sources = True,
		suptitle="{gof}",
			):

		def model_width(ids):

			ids = np.array(sorted(set(ids)))
			w_id = np.zeros(max(ids)+1)
			w_id[ids] = 1

			def f(xdata, *params):
				"""model width of area distribution based on device, energy, and mu"""

				# unpack xdata
				fit_s,fit_mu,fit_ch,fit_id,fit_e = xdata
				fit_id = fit_id.astype(int)
				fit_ch = fit_ch.astype(int)

				# unpack parameters
				w_ch = np.array(params[0:3])
				# c_ch = np.array(params[3:6])
				w_id[ids] = np.array(params[3:])

				# calculate intermediate quantities
				one_over_root_npe = np.sqrt(self.epe_opt[fit_ch-1] / fit_e)

				sigma_res   = fit_mu * one_over_root_npe * w_ch[fit_ch-1] * w_id[fit_id]
				# sigma_smear = fit_mu * w_id[fit_id]
				sigma_smear = 0.0

				# sigma = sigma_res * sigma_smear
				sigma = np.sqrt(sigma_res**2 + sigma_smear**2)

				return sigma
				# return fit_mu * w_ch[fit_ch-1] * w_id[fit_id] * one_over_root_npe + c_ch[fit_ch-1] * np.sqrt(fit_mu)

			return f


		fit_mu  = self.mu[  self.include ]
		fit_mue = self.mue[ self.include ]
		fit_s   = self.s[   self.include ]
		fit_se  = self.se[  self.include ]
		fit_ch  = self.ch[  self.include ]
		fit_id  = self.id[  self.include ]
		fit_e   = self.e[   self.include ]
		fit_xdata = np.stack([fit_s,fit_mu,fit_ch,fit_id,fit_e], axis=0)


		ids = sorted(set(fit_id))
		p0  = [1.0] * 3
		# p0 += [0.0] * 3
		i_w_id = len(p0)
		p0 += [1.0] * len(ids)

		# perform the fit
		fit_model = model_width(ids)
		popt, pcov = opt.curve_fit(
			f = fit_model, 
			xdata = fit_xdata,
			ydata = fit_s,
			p0 = p0,
			sigma = fit_se,
			absolute_sigma = True,
		)
		perr = np.sqrt(np.diag(pcov))

		print(popt)
		print(perr)

		# calculate and print chi2, ndof
		chi2 = (((fit_s - fit_model(fit_xdata, *popt))/fit_se)**2).sum()
		ndof = fit_s.size - len(p0)
		gof_string = "chi2/ndof = {:>4.2f} / {:<4} = {:<2.4f}".format(chi2,ndof,chi2/ndof)
		print(gof_string)
		print("")


		# e   = np.array([peaks[_].e for _ in ids]) 
		# pf  = np.array(popt[i_w_id:])
		# pfe = np.array(perr[i_w_id:])

		# pf_popt, pf_pcov = opt.curve_fit(
		# 	lambda x,*p:p[0] + p[1]*np.sqrt(x),
		# 	xdata=e[1:],
		# 	ydata=pf[1:],
		# 	p0=np.array([0.0,0.001]),
		# 	sigma=pf[1:]*0.01,
		# 	absolute_sigma=True,
		# )
		# print(pf_popt)
		# plt.plot(
		# 	e,
		# 	pf_popt[0]+pf_popt[1]*np.sqrt(e),
		# 	'r.'
		# )

		plt.errorbar(
			np.array([peaks[_].e for _ in ids]),
			popt[i_w_id:],
			perr[i_w_id:],
			# [0.0 for _ in ids],
			color='k',
			marker='.',
			ls=''
		)
		plt.xlabel('peak energy (KeV)')
		plt.ylabel('Fp (peak factor)')
		plt.show()





		# create figure and axes
		nrows = 1
		ncols = len(plot_channels) if plot_separate else 1
		fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(ncols*8-1,nrows*8-1))
		if nrows*ncols == 1:
			ax = [ax]

		ch_ax = {
			1:ax[min([ncols, 1]) - 1],
			2:ax[min([ncols, 2]) - 1],
			3:ax[min([ncols, 3]) - 1],
		}

		# plot stuff
		for ich,ch in enumerate(plot_channels):
			this_ax = ch_ax[ch]
			
			for _id in self.ids:

				ftr_this = (self.ch == ch) & (self.id == _id)

				this_incl = self.include[ftr_this]
				this_excl = np.logical_not(this_incl)

				this_mu  = self.mu[  ftr_this ][this_incl]
				this_s   = self.s[   ftr_this ][this_incl]
				this_se  = self.se[  ftr_this ][this_incl]
				this_mue = self.mue[ ftr_this ][this_incl]
				this_e   = self.e[   ftr_this ][this_incl]
				this_ch = np.ones(this_mu.shape, dtype=int) * ch
				this_id = np.ones(this_mu.shape, dtype=int) * _id
				this_xdata = np.stack([this_s,this_mu,this_ch,this_id,this_e])
				
				if any(this_incl):
					this_ax.errorbar(
						this_mu ,
						this_s  ,
						this_se ,
						this_mue,
						color=color_by_ch[ch],
						marker=marker_by_src[_id//100],
						ls='',
					)

				if any(this_excl):
					this_ax.errorbar(
						self.mu[  ftr_this ][this_excl],
						self.s[   ftr_this ][this_excl],
						self.se[  ftr_this ][this_excl],
						self.mue[ ftr_this ][this_excl],
						color=alt_color_by_ch[ch],
						marker=marker_by_src[_id//100],
						ls='',
					)

				# plot fit results
				this_ax.plot(
					this_mu,
					fit_model(this_xdata, *popt),
					color=color_by_ch[ch],
					marker='',
					ls='--',
				)


		devices = ["BigNaI", "SmallNaI 1", "SmallNaI 2", "big LYSO"]
		for ch in plot_channels:

			ax = ch_ax[ch]

			if plot_separate:
				ax.set_title("channel {} - {}".format(ch, devices[ch-1]))

			legend_items = []

			if not plot_separate:
				legend_items += [
					lines.Line2D([],[],color=color_by_ch[1],ls='',marker='o',label='ch1 (BigNaI, LG)'),
					lines.Line2D([],[],color=color_by_ch[2],ls='',marker='o',label='ch2 (SmallNaI 1, HG)'),
					lines.Line2D([],[],color=color_by_ch[3],ls='',marker='o',label='ch3 (SmallNaI 2, HG)'),
				]

			if label_sources:
				# custom labels
				# legend_items.append(lines.Line2D([],[],color='k',ls='',marker='.',label='SPE'))
				legend_items += [lines.Line2D([],[],color='k',ls='',marker=mkr,label=src) for src,mkr in marker_by_src.items() if type(src) is str]

			# if residuals:
			# 	legend_items += [lines.Line2D([],[],color=c,ls=l,marker='',label=f.format(int(v*100))) for c,v,l,f in res_lines]

			# for values,c,l,label in hlines:
			# 	# single value -> same for each channel
			# 	if type(values) in (int,float):
			# 		values=[values]*3
			# 	ax.axhline(values[ch-1],color=c,ls=l)
			# 	legend_items.append(lines.Line2D([],[],color=c,ls=l,marker='',label=label.format(values[ch-1])))

			if legend_items:
				ax.legend(handles=legend_items)

			# markup and show plot
			ax.set_xlabel("mu (pVs)")
			ax.set_ylabel("sigma (pVs)")
			ax.set_xscale("log")
			ax.set_yscale("log")
		
		fig.subplots_adjust(left=0.04, right=0.96, bottom=0.1, top=0.9, hspace=0.1, wspace=0.1)
		if suptitle:
			plt.suptitle(suptitle.format(gof=gof_string))
		plt.show()








def gain_curve(

	v_lim_by_id = {},
	a_lim_by_ch = {},
	exclude_ids = set(),
	peaks_vary = [],
	area_err_fudge = 0.0,
	vref=1450,

	plot_channels = [1,2,3],
	plot_separate = True,
	draw_data = True,
	draw_fits = True,
	label_sources = True,
	data_ls = "",

	hlines=[],
	vlines=[],

	suptitle="",


		):

	rtn = routine()
	rtn.get_peak_data(SPE_FILES, SRC_FILES)
	rtn.perform_fit(
		v_lim_by_id=v_lim_by_id,
		a_lim_by_ch=a_lim_by_ch,
		exclude_ids=exclude_ids,
		peaks_vary=peaks_vary,
		area_err_fudge=area_err_fudge,
		vref=vref,
		constants=False,
	)

	rtn.analyze_mu_vs_sigma()
	sys.exit(0)
	
	rtn.display_results(
		plot_channels = [1,2,3],
		plot_separate = True,
		residuals = False,
		draw_data = True,
		draw_fits = True,
		label_sources = True,
		data_ls = "",
		hlines=hlines,
		vlines=vlines,
		suptitle=suptitle,
	)

	rtn.display_results(
		plot_channels = [1,2,3],
		plot_separate = True,
		residuals = True,
		draw_data = True,
		draw_fits = True,
		label_sources = True,
		data_ls = "",
		suptitle=suptitle,
	)

	sys.exit(0)





def main():


	# inclusive limits per peak ID
	v_lim_by_id = {
		0  :[1430, np.inf],
		
		# 200:[ 751, np.inf],
		# 203:[ 701, np.inf],
		200:[ 751, 1099],
		203:[ 751, np.inf],

		500:[ 801, 1149],
		501:[ 701, np.inf],

		401:[-np.inf, 750],
	}

	# 100000 130000 190000 230000
	a_lim_hi = 130000
	a_lim_by_ch = {
		1:[-np.inf, a_lim_hi],
		2:[-np.inf, a_lim_hi],
		3:[-np.inf, a_lim_hi],
	}

	# exclude some ids altogether (from the fit)
	# exclude_ids = {101, 300, 401, }
	exclude_ids = set()


	# all but Co57
	peaks_vary = {101,200,203,300,401,500,501,601} - {601}

	# # just high energy + Cd109-22KeV
	# peaks_vary = {101, 300, 401, 500}

	# # none
	# peaks_vary = set()


	# additional error on peak locations, fraction of location value
	# area_err_fudge = 0.004
	area_err_fudge = 0.0


	# hlines=[
	# 	(100000,'k','-','{} pVs'),
	# 	(130000,'k','-','{} pVs'),
	# 	(188000,'k','-','{} pVs'),
	# 	(230000,'k','-','{} pVs'),
	# ]
	hlines=[]
	vlines=[]


	suptitle="{a_lim} | {gof}".format(
		a_lim = "area < {} nVs".format(a_lim_hi//1000),
		gof = '{gof}',
	)

	gain_curve(
		
		v_lim_by_id = v_lim_by_id,
		a_lim_by_ch = a_lim_by_ch,
		exclude_ids = exclude_ids,
		peaks_vary = peaks_vary,
		area_err_fudge = area_err_fudge,

		draw_data = True,
		draw_fits = True,

		hlines=hlines,
		vlines=vlines,

		suptitle=suptitle,

	)





if __name__ == "__main__":
	main()

