"""
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"


import os
import sys
import math

import numpy as np

import utils.data as data
import utils.fit as fit

import matplotlib.pyplot as plt
import scipy.optimize as opt




sim_version = 1
file = './data/spectra/simv{}_ch{}.npz'
CH_COMP = [1,2,3]

SOURCES = [
	"Am241",
	"Cd109",
	"Ba133",
	"Co57" ,
]

MIX_AMOUNT = 1.00
MIX_ABS = False
MIX = {
	"Am241":MIX_AMOUNT,
	"Cd109":MIX_AMOUNT,
	"Ba133":MIX_AMOUNT,
	"Co57" :MIX_AMOUNT,
}

COL_TRUE = {
	"Am241":"violet",
	"Cd109":"yellow",
	"Ba133":"lightcoral",
	"Co57" :"cyan",
}
COL_FIT  = {
	"Am241":"darkviolet",
	"Cd109":"tab:olive",
	"Ba133":"darkred",
	"Co57" :"steelblue",
}

def comp(ch, arrays, show=True):
	print("\n\n")

	# print("")
	# for _ in arrays.keys():
	# 	print(_)
	
	# all edges are same
	this_edges = arrays["a_edges_Am241"]
	this_mids  = arrays["a_mids_Am241"]
	lo = this_edges[0]
	hi = this_edges[-1]

	# compose mixture
	events_src = {}
	counts_src = {}
	mix_actual = {}
	for src in SOURCES:
		this_branch = "exp_area_nVs_3046_{}_{}".format(ch, src)
		this_data = arrays[this_branch]
		this_data = this_data[np.logical_and(this_data>lo, this_data<hi)]
		
		if MIX_ABS:
			this_count = min([MIX[src], this_data.size])
		else:
			this_count = min([MIX[src] * this_data.size, this_data.size])
		this_count = math.floor(this_count)

		mix_actual[src] = this_count
		this_sample = np.random.choice(this_data, this_count, replace=False)
		counts_src[src] = this_count
		print("added {} out of {} events for {}".format(this_count, this_data.size, src))
		events_src[src]=this_sample
	events = np.concatenate([events_src[_] for _ in SOURCES], axis=0)
	
	sample_spec = np.histogram(events, this_edges)[0]
	sample_spec_err = np.sqrt(sample_spec)

	# plot mixture
	plt.step(this_mids, sample_spec, 'k-', where='mid', label='samples, mixed')
	for src in SOURCES:
		plt.step(
			this_mids,
			np.histogram(events_src[src], this_edges)[0],
			where='mid', label='sample, {}'.format(src),
			color=COL_TRUE[src],
		)

	# compose model
	src_spec     = {_:arrays["ym_{}".format(    _)] for _ in SOURCES}
	src_spec_err = {_:arrays["ym_err_{}".format(_)] for _ in SOURCES}

	src_spec_norm     = {k:v/np.sum(v) for k,v in src_spec.items()}
	src_spec_norm_err = {k:src_spec_err[k]/np.sum(v) for k,v in src_spec.items()}

	
	def mix(xdata, *amp):
		return np.sum(np.stack([a*src_spec_norm[SOURCES[ia]] for ia,a in enumerate(amp)],axis=1), axis=1)

	# plot truth fit
	amp_true = [mix_actual[_] / np.sum(src_spec_norm[_]) for _ in SOURCES]
	print("true amplitudes: ".format(amp_true))
	mix_truth = mix(None, *amp_true)
	plt.step(this_mids, mix_truth, 'c-', where='mid', label='model, true mix')

	# fit
	# guess equal mixture: 0.25 for each amplitude
	amp_p0 = [events.size * 0.25] * 4
	amp_popt, amp_pcov = opt.curve_fit(
		f = mix,
		xdata = this_mids,
		ydata = sample_spec,
		sigma          = sample_spec_err,
		absolute_sigma = True,
		p0 = amp_p0,
		# bounds = [0, np.inf],
	)
	amp_perr = np.sqrt(np.diag(amp_pcov))

	mix_popt = mix(None, *amp_popt)
	plt.step(this_mids, mix_popt, color='tab:brown', where='mid', label='model, best fit')
	for isrc,src in enumerate(SOURCES):
		plt.step(
			this_mids,
			src_spec_norm[src] * amp_popt[isrc],
			where='mid', label='best fit, {}'.format(src),
			color=COL_FIT[src],
		)


	ets = "{:>12}"
	etf = "{:>12.3f}"
	print("\nfit results")
	print(' ,'.join([ets.format(_) for _ in ["ch{} {}%".format(ch, int(MIX_AMOUNT*100))] + SOURCES]))
	print(' ,'.join([ets.format("p0"      )] + [etf.format(_) for _ in amp_p0  ]))
	print(' ,'.join([ets.format("truth"   )] + [etf.format(_) for _ in amp_true]))
	print(' ,'.join([ets.format("popt"    )] + [etf.format(_) for _ in amp_popt]))
	print(' ,'.join([ets.format("perr"    )] + [etf.format(_) for _ in amp_perr]))
	print(' ,'.join([ets.format("pull"    )] + [etf.format(_) for _ in (amp_true-amp_popt)/amp_perr]))
	print(' ,'.join([ets.format(f"% diff" )] + [etf.format(_) for _ in 100*(amp_true-amp_popt)/amp_true]))


	# decorate and show
	plt.title('mixture and components')
	plt.legend()
	plt.xlabel('area (nVs)')
	if show:
		plt.show()





def main():
	arrays = {ch:np.load(file.format(sim_version, ch)) for ch in CH_COMP}
	print([_ for _ in arrays[1].keys()])
	for ich,ch in enumerate(CH_COMP):
		plt.subplot(1,3,ich+1)
		comp(ch, arrays[ch], show=False)
	plt.show()




if __name__ == "__main__":
	main()
