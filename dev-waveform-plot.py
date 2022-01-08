"""
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"


import sys
import os
import math
import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils.fileio  as fileio
import utils.display as display
import utils.model   as model

def afw(wvf):

	n = wvf.shape[0]
	s = wvf.shape[1]

	pre  = wvf[:,:150]
	post = wvf[:,150:]

	pedestal = pre.mean(-1).reshape([n,1])
	pednoise = pre.std(-1).reshape([n,1])

	pulse = post * (post > (pedestal - 3*pednoise)).astype(int)

	vmax = pulse.max(-1)
	area = pulse.sum(-1)

	return vmax, area


def area_from_waveform(waveform):
	
	pre  = waveform[:150]
	post = waveform[150:]

	# pedestal = avg and stdev of up to sample 150
	pedestal = pre.mean()
	pednoise = np.std(pre)

	print(pedestal, pednoise)

	# count eveything that's not more than 3 stdevs below pedestal
	nonnegative = post > (pedestal - 3*pednoise)
	pulse = post[nonnegative] - pedestal

	# pulse = abs(post-pedestal)

	vmax = pulse.max()
	area = pulse.sum()

	return vmax, area


if __name__ == '__main__':

	# DIR_NI0 = os.sep.join([os.getcwd(), "data/root/scintillator/Run{}.root"])
	DIR_NI1     = os.sep.join([os.getcwd(), "data/root/noiseinjected/1mv/Run{}.root"])
	DIR_NI1_MOD = os.sep.join([os.getcwd(), "data/root/noiseinjected/1mv/{}mv/Run{}.npz"])
	# DIR_NI5 = os.sep.join([os.getcwd(), "data/root/noiseinjected/5mv/Run{}.root"])

	v_branches = {"voltages_2988_1":True, "voltages_2988_2":True}

	# which_run = 3427
	# runs_process = [3427]
	runs_process = range(3427, 3443)
	noise_amounts = [25]#[0,1,5]

	for which_run in runs_process:

		voltages = fileio.load_branches(DIR_NI1.format(which_run), v_branches)
		v1 = voltages["voltages_2988_1"]
		v2 = voltages["voltages_2988_2"]

		for noise_amount in noise_amounts:

			if noise_amount:
				vmax1,area1 = afw(v1 + np.random.normal(0, noise_amount, v1.shape))
				vmax2,area2 = afw(v2 + np.random.normal(0, noise_amount, v2.shape))
			else:
				vmax1,area1 = afw(v1)
				vmax2,area2 = afw(v2)

			np.savez(
				DIR_NI1_MOD.format(noise_amount, which_run),
				vMax_2988_1=vmax1,
				area_2988_1=area1,
				vMax_2988_2=vmax2,
				area_2988_2=area2,
			)

			del vmax1
			del vmax2
			del area1
			del area2

			print("{}mV done".format(noise_amount))

		del voltages
		del v1
		del v2

		print("completed processing run {}".format(which_run))
		print("")

	



	# test_events = [0, 405, 4500, 8002, 12323, 15999]
	# x=np.linspace(0,1023,1024)
	# for test_event in test_events:

	# 	print("event {}".format(test_event))
	# 	pch1 = area_from_waveform(v1[test_event])
	# 	print(pch1)
	# 	pch5 = area_from_waveform(v5[test_event])
	# 	print(pch5)
	# 	print("")

		# plt.plot(x,v1[test_event],'r-',label="+1")
		# plt.plot(x,v5[test_event],'b-',label="+5")
		# plt.xlabel("sample number")
		# plt.ylabel("voltage")
		# plt.title("+1mV vs. +5mV datasets\nevent #{}".format(test_event))
		# plt.legend()
		# plt.savefig("./figs/ni_waveform_ev{}_abs".format(test_event))
		# plt.clf()
		# # plt.show()

		# plt.plot(x,v5[test_event] - v1[test_event],'k-',label="v5-v1")
		# plt.xlabel("sample number")
		# plt.ylabel("voltage difference")
		# plt.title("difference between +1mV vs. +5mV datasets\nevent #{}".format(test_event))
		# plt.legend()
		# plt.savefig("./figs/ni_waveform_ev{}_rel".format(test_event))
		# plt.clf()
		# # plt.show()