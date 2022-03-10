"""
script used for testing features during development
don't actually use this for anything else
"""

import numpy
import random
import math
import numpy as np

import matplotlib.pyplot as plt

import utils.fileio as fileio
import utils.data   as data




test_branch_manager = False
if test_branch_manager:

	test_file = '../xrd-analysis/data/root/scintillator/Run4039.root'
	branches = fileio.load_branches(test_file, {"area_3046_1","vMax_3046_1","tMax_3046_1","scaler_3046_1","area_3046_1","tMax_3046_2","scaler_3046_2"})

	bm = data.BranchManager(branches, export_copies=False, import_copies=False, )

	print("look at properties")
	print("keys: {}".format(bm.keys))
	print("")

	print("new branch by scaling area down by factor of 1000")
	print(bm["area_3046_1"])
	bm.bud([lambda m:{"area_3046_1_nvs":m["area_3046_1"]/1000.0}])
	print(bm["area_3046_1_nvs"])
	print("")

	print("overwrite scaler branches via convolving")
	print(bm["scaler_3046_1"])
	plt.hist(bm["scaler_3046_1"], bins=np.linspace(0,3000,100), histtype='step', log=True, color='k', label='raw')
	bm.bud([data.rectify_scalers()],overwrite=True)
	bm.bud([data.convolve_branch("scaler_3046_1",12,"con")],overwrite=False)
	print(bm["scaler_3046_1"])
	plt.hist(bm["scaler_3046_1"]    , bins=np.linspace(0,3000,100), histtype='step', log=True, color='g', label='with rectification')
	plt.hist(bm["scaler_3046_1_con"], bins=np.linspace(0,3000,100), histtype='step', log=True, color='b', label='with convolution')
	plt.xlabel('scaler_3046_1')
	plt.legend()
	plt.show()
	print("")

	print("make a cut comparison without applying mask to whole dataset")
	area_all = bm["area_3046_1_nvs"]
	mask = data.mask_any(
		data.cut("tMax_3046_1",500,np.inf),
		data.cut("area_3046_1_nvs",30,50),
		data.cut("scaler_3046_1",1500),
		)
	area_cut = bm.mask(mask, "area_3046_1_nvs")
	plt.hist(area_all, bins=np.logspace(1,3,500), histtype='step', log=True, color='k', label='all')
	plt.hist(area_cut, bins=np.logspace(1,3,500), histtype='step', log=True, color='g', label='tmax>500 or 30<area<50')
	plt.xscale("log")
	plt.xlabel("area_3046_1 (nVs)")
	plt.legend()
	plt.show()
	print("")

	print("apply a cut to the whole dataset")
	plt.hist(bm["area_3046_1_nvs"], bins=np.logspace(1,3,500), histtype='step', log=True, color='k', label='all')
	print("calling without specifying key_or_keys or apply_mask=True will do nothing, and warn.")
	bm.mask(data.cut("vMax_3046_1",90))
	print("call with apply_mask=True to make the change")
	bm.mask(data.cut("vMax_3046_1",90),apply_mask=True)
	plt.hist(bm["area_3046_1_nvs"], bins=np.logspace(1,3,500), histtype='step', log=True, color='g', label='vMax_3046_1>90')
	plt.xlabel("area_3046_1 (nVs)")
	plt.xscale("log")
	plt.legend()
	plt.show()
	print("")
