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




# specify dataset

# list of sources to perform analysis on
SOURCES = [
	# "Cs137", # no sim file
	# "Mn54" , # DRS saturation
	# "Na22" , # DRS saturation
	"Am241",
	"Cd109",
	"Ba133",
	"Co57" ,
]

# ranges for splitting sim datasets based on primary vertex energy
# 
# The keys are values in KeV read off plots of simulation data/
# They are not values from some other reference.
# The keys will change if/when the simulation parameters are updated
# to better match reference values and precision.
# 
# These are also used to mark vertical lines on plots
PVE_KEV = {
	"Am241": { "14.1":[ 13, 15],  "18.1":[ 17, 19],  "59.5":[ 59, 60], },
	"Cd109": { "22.1":[ 21, 23],  "25.1":[ 24, 26],  "88.0":[ 86, 90], },
	"Ba133": { "31.1":[ 30, 32],  "81.0":[ 80, 82], "303.2":[302,304], "356.0":[355,357], },
	"Co57" : { "14.1":[ 13, 15], "122.1":[121,123], "136.1":[135,137], },
	"Mn54" : {"835.0":[834,836], },
	"Na22" : {"511.0":[510,512], },
}
del PVE_KEV["Am241"]["18.1"] # 
del PVE_KEV["Am241"]["14.1"] # these have no associated energy deposits in scintillators
del PVE_KEV["Co57" ]["14.1"] # they would cause issues for fitting if included.

del PVE_KEV["Co57" ]["136.1"] # temporarily disable while area range
del PVE_KEV["Ba133"]["303.2"] # is reduced to exclude PMT saturation

# run numbers for experimental source spectra
RUNS_EXP_SRC = {
	# "Cs137":0,
	# "Mn54" :0,
	# "Na22" :0,
	"Am241":4291,
	"Ba133":4293,
	"Cd109":4292,
	"Co57" :4294,
}

# run numbers for background spectra
RUNS_EXP_BG = [4225, 4226]

# data directories and files
DIR_SIM_SRC = "/home/bode/Documents/GitHub/xrd-analysis-refactor/data/sim"
# DIR_EXP_SRC = "/home/bode/Documents/GitHub/xrd-analysis/data/root/scintillator"
# DIR_EXP_BG  = "/home/bode/Documents/GitHub/xrd-analysis/data/root/scintillator"
DIR_EXP_SRC = "/home/bode/Documents/GitHub/xrd-analysis-refactor/data/exp"
DIR_EXP_BG  = "/home/bode/Documents/GitHub/xrd-analysis-refactor/data/exp"
FILENAME_SIM_SRC = "{src}.npz"
# FILENAME_EXP_SRC = "Run{run}.root"
# FILENAME_EXP_BG  = "Run{run}.root"
FILENAME_EXP_SRC = "Run{run}.npz"
FILENAME_EXP_BG  = "Run{run}.npz"
FILE_SIM_SRC = os.sep.join([DIR_SIM_SRC, FILENAME_SIM_SRC])
FILE_EXP_SRC = os.sep.join([DIR_EXP_SRC, FILENAME_EXP_SRC])
FILE_EXP_BG  = os.sep.join([DIR_EXP_BG , FILENAME_EXP_BG ])
FILES_SIM_SRC = {_:FILE_SIM_SRC.format(src=_)               for _ in SOURCES}
FILES_EXP_SRC = {_:FILE_EXP_SRC.format(run=RUNS_EXP_SRC[_]) for _ in SOURCES}
FILES_EXP_BG  = [FILE_EXP_SRC.format(run=_) for _ in RUNS_EXP_BG]

# channel number identification
CH_COMP = [1,2,3]
CH_LYSO = 4
CH_ALL  = CH_COMP + [CH_LYSO]

# names and lists of leaves
LEAF_SIM_PVE_MEV  = "GammaTracks.initialEnergy_MeV"
LEAF_SIM_EDEP_MEV = "Edep_MeV_Si{ch}"
LEAF_SIM_EDEP_KEV = "Edep_KeV_Si{ch}"
# LEAF_SIM_AREA_PVS = "area_pvs_{ch}"
LEAF_EXP_AREA_PVS = "area_{drs}_{ch}"
LEAF_EXP_AREA_NVS = "area_nVs_{drs}_{ch}"
DRS_ID = "3046"

LEAF_LYSO_AREA_PVS = LEAF_EXP_AREA_PVS.format(ch=CH_LYSO, drs=DRS_ID)
LYSO_AREA_PVS_CUT_HI = 14000.0

# sets of leaves to load for each dataset
# the rest of the leaves defined above are constructed, rather than loaded from the files.
LEAVES_LOAD_SIM_SRC = {LEAF_SIM_PVE_MEV} | {LEAF_SIM_EDEP_MEV.format(ch=_) for _ in CH_ALL}
LEAVES_LOAD_EXP_SRC = {LEAF_EXP_AREA_PVS.format(drs=DRS_ID, ch=_) for _ in CH_ALL}
LEAVES_LOAD_EXP_BG  = {LEAF_EXP_AREA_PVS.format(drs=DRS_ID, ch=_) for _ in CH_ALL}



# specify data ranges and binning

# Area range per source and channel for making
# comparisons between model and observation.

# # extended - includes PMT saturation region
# AREA_RANGE_NVS = {
# 	"Am241":{1:[11, 90], 2:[11, 90], 3:[11, 90]},
# 	"Ba133":{1:[11,600], 2:[11,600], 3:[11,600]},
# 	"Cd109":{1:[11,130], 2:[11,130], 3:[11,130]},
# 	"Co57" :{1:[11,200], 2:[11,200], 3:[11,200]},
# 	# "Mn54" :[],
# 	# "Na22" :[],
# }

# # semi restrictive - excludes PMT saturation but includes Am241 low peak
# AREA_RANGE_NVS = {
# 	"Am241":{1:[11, 80], 2:[11, 80], 3:[11, 80]},
# 	"Ba133":{1:[11, 80], 2:[11, 80], 3:[11, 80]},
# 	"Cd109":{1:[11, 42], 2:[11, 42], 3:[11, 42]},
# 	"Co57" :{1:[11, 80], 2:[11, 80], 3:[11, 80]},
# }

# restrictive - excludes PMT saturation and AM241 low peak
AREA_RANGE_NVS = {
	"Am241":{1:[40, 80], 2:[40, 80], 3:[40, 80]},
	"Ba133":{1:[11, 80], 2:[11, 80], 3:[11, 80]},
	"Cd109":{1:[11, 42], 2:[11, 42], 3:[11, 42]},
	"Co57" :{1:[11, 80], 2:[11, 80], 3:[11, 80]},
}

# resolution for projections (unitless; number of bins)
RES_A = 200
RES_E = 100

# Source peak energies for plotting vlines.
# These differ from sim component values, so they're tracked separately.
SOURCE_PEAK_ENERGY_KEV = {
	"Am241": [ 26.34,  59.54],
	"Ba133": [ 30.85,  81.00],
	"Cd109": [ 22.1 ,  88.04],
	"Co57" : [122.06, 136.47],
	"Mn54" : [834.85        ],
	"Na22" : [511.0         ],
}


# parameter guesses, based on some initial fit results

# # normalized counts
# PARAM_GUESS = {
# 	1:{
# 		"gamma"         :     1.107039,
# 		"res_s_a"       :     0.058332,
# 		"rho_p"         :     0.002376,
# 		"n_bg_Am241"    :  4764.428653,
# 		"n_18.1_Am241"  :   576.523384,
# 		"n_59.5_Am241"  : 15946.410121,
# 		"n_bg_Cd109"    :  7783.941640,
# 		"n_22.1_Cd109"  : 37328.885689,
# 		"n_25.1_Cd109"  : 14931.307143,
# 		"n_88.0_Cd109"  :  1239.068423,
# 		"n_bg_Ba133"    :  7274.405525,
# 		"n_303.2_Ba133" :     0.000000,
# 		"n_31.1_Ba133"  : 27810.109143,
# 		"n_356.0_Ba133" :  4080.333789,
# 		"n_81.0_Ba133"  :  4467.673179,
# 		"n_bg_Co57"     :  9912.613250,
# 		"n_122.1_Co57"  :   133.634939,
# 		"n_136.1_Co57"  :     0.000000,
# 	},
# 	2:{
# 		"gamma"         :    1.134562,
# 		"res_s_a"       :    0.070939,
# 		"rho_p"         :    0.002941,
# 		"n_bg_Am241"    :  713.036311,
# 		"n_18.1_Am241"  :   86.507064,
# 		"n_59.5_Am241"  : 4261.792237,
# 		"n_bg_Cd109"    : 1306.638188,
# 		"n_22.1_Cd109"  : 7578.916941,
# 		"n_25.1_Cd109"  : 4116.593253,
# 		"n_88.0_Cd109"  :  921.730587,
# 		"n_bg_Ba133"    :  890.712460,
# 		"n_303.2_Ba133" :    0.000000,
# 		"n_31.1_Ba133"  : 7989.801404,
# 		"n_356.0_Ba133" : 1527.871611,
# 		"n_81.0_Ba133"  : 1932.030914,
# 		"n_bg_Co57"     : 1874.410760,
# 		"n_122.1_Co57"  :  695.526326,
# 		"n_136.1_Co57"  :    0.000000,
# 	},
# 	3:{
# 		"gamma"         :     1.195195,
# 		"res_s_a"       :     0.076302,
# 		"rho_p"         :     0.003426,
# 		"n_bg_Am241"    :  1060.858535,
# 		"n_18.1_Am241"  :   197.360793,
# 		"n_59.5_Am241"  :  4655.759070,
# 		"n_bg_Cd109"    :  1286.354834,
# 		"n_22.1_Cd109"  : 10479.065108,
# 		"n_25.1_Cd109"  :  4825.366616,
# 		"n_88.0_Cd109"  :   809.886161,
# 		"n_bg_Ba133"    :   256.857389,
# 		"n_303.2_Ba133" :   371.944536,
# 		"n_31.1_Ba133"  :  9639.394966,
# 		"n_356.0_Ba133" :   909.448008,
# 		"n_81.0_Ba133"  :  2210.247697,
# 		"n_bg_Co57"     :  1650.147094,
# 		"n_122.1_Co57"  :   627.245179,
# 		"n_136.1_Co57"  :     0.000000,
# 	},
# }

# # non-normalized counts
# PARAM_GUESS = {
# 	1:{
# 		"gamma"         :  1.106098,
# 		"res_s_a"       :  0.054130,
# 		"rho_p"         :  0.002592,
# 		"n_bg_Am241"    :  0.387854,
# 		"n_59.5_Am241"  :  1.525951,
# 		"n_bg_Cd109"    :  0.575178,
# 		"n_22.1_Cd109"  : 13.247709,
# 		"n_25.1_Cd109"  : 12.849556,
# 		"n_88.0_Cd109"  :  8.946416,
# 		"n_bg_Ba133"    :  0.534179,
# 		"n_31.1_Ba133"  :  4.487634,
# 		"n_356.0_Ba133" : 58.393168,
# 		"n_81.0_Ba133"  :  7.707150,
# 		"n_bg_Co57"     :  0.704529,
# 		"n_122.1_Co57"  :  0.000000,
# 	},
# 	2:{
# 		"gamma"         :  1.133792,
# 		"res_s_a"       :  0.058869,
# 		"rho_p"         :  0.003742,
# 		"n_bg_Am241"    :  0.331280,
# 		"n_59.5_Am241"  :  0.350690,
# 		"n_bg_Cd109"    :  0.448344,
# 		"n_22.1_Cd109"  :  2.537088,
# 		"n_25.1_Cd109"  :  3.395640,
# 		"n_88.0_Cd109"  :  8.313754,
# 		"n_bg_Ba133"    :  0.291278,
# 		"n_31.1_Ba133"  :  1.176115,
# 		"n_356.0_Ba133" : 20.391918,
# 		"n_81.0_Ba133"  :  3.216915,
# 		"n_bg_Co57"     :  0.632431,
# 		"n_122.1_Co57"  :  1.450615,
# 	},
# 	3:{
# 		"gamma"         :  1.193533,
# 		"res_s_a"       :  0.065706,
# 		"rho_p"         :  0.004089,
# 		"n_bg_Am241"    :  0.504532,
# 		"n_59.5_Am241"  :  0.170431,
# 		"n_bg_Cd109"    :  0.505059,
# 		"n_22.1_Cd109"  :  1.414753,
# 		"n_25.1_Cd109"  :  1.557634,
# 		"n_88.0_Cd109"  :  4.156422,
# 		"n_bg_Ba133"    :  0.315509,
# 		"n_31.1_Ba133"  :  0.567587,
# 		"n_356.0_Ba133" :  6.923895,
# 		"n_81.0_Ba133"  :  1.826853,
# 		"n_bg_Co57"     :  0.606412,
# 		"n_122.1_Co57"  :  0.719298,
# 	},
# }

# increased area res, better(?) stuff
PARAM_GUESS = {
	1:{
		"gamma"         : 1.097432, 
		"res_s_a"       : 0.066664, 
		"rho_p"         : 0.001989, 
		"n_bg_Am241"    : 0.277073, 
		"n_59.5_Am241"  : 0.006120, 
		"n_bg_Cd109"    : 0.553708, 
		"n_22.1_Cd109"  : 0.307077, 
		"n_25.1_Cd109"  : 0.212869, 
		"n_88.0_Cd109"  : 0.000000, 
		"n_bg_Ba133"    : 0.193090, 
		"n_31.1_Ba133"  : 0.110131, 
		"n_356.0_Ba133" : 0.248205, 
		"n_81.0_Ba133"  : 0.017168, 
		"n_bg_Co57"     : 0.399963, 
		"n_122.1_Co57"  : 0.063805, 
	},
	2:{
		"gamma"         : 1.121880, 
		"res_s_a"       : 0.056019, 
		"rho_p"         : 0.003989, 
		"n_bg_Am241"    : 0.269309, 
		"n_59.5_Am241"  : 0.001301, 
		"n_bg_Cd109"    : 0.295428, 
		"n_22.1_Cd109"  : 0.056627, 
		"n_25.1_Cd109"  : 0.053310, 
		"n_88.0_Cd109"  : 0.071958, 
		"n_bg_Ba133"    : 0.125694, 
		"n_31.1_Ba133"  : 0.026863, 
		"n_356.0_Ba133" : 0.078880, 
		"n_81.0_Ba133"  : 0.003011, 
		"n_bg_Co57"     : 0.228454, 
		"n_122.1_Co57"  : 0.025074, 
	},
	3:{
		"gamma"         : 1.179257, 
		"res_s_a"       : 0.070185, 
		"rho_p"         : 0.003854, 
		"n_bg_Am241"    : 0.363776, 
		"n_59.5_Am241"  : 0.000616, 
		"n_bg_Cd109"    : 0.526177, 
		"n_22.1_Cd109"  : 0.028869, 
		"n_25.1_Cd109"  : 0.024996, 
		"n_88.0_Cd109"  : 0.003722, 
		"n_bg_Ba133"    : 0.171747, 
		"n_31.1_Ba133"  : 0.012543, 
		"n_356.0_Ba133" : 0.030650, 
		"n_81.0_Ba133"  : 0.001752, 
		"n_bg_Co57"     : 0.311807, 
		"n_122.1_Co57"  : 0.009490, 
	}
}




# convenience and utility functions

def pprint(stuff, fmt=False, join=False, pre=False, sep=''):
	lines = []
	for _ in stuff:
		if join:
			line = join.join(map(str, [fmt.format(__) if fmt else __ for __ in _]))
		else:
			line = fmt.format(_) if fmt else _
		if pre:
			print(pre, line)
		else:
			print(line)

def dict_reduce(method, *dicts):
	"""make new dict by performing reduce per key,value on list of dicts"""

	# no action if one dict; error if zero.
	if len(dicts) < 2:
		return dicts[0]

	# assume all dicts have identical keys
	new_dict = {}
	for key,value in dicts[0].items():
		new_value = value
		for dict_ in dicts[1:]:
			new_value = method(value, dict_[key])
		new_dict[key] = new_value

	return new_dict

def normed(arr, axis=None):
	return arr / np.sum(arr, axis)

def normed_if(arr, axis=None, do=False):
	return normed(arr,axis) if do else arr

def summary(arr, ):
	return ", ".join(map(str,[arr[0], arr[1], arr[2], "...", arr[-1]]))

def fn_indent(fn):
	def wrapped(inst, *args, **kwargs):
		# f_inc()
		inst.vp(1, '\n')
		inst.vp(1, fn.__name__)
		inst.inc_indent()
		ans = fn(inst, *args, **kwargs)
		inst.dec_indent()
		return ans
	return wrapped




class routine(object):
	
	indent = 0
	def inc_indent(self):
		self.indent += 1
		# print(self.indent)
	def dec_indent(self):
		self.indent = max([0, self.indent - 1])
		# print(self.indent)

	vb = 2
	def vp(self,lv=1,*args,**kwargs):
		kwargs['sep']=kwargs['sep'] if 'sep' in kwargs else ''
		if self.vb >= lv:
			print('\t'*self.indent, *args, **kwargs)
	def vpp(self,lv=1,*args,**kwargs):
		kwargs['sep']=kwargs['sep'] if 'sep' in kwargs else ''
		if self.vb >= lv:
			pprint(*args,**kwargs)


	def __init__(self):
		self.load_data()
		self.prepare_data()
		self.setup_model()


	@fn_indent
	def load_data(self):
		"""load arrays from specified files and initialize branch managers"""

		# load simulated source data
		# {src:{"E":manager for component E}}
		self.sim_src = {}
		eks = sorted(LEAVES_LOAD_SIM_SRC)[0]
		self.vp(1, "")
		self.vp(1, "loading simulation source data")
		self.vp(2, "leaves: {}".format(LEAVES_LOAD_SIM_SRC))
		for src,file in FILES_SIM_SRC.items():
			self.vp(2, "source {}, file {}".format(src,file))
			arrays = np.load(file)
			self.vp(2, "leaf {} has shape {}".format(eks, arrays[eks].shape))
			self.vpp(3, [(k,list(v[:10])) for k,v in arrays.items()], join=" = ")
			arrays = {k:v for k,v in arrays.items() if k in LEAVES_LOAD_SIM_SRC}

			# separate simulated source spectra by primary vertex energy
			# "E":manager where "E" is primary vertex energy
			self.sim_src[src] = {}
			for label_e_kev, (lo_kev,hi_kev) in PVE_KEV[src].items():
				self.vp(2, "finding events for Epv={} KeV, accepting ({},{})".format(
					label_e_kev, lo_kev, hi_kev))

				# find events within the specified energy range
				this_ftr = np.logical_and(
					arrays[LEAF_SIM_PVE_MEV] > (lo_kev / 1000),
					arrays[LEAF_SIM_PVE_MEV] < (hi_kev / 1000))
				this_n_pass = this_ftr.sum()
				self.vp(2, "{} / {} pass".format(this_n_pass, this_ftr.size))

				# todo: check the there are energy deposits
				#       for now, I've just disabled the 14.1KeV ones,
				#       which are the only ones with no deposits.

				# setup branch manager for this component
				if this_n_pass:
					self.sim_src[src][label_e_kev] = data.BranchManager(
						{k:v[this_ftr] for k,v in arrays.items()},
						import_copies=False, export_copies=False)
				else:
					self.vp("discarding this component since it has no events")

				del this_ftr
			del arrays


		# load experimental source data
		# {src:manager}
		self.exp_src = {}
		eke = sorted(LEAVES_LOAD_EXP_SRC)[0]
		self.vp(1, "")
		self.vp(1, "loading experimental source data")
		self.vp(2, "leaves: {}".format(LEAVES_LOAD_EXP_SRC))
		for src,file in FILES_EXP_SRC.items():
			self.vp(2, "source {}, file {}".format(src,file))
			arrays = np.load(file)
			self.vp(2, "leaf {} has shape {}".format(eke, arrays[eke].shape))
			self.vpp(3, [(k,list(v[:10])) for k,v in arrays.items()], join=" = ")
			self.exp_src[src] = data.BranchManager(
				{k:v for k,v in arrays.items() if k in LEAVES_LOAD_EXP_SRC},
				export_copies=False, import_copies=False)
			del arrays


		# load experimental background data
		exp_bg_pieces = []
		self.vp(1, "")
		self.vp(1, "loading experimental background data")
		self.vp(2, "leaves: {}".format(LEAVES_LOAD_EXP_BG))
		for file in FILES_EXP_BG:
			self.vp(2, "file {}".format(file))
			arrays = np.load(file)
			self.vp(2, "leaf {} has shape {}".format(eke, arrays[eke].shape))
			self.vpp(3, [(k,list(v[:10])) for k,v in arrays.items()], join=" = ")
			exp_bg_pieces.append({k:v for k,v in arrays.items() if k in LEAVES_LOAD_EXP_SRC})
			del arrays
		self.vp(1, "concatenating background datasets")
		exp_bg = dict_reduce(lambda a,b: np.concatenate((a,b), axis=0), *exp_bg_pieces)
		del exp_bg_pieces
		self.vp(2, "leaf {} has shape {}".format(eke, exp_bg[eke].shape))
		self.exp_bg = data.BranchManager(exp_bg, export_copies=False, import_copies=False)
		del exp_bg


	@fn_indent
	def prepare_data(self):
		"""process loaded data, creating forms needed by analysis routines"""

		# process experimental source data
		# buds for converting pVs area measurements to nVs
		buds_pvs_to_nvs = [
			data.bud_function(
				LEAF_EXP_AREA_PVS.format(ch=_,drs=DRS_ID),
				LEAF_EXP_AREA_NVS.format(ch=_,drs=DRS_ID),
				lambda a:a/1000
			) for _ in CH_COMP
		]
		# mask for LYSO activity
		mask_lyso_cut = data.cut(LEAF_LYSO_AREA_PVS,hi=LYSO_AREA_PVS_CUT_HI)
		# apply masks and buds
		self.vp(1, "")
		self.vp(1, "preparing experimental source data")
		self.vp(2, "cutting on LYSO area (pVs) < {} and converting pVs -> nVs".format(LYSO_AREA_PVS_CUT_HI))
		for src,manager in self.exp_src.items():
			self.vp(3, src)
			manager.mask(mask_lyso_cut, apply_mask=True)
			manager.bud(buds_pvs_to_nvs)
			self.vp(3, "before : {}".format(list(manager[LEAF_EXP_AREA_PVS.format(ch=CH_COMP[0],drs=DRS_ID)][:10])))
			self.vp(3, "after  : {}".format(list(manager[LEAF_EXP_AREA_NVS.format(ch=CH_COMP[0],drs=DRS_ID)][:10])))


		# process experimental background data
		self.vp(1, "")
		self.vp(1, "preparing experimental background data")
		self.vp(2, "cutting on LYSO area (pVs) < {} and converting pVs -> nVs".format(LYSO_AREA_PVS_CUT_HI))
		self.exp_bg.mask(mask_lyso_cut, apply_mask=True)
		self.exp_bg.bud(buds_pvs_to_nvs)
		self.vp(3, "before : {}".format(list(self.exp_bg[LEAF_EXP_AREA_PVS.format(ch=CH_COMP[0],drs=DRS_ID)][:10])))
		self.vp(3, "after  : {}".format(list(self.exp_bg[LEAF_EXP_AREA_NVS.format(ch=CH_COMP[0],drs=DRS_ID)][:10])))


		# process simulated source data
		# buds for converting MeV energy deposits to KeV
		buds_mev_to_kev = [
			data.bud_function(
				LEAF_SIM_EDEP_MEV.format(ch=_),
				LEAF_SIM_EDEP_KEV.format(ch=_),
				lambda e:e*1000
			) for _ in CH_COMP
		]
		# apply buds
		self.vp(1, "")
		self.vp(1, "preparing simulated source data")
		self.vp(2, "converting MeV -> KeV")
		for src,managers in self.sim_src.items():
			for e_kev,manager in managers.items():
				self.vp(3, "{}, {}".format(src, e_kev))
				kev = manager.bud(buds_mev_to_kev)
				self.vp(3, "before : {}".format(list(manager[LEAF_SIM_EDEP_MEV.format(ch=CH_COMP[0])][:20])))
				self.vp(3, "after  : {}".format(list(manager[LEAF_SIM_EDEP_KEV.format(ch=CH_COMP[0])][:20])))


		# calculate bins and bin data in to spectra
		self.vp(1, "")
		self.vp(1, "binning datasets")

		# {channel:{source:{component:edges}}}
		self.edges_e_kev = {}
		self.mids_e_kev  = {}
		
		# {channel:{source:edges}}
		self.edges_a_nvs = {}
		self.mids_a_nvs  = {}
		
		# {channel:{source:{component:spectrum}}}
		self.spec_sim_src_kev     = {}
		self.spec_sim_src_kev_err = {}

		# {channel:{source:spectrum}}
		self.spec_exp_src_nvs     = {}
		self.spec_exp_src_nvs_err = {}

		# {channel:{source:spectrum}}
		self.spec_exp_bg_nvs      = {}
		self.spec_exp_bg_nvs_err  = {}
		
		for ch in CH_COMP:
			self.vp(2, "channel {}".format(ch))

			self.edges_a_nvs[ch]      = {}
			self.edges_e_kev[ch]      = {}
			self.mids_a_nvs[ch]       = {}
			self.mids_e_kev[ch]       = {}
			self.spec_sim_src_kev[ch] = {}
			self.spec_exp_src_nvs[ch] = {}
			self.spec_exp_bg_nvs[ch]  = {}
			self.spec_sim_src_kev_err[ch] = {}
			self.spec_exp_src_nvs_err[ch] = {}
			self.spec_exp_bg_nvs_err[ch]  = {}

			for src in SOURCES:
				self.vp(2, "")
				self.vp(2, "source {}".format(src))

				# populate edges_a_nvs
				this_data = self.exp_src[src][LEAF_EXP_AREA_NVS.format(ch=ch,drs=DRS_ID)]
				this_data_inrange = this_data[np.logical_and(this_data>AREA_RANGE_NVS[src][ch][0], this_data<AREA_RANGE_NVS[src][ch][1])]
				this_ndata = this_data_inrange.size

				this_nbins = min([math.floor(this_ndata / 20), RES_A])				
				this_edges = data.edges_lin(*AREA_RANGE_NVS[src][ch], RES_A)
				# this_edges = data.edges_equal_count(RES_A, this_data, *AREA_RANGE_NVS[src][ch])
				
				self.edges_a_nvs[ch][src] = this_edges
				self.mids_a_nvs[ch][src] = 0.5*(this_edges[1:]+this_edges[:-1])
				self.vp(2, "area (nVs), {} data, {} edges {}".format(this_ndata, this_nbins, summary(this_edges)))

				# populate spec_exp_src_nvs
				this_hist = np.histogram(this_data, this_edges)[0]
				self.spec_exp_src_nvs[ch][src] = this_hist
				self.spec_exp_src_nvs_err[ch][src] = np.sqrt(this_hist)
				del this_data

				# populate spec_exp_bg_nvs
				this_data = self.exp_bg[LEAF_EXP_AREA_NVS.format(ch=ch,drs=DRS_ID)]
				this_hist = np.histogram(this_data, this_edges)[0]
				# this_sum  = this_hist.sum()
				this_sum  = 1 # don't normalize counts
				self.spec_exp_bg_nvs[ch][src] = this_hist / this_sum
				self.spec_exp_bg_nvs_err[ch][src] = np.sqrt(this_hist) / this_sum
				del this_data, this_edges, this_hist, this_sum

				# populate edges_e_kev and spec_sim_src_kev
				self.edges_e_kev[ch][src] = {}
				self.mids_e_kev[ch][src] = {}
				self.spec_sim_src_kev[ch][src] = {}
				self.spec_sim_src_kev_err[ch][src] = {}
				self.vp(2, "components of sim spectrum")
				for e_kev,manager in self.sim_src[src].items():
					this_data = manager[LEAF_SIM_EDEP_KEV.format(ch=ch,drs=DRS_ID)]
					this_data_nonzero = this_data[this_data>0]

					this_nbins = min([math.floor(this_data_nonzero.size / 20), RES_E])
					this_edges = data.edges_lin(this_data[this_data>0].min(), this_data.max(), RES_E)
					# this_edges = data.edges_equal_count(this_nbins, this_data, this_data_nonzero.min(), this_data.max())

					self.edges_e_kev[ch][src][e_kev] = this_edges
					self.mids_e_kev[ch][src][e_kev] = 0.5*(this_edges[1:]+this_edges[:-1])
					this_hist = np.histogram(this_data,this_edges)[0]
					# this_sum = np.sum(this_hist)
					this_sum = 1 # don't normalize counts
					self.spec_sim_src_kev[ch][src][e_kev] = this_hist / this_sum
					self.spec_sim_src_kev_err[ch][src][e_kev] = np.sqrt(this_hist) / this_sum
					self.vp(2, "{} KeV, {} data, {} edges {}".format(e_kev, this_data_nonzero.size, this_nbins, summary(this_edges)))
					# self.vp(2, "energy (KeV) edges {}".format())
					del this_data, this_edges


		# set up concatenated experimental spectra,
		# and objects, methods, and information for convenience,
		# packing and unpacking
		# 
		# Order alphabetically by source so that consistency can
		# be achieved withouth storing the order used.
		# 
		# {channel:concatenated_source_spectra}
		self.spec_exp_src_nvs_flat     = {}
		self.spec_exp_src_nvs_err_flat = {}
		src_sorted = sorted(SOURCES)
		self.vp(1, "")
		self.vp(1, "concatenating experimental source spectra")
		self.vp(1, "source order is {}".format(src_sorted))
		for ch in CH_COMP:
			this_pieces = [self.spec_exp_src_nvs[ch][_] for _ in src_sorted]
			self.spec_exp_src_nvs_flat[ch] = np.concatenate(this_pieces, axis=0)
			self.spec_exp_src_nvs_err_flat[ch] = np.sqrt(self.spec_exp_src_nvs_flat[ch])
			self.vp(2, "channel {}, size {}".format(ch, self.spec_exp_src_nvs_flat[ch].size))


	def unflatten_sources(self, data_flat, ch=None):
		if ch is None:
			ch = CH_COMP[0]
		pieces = {}
		istart = 0
		for src in sorted(SOURCES):
			this_size = self.spec_exp_src_nvs[ch][src].shape[0]
			pieces[src] = data_flat[istart:istart+this_size]
			istart += this_size
		return pieces


	def show_spectra(self):
		# one channel at a time
		for ch in CH_COMP:
			nsrc = len(SOURCES)
			nr = 2
			nc = nsrc # math.ceil(nsrc/2)
			fig,ax = plt.subplots(nr,nc,sharex=False,sharey=False)
			fig.subplots_adjust(
			    top=0.90,
			    bottom=0.10,
			    left=0.06,
			    right=0.94,
			    hspace=0.2,
			    wspace=0.2,
			)
			plt.suptitle("observed area and simulated energy spectra, channel {}".format(ch))
			fig.set_size_inches(20,10)
			fig.set_dpi(100)

			for i,src in enumerate(SOURCES):

				plt.subplot(nr,nc,i+1)
				# plt.fill_between(self.mids_a_nvs[ch][src], self.spec_exp_src_nvs[ch][src], step='mid', alpha=0.5)#facecolor=(1,0.4,0.4,0.4), edgecolor=(1,0.4,0.4,1))
				mids   = self.mids_a_nvs[ch][src]
				edges  = self.edges_a_nvs[ch][src]
				widths = edges[1:] - edges[:-1]
				plt.fill_between(mids, self.spec_exp_src_nvs[ch][src] / widths, color='k', step='mid', alpha=0.1)#facecolor=(1,0.4,0.4,0.4), edgecolor=(1,0.4,0.4,1))
				plt.step(mids, self.spec_exp_src_nvs[ch][src] / widths, color='k', where='mid')
				
				plt.fill_between(mids, self.spec_exp_bg_nvs[ch][src] / widths, color='g', step='mid', alpha=0.2)#facecolor=(1,0.4,0.4,0.4), edgecolor=(1,0.4,0.4,1))
				plt.step(mids, self.spec_exp_bg_nvs_err[ch][src] / widths, color='g', where='mid')
				
				plt.xlabel('Area (nVs)')
				plt.title("observed area spectrum, {}".format(src))



				plt.subplot(nr,nc,i+1+nsrc)
				for e,spec in self.spec_sim_src_kev[ch][src].items():
					mids   = self.mids_e_kev[ch][src][e]
					edges  = self.edges_e_kev[ch][src][e]
					widths = edges[1:] - edges[:-1]
					plt.fill_between(mids, spec/widths, step='mid', alpha=0.3)
					plt.step(mids, spec/widths, where='mid', label="{} KeV".format(e))
				plt.xlabel('Energy (KeV)')
				plt.title("simulated energy deposits, {}".format(src))
				plt.yscale('log')
				plt.legend()
			plt.show()

	def show_evaluate(self, pm, ch, xdata=None, incl_exp=True, incl_sep=True, incl_sim_e=False):

		ev     = self.evaluate(xdata, pm, ch)
		ev_err = self.parametrizer.vector_num_error_p_only(pm, self.evaluate, xdata, [ch])

		model     = self.unflatten_sources(ev)
		model_err = self.unflatten_sources(ev_err)

		nc = len(model)
		for i,src in enumerate(sorted(model.keys())):
			this_edges = self.edges_a_nvs[ch][src]
			this_w = this_edges[1:] - this_edges[:-1]
			this_x = self.mids_a_nvs[ch][src]
			this_model     = model[src]
			this_model_err = model_err[src]
			plt.subplot(1,nc,i+1)

			if incl_exp:
				this_exp = self.spec_exp_src_nvs[ch][src]
				this_exp_err = np.sqrt(this_exp)
				plt.step(this_x, this_exp/this_w, 'k', where='mid', label='observed')
				plt.fill_between(
					x  = this_x,
					y1 = (this_exp + this_exp_err) / this_w,
					y2 = (this_exp - this_exp_err) / this_w,
					step='mid',
					alpha=0.2,
					color='k',
				)

			plt.step(this_x, this_model/this_w, color='b', where='mid', label='best fit')
			plt.fill_between(
				x  = this_x,
				y1 = (this_model+this_model_err) / this_w,
				y2 = (this_model-this_model_err) / this_w,
				step='mid',
				alpha=0.3,
				color='b',
			)

			if incl_sep:
				# plot background component
				plt.step(
					this_x,
					(self.spec_exp_bg_nvs[ch][src] * pm["n_bg_{src}".format(src=src)]) / this_w,
					color='c',
					where='mid',
					label='model bg',
				)
				# plot simulation components
				for epv, projector in self.projectors[ch][src].items():
					plt.step(
						this_x,
						(normed_if(projector(pm)) * pm["n_{epv}_{src}".format(epv=epv, src=src)]) / this_w,
						where='mid',
						label='model {} KeV'.format(epv),
					)

			if incl_sim_e:
				for k,v in PVE_KEV[src].items():
					pve = float(k)
					mu_a_pve, _ = self.E_to_mu_sigma(pve, pm)
					plt.axvline(mu_a_pve, label='{} KeV'.format(pve))

			plt.title(src)
			plt.xlabel('area (nVs)')
			plt.legend()

		if pm.chi2 is not None:
			plt.suptitle('channel {} - chi2/dof = {:.1f}/{} = {:.3f}'.format(
				ch, pm.chi2, pm.ndof, pm.rchi2))
		else:
			plt.suptitle('channel {}'.format(ch))
		plt.show()

	def show_evaluate_p0(self):
		for ch in CH_COMP:
			self.show_evaluate(self.parametrizer.get_p0(), ch)


	@fn_indent
	def setup_model(self):
		"""initialize functions and objects for performing model calculation"""


		# initialize parametrizer class so that parameters can be
		# added at time of defining each stage
		self.parametrizer = fit.parametrizer()

		# do these in a module-level function and just access it here
		#  define transformation for E -> mu,sigma
		#  define transformation for A* -> A
		#  create point spread function for E -> (A* density)

		# todo: automatically get set of desired parameters

		def E_to_res(E, pm):
			# res_s = pm.res_s_a + (pm.res_s_b * 50) / E
			# res_s = pm.res_s_a + (pm.res_s_b) * np.sqrt(50/E)
			res_s = pm.res_s_a # constant res_s
			res_p_squared = (pm.rho_p * 50) / E
			return np.sqrt(res_s**2 + res_p_squared)
		self.E_to_res = E_to_res

		def E_to_mu_sigma(E, pm):
			mu = E * pm.gamma
			sigma = mu * E_to_res(E, pm)
			return mu, sigma
		self.E_to_mu_sigma = E_to_mu_sigma

		# parameters for E -> mu, sigma
		self.parametrizer.add_parameters({
			"gamma"  : 1.195, 
			"rho_p"  : 0.003,
			"res_s_a": 0.076,
			# "res_s_b": 0.001,
		})

		# # parameters for A <-> A*
		# self.parametrizer.add_parameters({
		# 	"sat_xf_a": 1.0,
		# 	"sat_xf_b": 0.0,
		# })

		# trivial transformation for testing
		def xf_and_dxf(A, pm):
			return A, 1
		self.xf_and_dxf = xf_and_dxf

		# self.parametrizer.add_parameters({
		# 	"":	
		# })

		# make binned projectors for each piece of each source spectrum
		# using the point spread function as above
		# 
		# note: normalize the spectrum components to each have unit amplitude
		#       before giving them to the projectors, so that the parameter
		#       for component magnitude is a true "number of events" measure.
		# 
		# {ch:{src:{Epv:projector}}}
		self.projectors = {}
		for ch in CH_COMP:
			self.projectors[ch] = {}
			for src in SOURCES:
				self.projectors[ch][src] = {}
				for e in PVE_KEV[src].keys():

					# todo: calculate 2d bins once and give references
					#       if memory use becomes too large
					# test_emus = lambda E,pm:(E*pm.gamma, E*pm.gamma*pm.res)
					this_e_mids  = self.mids_e_kev[ch][src][e]
					this_e_edges = self.edges_e_kev[ch][src][e]
					this_e_width = this_e_edges[1:]-this_e_edges[:-1]

					this_a_mids  = self.mids_a_nvs[ch][src]
					this_a_edges = self.edges_a_nvs[ch][src]
					this_a_width = this_a_edges[1:] + this_a_edges[:-1]

					# psf_EA = fit.gaus_spread(E_to_mu_sigma)
					self.psf_EA = fit.transformed_gaus_spread(E_to_mu_sigma, xf_and_dxf)

					this_projector = fit.binned_projector(
						func = self.psf_EA,
						xMids  = this_e_mids,
						xWidth = this_e_width,
						yMids  = this_a_mids,
						yWidth = this_a_width,
						xSpec  = self.spec_sim_src_kev[ch][src][e],
					)
					self.projectors[ch][src][e] = this_projector


		# parameters for background contribution
		for src in SOURCES:
			self.parametrizer.add_parameters({"n_bg_{}".format(src):4000.0})
			self.parametrizer.add_parameters({"n_{}_{}".format(_,src):10000.0 for _ in PVE_KEV[src].keys()})

	def get_xdata_flat(self, ch):
		pieces = []
		for src in sorted(SOURCES):
			pieces.append(self.spec_exp_bg_nvs[ch][src])
			for e in PVE_KEV[src].keys():
				pieces.append(self.spec_sim_src_kev[ch][src][e])
		return np.concatenate(pieces, axis=0)

	def get_xdata_err_flat(self, ch):
		pieces = []
		for src in sorted(SOURCES):
			pieces.append(self.spec_exp_bg_nvs_err[ch][src])
			for e in PVE_KEV[src].keys():
				pieces.append(self.spec_sim_src_kev_err[ch][src][e])
		return np.concatenate(pieces, axis=0)

	def evaluate(self, xdata, pm, ch):
		""""""

		# xdata is not needed since all the data used in modeling the spectra
		# is contained within the class instance. However, it's necessary to
		# accept it as a parameter, since curve_fit expects it to be there. We
		# can just pass None to the fitter and ignore the parameter here.


		# Unfortunately, curve_fit is the only acceptable option here.
		# We need to get parameter covariance, and we also need to specify
		# sigma and set absolute_sigma = True.
		# Neither leastsq nor least_squares behaves properly with regards
		# to sigma, and minimize cannot calculate covariance.
		# 
		# So, to make curve_fit work properly, we need to concatenate all the
		# observed and modeled source spectra.
		# 
		# The concatenation of the experimental sepctra is supplied to the minimizer;
		# it does not need to be known here.
		# 
		# This function needs to return the concatenation of all of the modeled spectra.
		# Crucially, these must be in the same order as the experimetnal spectra.
		# To achieve this without memory, the order will be alphabetical by source.


		# list of spectra, not concatenated yet.
		model_src_spectra = []

		# for tracking position in xdata if supplied
		istart = 0

		# calculate spectra for each source, sorted
		for src in sorted(SOURCES):

			# start with background component
			if xdata is None:
				this_spectrum = self.spec_exp_bg_nvs[ch][src] * pm["n_bg_{src}".format(src=src)]
			else:
				this_size = self.spec_exp_bg_nvs[ch][src].size
				this_spectrum = xdata[istart:istart+this_size] * pm["n_bg_{src}".format(src=src)]
				istart += this_size

			# add each simulation component
			for epv, projector in self.projectors[ch][src].items():

				if xdata is None:
					xSpec = None
				else:
					this_size = self.spec_sim_src_kev[ch][src][epv].size
					xSpec = xdata[istart:istart+this_size]
					# xSpec = None
					istart += this_size
				
				# Since projection to an incomplete range doesn't preserve norm,
				# we have to normalize here again if we want the meaning of the
				# number parameters to stay the same.
				# 
				# However, for the purpose of understanding the effects of perturbing
				# the spectrum data, it's necessary disable this feature.
				this_spectrum += normed_if(projector(pm, xSpec)) * pm["n_{epv}_{src}".format(epv=epv, src=src)]

			# append this source's modeled spectrum to the list
			model_src_spectra.append(this_spectrum)
			del this_spectrum

		# concatenate list of spectra and return result
		return np.concatenate(model_src_spectra, axis=0)


	def optimize(self, ch):

		pm0 = self.parametrizer.get_p0()
		xdata_flat = self.get_xdata_flat(ch)
		xerr_flat  = self.get_xdata_err_flat(ch)
		# xdata_flat = None
		# xerr_flat  = None

		# plt.plot(np.linspace(0,1,xdata_flat.size), xdata_flat+xerr_flat, 'r-')
		# plt.plot(np.linspace(0,1,xdata_flat.size), xdata_flat-xerr_flat, 'r-')
		# plt.plot(np.linspace(0,1,xdata_flat.size), xdata_flat, 'k-')
		# plt.show()
		# sys.exit(0)

		bounds = {_:[0,np.inf] for _ in pm0.v_names}
		# del bounds["res_s_b"]

		pm_opt, ym_opt, ym_err = self.parametrizer.curve_fit(
			xdata  = xdata_flat,
			xerr   = xerr_flat,
			ydata  = self.spec_exp_src_nvs_flat[ch],
			yerr   = self.spec_exp_src_nvs_err_flat[ch],
			f      = self.evaluate,
			f_args = [ch],
			bounds = bounds,
			p0     = PARAM_GUESS.get(ch, None),
		)
		for ip,p in enumerate(pm_opt.v_names):
			print("{:>2} {:>16} = {:>16.6f} \xb1 {:>16.6f}".format(ip, p, pm_opt.v_opt[ip], pm_opt.v_err[ip]))
		
		self.show_evaluate(pm_opt, ch, xdata=xdata_flat, incl_sep=True)








def main():
	rtn = routine()

	# rtn.show_spectra()
	# rtn.show_evaluate_p0()
	for ch in CH_COMP:
		rtn.optimize(ch)

if __name__ == "__main__":
	# todo: argparse for control
	main()

