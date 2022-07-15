"""
contains code for reading and writing files
and associated routines and data structures
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"


import os
import re
import csv
import numpy as np

try:
	import uproot
except:
	_has_uproot = False
else:
	_has_uproot = True



# globals

# root file key behavior, priority
# 1) supplied value
# 2) default mode result
# 3) default value
# 4) default index in all existing keys
ROOTKEY_DEFAULT_MODE = 'last'
ROOTKEY_DEFAULT_MODE_TEMPLATE = 'Events;'
ROOTKEY_DEFAULT_KEY='Events;1'
ROOTKEY_DEFAULT_INDEX = -1

# list of types that csv entries aren't allowed to be
CSV_TYPES_FORBIDDEN = [list, tuple]

# list of types recognized as a row for csv files
CSV_TYPES_ROW = [list, tuple]

# error message templates
ERR_FILE_MISSING = "file {} does not exist"
ERR_FORBIDDEN_TYPE = "forbidden type {} found as entry"

# csv typelists for established uses
TYPELIST_CALIBRATION = [str, float, int, int, float, int, float]
TYPELIST_XF = [str, int, float]

# internal utility functions

def to_str(str_or_bytes):
	"""if input type is bytes, cast to string; return"""
	if type(str_or_bytes) is str:
		return str_or_bytes
	elif type(str_or_bytes) is bytes:
		return str_or_bytes.decode('utf-8')
	else:
		raise ValueError("str_or_bytes is not type str or bytes")

gmap = lambda fn,it:list(map(fn,it))




# low level, general functions

def load_csv(file,typelist,tolerate_missing=True):
	"""loads a csv file to a 2d list of [row, row, ...]
	where row = [entry, entry, ...]
	types of entries in each row cast to corresponing member of typelist
	the last type in typelist will be applied to all further entries"""

	# validate typelist
	# lists and tuples aren't allowed, as they mess with csv reading
	for tf in CSV_TYPES_FORBIDDEN:
		if tf in typelist:
			raise ValueError(ERR_FORBIDDEN_TYPE.format(tf))
	
	ntypes = len(typelist)
	
	# case: file does not exist
	if not os.path.exists(file):
		if tolerate_missing:
			return []
		else:
			raise ValueError(ERR_FILE_MISSING.format(file))

	# case: file exists
	with open(file, 'r') as csvfile:
		contents = []
		r = csv.reader(csvfile, delimiter=' ')
		for row in r:
			contents.append([typelist[min([ie,ntypes-1])](e) for ie,e in enumerate(row)])

	return contents

def save_csv(file, contents):
	"""saves list of rows to a csv file"""

	# replace single row with list containing it
	# so that for a single row, a 2d list isn't needed
	if type(contents[0]) not in CSV_TYPES_ROW:
		contents = [contents]
	
	# write contents to the csv file (overwriting)
	with open(file, 'w') as csvfile:
		w = csv.writer(csvfile, delimiter=' ')
		for row in contents:
			w.writerow(row)

def update_csv(file, new_rows, typelist, n_match_exact, tolerate_missing=True, backup=None):
	"""updates a csv file, adding rows defined by new_rows
	overwrites old rows with new if leading entries match exactly
	number of entries that have to match defined by n_match_exact
	sorts by first n_match_exact entries per row
	n_match_exact=0 means no replacement or sorting is done
	makes a backup if backup is specifed as a file location"""

	# load previous file contents
	contents = load_csv(file, typelist, tolerate_missing)
	
	# if backup specified, and previous contents not empty, make a backup
	if backup and contents:
		save_csv(backup, contents)

	# replace single row with list containing it
	# so that for a single row, a 2d list isn't needed
	if type(new_rows[0]) not in CSV_TYPES_ROW:
		new_rows = [new_rows]

	# case: n_match_exact = 0: no overwriting or sorting
	if n_match_exact == 0:
		contents += new_rows

	# case: n_match_exact != 0: overwrite and sort
	else:

		# enumerate new_rows
		for row in new_rows:

			# start with no overwrite index
			overwrite = None
			
			# check all old rows
			for icont, cont in enumerate(contents):

				# check if first n_match_exact entries match
				if cont[:n_match_exact] == row[:n_match_exact]:
					overwrite = icont
					break
			
			# no match found -> add new row
			if overwrite is None:
				contents.append(row)
			
			# match found -> overwrite old row
			else:
				contents[overwrite] = row

			# sort by first n_match_exact entries
			contents.sort(key=lambda c:c[:n_match_exact])
	
	# save new contents
	save_csv(file, contents)

def get_keys(rootfile, rootkey=None):
	"""gets a list of all branch names (keys) for specified root file and top level rootkey"""

	if not _has_uproot:
		raise ImportError("function fileio.get_keys requires module uproot, which could not be imported.")
	
	# open root file and get list of branch names (keys)
	with uproot.open(rootfile) as root_obj:

		# default rootkey if not specified
		if rootkey is None:
			if ROOTKEY_DEFAULT_KEY in root_obj.keys():
				rootkey = ROOTKEY_DEFAULT_KEY
			else:
				rootkey = root_obj.keys()[0]

		trees = root_obj[rootkey]
		keys = trees.keys()

	# cast to strings and return
	return list(map(to_str, keys))

def branch_to_array(branch, key=False, dtype=float):
	# is 2d
	if key:
		return np.array([_[key].to_numpy() for _ in branch], dtype=dtype)
	else:
		return branch.to_numpy().astype(dtype)

def load_branches(rootfile, which=set(), rootkey=None, dtypes={}):
	"""loads branches specified in <which> from <rootfile>. loads all if <which> is empty"""

	if not _has_uproot:
		raise ImportError("function fileio.load_branches requires module uproot, which could not be imported.")

	# allows single branch to be specified as string instead of set
	if type(which) is str:
		which = {which}

	# convert to dict of key:is_2d if not already
	if type(which) in [set,list,tuple]:
		which = {key:False for key in which}

	# open root file and extract branches as arrays
	with uproot.open(rootfile) as root_obj:

		# no rootkey specified
		if rootkey is None:

			# priority 1: ROOTKEY_DEFAULT_MODE
			rootkeys_match = [_ for _ in sorted(root_obj.keys()) if _.startswith(ROOTKEY_DEFAULT_MODE_TEMPLATE)]
			if rootkeys_match:
				if ROOTKEY_DEFAULT_MODE == "last":
					rootkey = rootkeys_match[-1]
				elif ROOTKEY_DEFAULT_MODE == "first":
					rootkey = rootkeys_match[0]
				elif ROOTKEY_DEFAULT_MODE == "all":
					rootkey = rootkeys_match
				# none of the above -> assume integer as index
				else:
					rootkey = rootkeys_match[ROOTKEY_DEFAULT_MODE]

			# no keys matching template
			else:
				
				# priority 2: default value
				if ROOTKEY_DEFAULT_KEY in root_obj.keys():
					rootkey = ROOTKEY_DEFAULT_KEY

				# priority 3: index
				else:
					rootkey = sorted(root_obj.keys())[ROOTKEY_DEFAULT_INDEX]

		# convert single to list
		if not (type(rootkey) in [list,tuple,set]):
			rootkey = [rootkey]

		first_key = True
		branches = {}

		# load arrays from each rootkey and concatenate
		for rk in rootkey:

			# print(rootfile)
			# bit of a hack here
			if 'simulation' in rootfile:
				tree = root_obj[rk]["ROOTEvent"]
			else:
				tree = root_obj[rk]
			
			keys = tree.keys()

			branches_get = [_ for _ in keys if to_str(_) in which]

			for b in branches_get:
				this_array = branch_to_array(tree.arrays(b),b if which.get(b,False) else False,dtypes.get(b,float))
				# print(this_array.size)
				# print(this_array[:10])

				if first_key:
					branches[b] = this_array

				else:
					branches[b] = np.concatenate((branches[b], this_array))

			first_key = False

	return branches




# high level, for specific data types.
# format and unpacking handled here.
# for established use cases, only these
# functions should be used. If a case
# is going to be commonly used, write a
# function to handle it.

# csv cases
# peak lists / multifit models
# spectrum models
# calibrations (load, update/save, get best, )
# 

# structure for CSV file case
# 
# class for entry, holds properties in self attributes in accessible way
# has methods to pack and unpack CSV entries
# may have convenience functions for common use cases (EG, applying a model that a calibration entry represents)
# 
# functions to save, load, update CSV files of specific type



def load_calibration(calibration_file,tolerate_missing=False):
	return load_csv(calibration_file, TYPELIST_CALIBRATION, tolerate_missing)

def get_best_calibration(calib, branch, voltage, run, model_id, ard=False):

	# string: file location -> load csv
	if type(calib) is str:
		calib = load_calibration(calib)

	# start at -1, indicating no match found yet
	# if the function returns -1, then no match was found at all
	best_model_index = -1

	# difference between run and lowest run used in matching models
	lowest_run_diff = -1

	# enumerate entries
	for ie, entry in enumerate(calib):

		# start with possible match 
		is_match = True

		# check conditions for no match
		if (branch != entry[0]) and (branch is not None):
			is_match = False
		if (voltage != entry[1]) and (voltage is not None):
			is_match = False
		if model_id != entry[3]:
			is_match = False

		# calculate run difference
		this_run_diff = run - entry[2]
		if ard:
			this_run_diff = abs(this_run_diff)

		if this_run_diff < 0:
			is_match = False

		# if it's still a match
		if is_match:

			# calculate run difference
			# this_run_diff = run - entry[2]

			# first match
			if lowest_run_diff == -1:
				lowest_run_diff = this_run_diff
				best_model_index = ie

			# subsequent marches
			elif this_run_diff < lowest_run_diff:
				lowest_run_diff = this_run_diff
				best_model_index = ie

		# can't go lower, so break now
		if lowest_run_diff == 0:
			break

	# extract best model details
	if best_model_index == -1:
		best_model = None
	else:
		best_model = [
			calib[best_model_index][6],
			calib[best_model_index][7:17],
			calib[best_model_index][17:27],
		]
	return best_model_index, best_model

FIT_CSV_TYPELIST = [int, str, float, float, int, str, int, int, str, int, str]
class fit_result(object):
	"""holds properties of fit parameters and result in convenient form"""

	# todo: support covariance matrix
	# 
	# todo: add functions for modifying top-level attributes
	#       which automatically manage low-level attributes
	#       Eg: add a component -> update ngaus, names, bounds, indices, etc.
	# 
	# todo: some or all of this functionality could be absorbed by model class itself
	#       for instance, components and bounds are already contained by model class
	#       perhaps model class could have storage for results (chi2,dof,popr,perr,cov)
	#       and fileio would just have pack and unpack functions (not a class.)
	#       
	#       actually, the model module could have a function which creates a populated
	#       model_multiple class from a CSV row, and one which does the reverse,
	#       and fileio would only have to handle CSV loading and saving.
	#       that seems like the best way to do it.

	def __init__(self, contents=False):
		if contents:
			self.unpack(contents)

	def pack(self):
		"""generate list for CSV entry from self attributes"""

		...

		# # code pasted from fitBranch
		# # will need to be changed to use correct properties
		# # not needed yet, so don't bother, since this functionality
		# # will likely be absorbed by model class (see above)

		# this_contents = [

		# 	# fit data
		# 	run_id, # numerical ID of run
		# 	fit[0], # branch fit
		# 	fit[1], # branch fit lo cut
		# 	fit[2], # branch fit hi cut

		# 	# model info
		# 	model_id, # numerical ID of model
		# 	model_cal_file if model_cal_file else "-", # calibration file used
		# 	int(raw_bounds), # whether bounds are specified on untransformed data

		# 	nbins, # number of bins used (copies atuo bin count)
		# 	background if background else "-", # background function
		# 	len(gaus), # number of gaussians
		# 	*csv_format_gaus(gaus), # name,*bounds per gaussian, flattened
		# 	len(smono),
		# 	*csv_format_smono(smono),
			
		# 	# cut data
		# 	len(cuts), # number of cuts
		# 	*csv_format_cuts(cuts), # br,lo,hi per cut, flattened

		# 	# results
		# 	chi2,
		# 	ndof,
		# 	n_bg_parameters,
		# 	*popt,
		# 	*perr,

		# ]

	def unpack(self, contents):
		"""parses contents of CSV entry and sets self attripbutes to match"""

		if contents is None:
			return

		self.run        = contents[0]
		self.fit_branch = contents[1]
		self.fit_hi     = contents[2]
		self.fit_lo     = contents[3]

		self.model_id       = contents[4]
		self.model_cal_file = contents[5]
		self.raw_bounds     = contents[6]

		self.nbins      = contents[7]
		self.background = contents[8]

		self.ngaus = contents[9]

		gb_start = 10
		gb_stop  = 10 + 7*self.ngaus
		self.gaus_names  = contents[gb_start:gb_stop:7]
		gaus_bounds_flat = [gmap(float,contents[gb_start+1+7*_:gb_start+7+7*_]) for _ in range(self.ngaus)]
		self.gaus_bounds = [list(zip(_[0::2], _[1::2])) for _ in gaus_bounds_flat]

		self.nsmono = int(contents[gb_stop])
		smono_start = gb_stop+1
		smono_stop  = gb_stop+1 + 6*self.nsmono
		# self.smono_order  = contents[smono_start:smono_stop:5]
		smono_bounds_flat = [gmap(float,contents[smono_start+6*_:smono_start+6+6*_]) for _ in range(self.nsmono)]
		self.smono_bounds = [list(zip(_[0::2], _[1::2])) for _ in smono_bounds_flat]

		# self.ncuts = int(contents[gb_stop])
		# c_start = gb_stop + 1
		# c_stop  = gb_stop + 1 + 3*self.ncuts
		self.ncuts = int(contents[smono_stop])
		c_start = smono_stop + 1
		c_stop  = smono_stop + 1 + 3*self.ncuts
		if self.ncuts:
			self.cut_br = contents[c_start+0:c_stop:3]
			self.cut_lo = gmap(float,contents[c_start+1:c_stop:3])
			self.cut_hi = gmap(float,contents[c_start+2:c_stop:3])
		else:
			self.cut_br = []
			self.cut_lo = []
			self.cut_hi = []

		self.chi2 = float(contents[c_stop+0])
		self.ndof = float(contents[c_stop+1])

		self.npars_bg = int(contents[c_stop+2])
		self.npars    = self.npars_bg + 3*self.ngaus + 2*self.nsmono

		p_start = c_stop + 3
		p_stop  = p_start + self.npars
		self.popt = gmap(float,contents[p_start:p_stop])
		self.perr = gmap(float,contents[p_stop:])

		self.popt_bg = self.popt[:self.npars_bg]
		self.perr_bg = self.perr[:self.npars_bg]

		self.popt_gaus = self.popt[self.npars_bg:self.npars_bg+3*self.ngaus]
		self.perr_gaus = self.perr[self.npars_bg:self.npars_bg+3*self.ngaus]

		self.popt_smono = self.popt[self.npars_bg+3*self.ngaus:]
		self.perr_smono = self.perr[self.npars_bg+3*self.ngaus:]

def load_fits(file):
	# extract entries
	entries = load_csv(file,FIT_CSV_TYPELIST)
	# process each to fit_result object
	results = [fit_result(_) for _ in entries]
	# return them
	return results

class xf_result(object):
	def __init__(self, contents=False):
		if contents:
			self.unpack(contents)
	def unpack(self, contents):
		self.branch = contents.pop(0)
		self.order  = contents.pop(0)
		self.opt    = np.array(contents[:self.order+1])
		self.cov    = np.array(contents[self.order+1:]).reshape([self.order+1,self.order+1])

def load_xf(file):
	entries = load_csv(file,TYPELIST_XF)
	results = [xf_result(_) for _ in entries]
	return results







def matches_any(patterns, string):
	"""return the result of the first matching pattern"""
	for pattern in patterns:
		match = re.match(pattern, string)
		if match:
			return match

def in_range_inclusive(number, bounds):
	if (not bounds) or (number < bounds[0]) or (number > bounds[1]):
		return False
	return True


RPI_BRANCHES = {
	"area_all" :[r"area_all_([0-9]+)", r"aa_([0-9]+)", r"aa([0-9]+)"],
	"area_trig":[r"area_([0-9]+)", r"area([0-9]+)", r"a_([0-9]+)" , r"a([0-9]+)" ],
	"area_sum" :[r"area_sum_([0-9]+)", r"as_([0-9]+)", r"as([0-9]+)"],
	"time_all" :[r"time_all_([0-9]+)", r"ta_([0-9]+)", r"ta([0-9]+)"],
	"time_trig":[r"time_([0-9]+)", r"time([0-9]+)", r"t([0-9]+)"],
	"n_pulses" :[r"n_pulses_([0-9]+)", r"n_([0-9]+)", r"n([0-9]+)"],

	# "event":[r"event"],
	"vmax" :[r"v([0-9]+)", r"vmax([0-9]+)", r"vmax_([0-9]+)"],
	"tmax" :[r"t([0-9]+)", r"tmax([0-9]+)", r"tmax_([0-9]+)"],
}

# # branches not defined per channel
# RPI_BRANCHES_EVENT = {"event"}

# load data from raspberry digitizer/RPI output (.txt file)
def load_rpi_txt(file, branches=set(), trigger_window=None, ):

	channels_seen = set()
	data = {key:{} for key in RPI_BRANCHES.keys()}

	# with open(file, 'r') as file:
		
	line=file.readline().strip()
	iline=0
	while line:
		command, _, arguments = line.partition(b':')

		# process line
		if command.startswith(b"AREA"):
			channel = int(command[4:])

			# populate data with empty lists the first time
			# we see a particular channel
			if channel not in channels_seen:
				channels_seen.add(channel)
				for key in data.keys():# - RPI_BRANCHES_EVENT:
					data[key][channel] = []

			# collect pulses
			args = list(map(float, (_ for _ in arguments.strip().split(b" ") if _)))
			area = args[0::2]
			time = args[1::2]

			# find first triggering pulse
			pulse_trig = next((i for i,_ in enumerate(time) if in_range_inclusive(_, trigger_window)), -1)

			# calculate each requestable datum
			# todo: only if any branches match
			data["n_pulses"][channel].append(len(area))
			data["area_all"][channel] += area
			data["time_all"][channel] += time
			data["area_sum"][channel].append(sum(area))
			if pulse_trig >= 0:
				data["area_trig"][channel].append(area[pulse_trig])
				data["time_trig"][channel].append(time[pulse_trig])
			else:
				data["area_trig"][channel].append(0)
				data["time_trig"][channel].append(0)

		elif line.startswith(b"ANT"):
			values = list(map(int, line[5:].split(b" ")))
			nch = (len(values) - 1) // 2
			for ich in range(nch):

				if (ich+1) not in channels_seen:
					channels_seen.add(ich+1)
					for key in data.keys():# - RPI_BRANCHES_EVENT:
						data[key][ich+1] = []

				data["vmax"][ich+1].append(values[1 + ich])
				data["tmax"][ich+1].append(values[1 + ich + nch])

		elif line.startswith(b"DATA"):
			...

		# load next line
		line=file.readline().strip()
		iline += 1

	# convert data to numpy arrays
	data = {key:{ch:np.array(d) for ch,d in value.items()} for key,value in data.items()}

	# assign data to requested branches
	requested = {}
	for branch in branches:
		for name, patterns in RPI_BRANCHES.items():
			match = matches_any(patterns, branch)
			if match:
				channel = int(match.groups()[0])
				requested[branch] = data[name][channel]
				continue
	
	return requested
