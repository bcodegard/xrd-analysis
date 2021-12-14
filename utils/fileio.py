"""
contains code for reading and writing files
and associated routines and data structures
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"


import os
import csv
import uproot
import numpy as np




# globals

# default top level rootkey
ROOTKEY_DEFAULT=b'Events;1'

# list of types that csv entries aren't allowed to be
CSV_TYPES_FORBIDDEN = [list, tuple]

# list of types recognized as a row for csv files
CSV_TYPES_ROW = [list, tuple]

# error message templates
ERR_FILE_MISSING = "file {} does not exist"
ERR_FORBIDDEN_TYPE = "forbidden type {} found as entry"

# csv typelists for established uses
TYPELIST_CALIBRATION = [str, float, int, int, float, int, float]




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
	
	# default rootkey if not specified
	if rootkey is None:
		rootkey = ROOTKEY_DEFAULT

	# open root file and get list of branch names (keys)
	with uproot.open(rootfile) as root_obj:
		trees = root_obj[rootkey]
		keys = trees.keys()

	# cast to strings and return
	return list(map(to_str, keys))

def load_branches(rootfile, which=set(), rootkey=None):
	"""loads branches specified in <which> from <rootfile>. loads all if <which> is empty"""

	# default rootkey if not specified
	if rootkey is None:
		rootkey = ROOTKEY_DEFAULT

	# allows single branch to be specified as string instead of set
	if type(which) is str:
		which = {which}

	# open root file and extract branches as arrays
	with uproot.open(rootfile) as root_obj:
		trees = root_obj[rootkey]
		keys = trees.keys()
		if which:
			branches = {to_str(key):trees.array(key) for key in keys if to_str(key) in which}
		else:
			branches = {to_str(key):trees.array(key) for key in keys}

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
		if branch != entry[0]:
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
	def __init__(self, contents):
		self.load_from_contents(contents)

	def load_from_contents(self, contents):
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
		self.gaus_bounds = [gmap(float,contents[gb_start+1+7*_:gb_start+7+7*_]) for _ in range(self.ngaus)]

		self.nsmono = int(contents[gb_stop])
		smono_start = gb_stop+1
		smono_stop  = gb_stop+1 + 5*self.nsmono
		self.smono_order  = contents[smono_start:smono_stop:5]
		self.smono_bounds = [gmap(float,contents[smono_start+1+5*_:smono_start+5+5*_]) for _ in range(self.nsmono)]

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




# root cases
# ...

