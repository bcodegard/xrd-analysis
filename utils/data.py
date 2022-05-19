"""
data handling and processing algorithms
"""

# __all__ = ...
__author__ = "Brunel Odegard"
__version__ = "0.0"


import math
import numpy as np




# utility functions

def add_suffix(string,suffix):
	"""if suffix, appends suffix to string with underscore. otherwise return it unchanged."""
	if suffix is None:
		return string
	else:
		return "{}_{}".format(string, suffix)

AP_DELIMITER = ","
def split_with_defaults(s,defaults,types=None,delimiter=AP_DELIMITER,name_delimiter=None,allow_blank=False):
	"""split string by delimiter and cast to list with defaults and types"""

	# list/tuple of strings -> individual calls per entry
	if type(s) in [list, tuple]:
		return [split_with_defaults(_,defaults,types,delimiter,name_delimiter,allow_blank) for _ in s]

	# single string
	else:
		result = []
		
		if name_delimiter is not None:
			if name_delimiter in s:
				name,_,s = s.rpartition(name_delimiter)
			else:
				name=""

		parts = s.split(delimiter)
		for i,part in enumerate(parts):
			if part or allow_blank:
				if types:
					result.append(types[i](part))
				else:
					result.append(part)
			else:
				result.append(defaults[i])

		result = result + defaults[len(result):]

		if name_delimiter is not None:
			result = [name] + result

		return result

def edges_lin(xmin, xmax, nbins):
	return np.linspace(xmin, xmax, nbins+1)

def edges_equal_count(nbins, xdata, xmin, xmax):
	xdata = xdata[np.logical_and(xdata>xmin, xdata<xmax)]
	count = xdata.size
	return np.interp(
		np.linspace(0,count,nbins+1),
		np.arange(count),
		np.sort(xdata),
	)

def edges_log(xmin, xmax, nbins):
	return np.logspace(math.log(xmin,10), math.log(xmax,10), nbins+1)

def edges_symlog(xmin, xmax, nbins, l=1):
	slxmin = symlog(xmin, l)
	slxmax = symlog(xmax, l)
	y = np.linspace(slxmin, slxmax, nbins+1)
	return isymlog(y, l)

def bin_count_from_ndata(ndata, mult=4.0, minimum=50):
	"""nbins proportional to sqrt(ndata), with minimum"""
	nraw = math.ceil(mult * math.sqrt(ndata))
	return max([minimum, nraw])




# data processing algorithms
# for internal and external use

def chi2_identical_poisson(a,b):
	"""calculate reduced chi2 for the hypothesis that pairs of elements
	from a and b are drawn from identical poisson distributions"""
	
	# exclude points where a and b are both zero
	ftr_valid = (a>0)|(b>0)
	av = a[ftr_valid]
	bv = b[ftr_valid]

	# return chi2, ndof
	return (((av-bv)**2)/(av+bv)).sum(), ftr_valid.sum()

def rectify(array, radius):
	"""raises values if they are lower than all their neighbors"""
	max_l = np.stack([np.roll(array, _) for _ in range(-radius, 0       )],axis=0).max(0)
	max_r = np.stack([np.roll(array, _) for _ in range(1      , radius+1)],axis=0).max(0)
	return np.maximum(array, np.minimum(max_l, max_r))

def fix_monotonic(array, round_to=0, copy=True):
	"""fixes inversions in array, which is supposed to be monotonic"""
	anomalies = array[1:] < array[:-1]
	differentials = array[:-1][anomalies] - array[1:][anomalies]
	if not (round_to is None):
		differentials = np.round(differentials, round_to)
	if copy:
		new_array = array.copy()
	else:
		new_array = array
	new_array[1:][anomalies] += differentials
	return new_array

def symlog(x, l):

	isscalar = np.isscalar(x)
	x = np.atleast_1d(x)

	y = np.zeros(x.shape)
	
	ftr_pos = (x >  l)
	ftr_neg = (x < -l)
	ftr_lin = np.logical_not(np.logical_or(ftr_pos,ftr_neg))
	
	b=math.e/l
	y[ftr_pos] =  np.log( b*x[ftr_pos])
	y[ftr_neg] = -np.log(-b*x[ftr_neg])
	y[ftr_lin] = x[ftr_lin] * (b/math.e)
	
	if isscalar:
		return y[0]
	else:
		return y

def isymlog(y, l):
	
	isscalar = np.isscalar(y)
	y = np.atleast_1d(y)

	x = np.zeros(y.shape)

	ftr_pos = (y >  1)
	ftr_neg = (y < -1)
	ftr_lin = np.logical_not(np.logical_or(ftr_pos,ftr_neg))

	x[ftr_pos] =  np.exp( y[ftr_pos] - 1)
	x[ftr_neg] = -np.exp(-y[ftr_neg] - 1)
	x[ftr_lin] = y[ftr_lin]

	if isscalar:
		return x[0] * l
	else:
		return x * l

def inrange(arr, lo, hi, lclosed=False, rclosed=False):
	pieces = []
	if lo not in [None, -np.inf]:
		pieces.append(arr>=lo if lclosed else arr>lo)
	if hi not in [None, np.inf]:
		pieces.append(arr<=hi if rclosed else arr<hi)
	if len(pieces) == 1:
		return pieces[0]
	else:
		return np.logical_and(*pieces)




# Accessor base class and subclasses
# Currently, no Accessor class except the base class is implemented.
# To create a new Accessor, sublass the Accessor class and overwrite the
# Accessor.get method to interface with your data.

class Accessor(object):
	"""base class for Accessor types"""
	def __init__(self):
		pass
	def __nonzero__(self):
		return True
	def get(self, key_or_keys):
		return False

# class RootFileAccessor(Accessor):
# 	"""Accessor object for loading branches as needed from root file"""
# 	def __init__(self, tree, rootkey=None):
# 		super(RootFileAccessor, self).__init__()
# 		self.tree = tree
# 		self.rootkey = rootkey
# 	def get(self, key_or_keys):
# 		...
# 	def set(self, item_or_items):
# 		...

# class MMapAccessor(Accessor):
# 	"""Accessor object for accessing arrays from memory-mapped disk storage"""
# 	def __init__(self, ):
# 		super(MMapAccessor, self).__init__()
# 		...




# BranchManager class and associated methods

ERR_BRANCHES_INVALID = "branches must be dict or be falsy"
ERR_BRANCH_NOT_FOUND = "branch with key {} not found"
class BranchManager(object):
	"""provides access and control for a collection of branches"""

	def __init__(self, branches=None, accessor=None, forgetful=False, export_copies=True, import_copies=True, tolerate_missing=False, ):

		if not branches:
			self._branches = {}
		elif type(branches) is dict:
			self._branches = branches.copy()
		else:
			raise ValueError(ERR_BRANCHES_INVALID)

		# accessor object
		# if defined, used to acquire branches as needed
		self._accessor = accessor

		# list of masks applied
		# new branches acquired from accessor must be masked accordingly
		self._accessor_masks = []

		# if True, we don't keep arrays in memory inside this object
		self._forgetful = forgetful
		self._forget_if_forgetful()

		# whether to apply copying as protetcion in external interactions
		self.export_copies = export_copies
		self.import_copies = import_copies
		
		# True: return None when asked for branch and can't find it
		# False: raise Error
		self.tolerate_missing = tolerate_missing

	def __len__(self):
		test_key = list(self.keys)[0]
		return self._get(test_key).shape[0]

	def __getitem__(self, key_or_keys):
		"""official method of external access to branches. allows for copying to protect original data."""

		# string: single key. just return branch
		if type(key_or_keys) is str:
			return self._get(key_or_keys, is_export=True)

		# set: dict of key:branch
		elif type(key_or_keys) is set:
			return {key:self._get(key,is_export=True) for key in key_or_keys}

		# list, tuple, or other iterable: list of branches in same order
		else:
			return [self._get(key,is_export=True) for key in key_or_keys]

	@property
	def keys(self):
		"""read-only keys method. does not include keys not yet loaded but accessible through Accessor"""
		return self._branches.keys()


	def _forget_if_forgetful(self):
		"""if we're forgetful, remove own references to arrays by setting branches to empty dict"""
		if self._forgetful:
			self.branches = {}

	@property
	def forgetful(self):
		return self._forgetful
	
	@forgetful.setter
	def forgetful(self, new_forgetful):
		self._forgetful = bool(new_forgetful)
		self._forget_if_forgetful()


	def _get_from_accessor(self, key):
		"""requests a branch from the accessor, and applies any saved masks as appropriate"""
		result = self._accessor.get(key)
		if result:
			for mask in self._accessor_masks:
				result = result[mask]
		return result

	def _get(self, key, is_export=False):
		"""fetch a branch from branches kept in memory, or request from accessor if needed."""
		branch = self._branches.get(key,None)

		# branch not found in self._branches
		if branch is None:

			# if we have an accessor, request the branch from it
			if self._accessor:
				result = self._get_from_accessor(key)

				# if the accessor found the branch
				if result:
					branch = result

					# remember if if we're not _forgetful
					if not self._forgetful:
						self._add(key, branch)

		# branch was not found in self._branches or with self._accessor
		if branch is None:
			if not (self.tolerate_missing):
				raise ValueError(ERR_BRANCH_NOT_FOUND.format(key))
		
		# copy if needed
		if (branch is not None) and is_export and self.export_copies and (not self._forgetful):
			return branch.copy()
		else:
			return branch

	def _set(self, key, value, is_import=False):
		"""assign key:value in self._branches"""
		assert not self._forgetful
		if is_import and self.import_copies:
			value = value.copy()
		self._branches[key] = value

	def _add(self, key, value, is_import=False):
		"""set, but fail if keys already exist"""
		assert key not in self.keys
		self._set(key, value, is_import)


	def graft(self, new_branches, overwrite=False):
		"""merge new_branches into stored branches"""
		if not overwrite:
			assert not (new_branches.keys()&self.keys)
		for key, value in new_branches.items():
			self._set(key, value, is_import=True)

	def bud(self, buds=[], keep=True, overwrite=False):
		"""create new branch(es)"""
		
		# single bud -> list
		if type(buds) not in (set, list, tuple):
			buds=[buds]

		# create dict of new branches
		new_branches = {}
		
		# iterate through buds
		for b in buds:
			
			this_new_branches = b(self)

			# check for overlap
			if not overwrite:
				assert not (new_branches.keys()&this_new_branches.keys())

			# merge
			new_branches |= this_new_branches
			del this_new_branches

		# graft new branches onto self
		if keep and (not self._forgetful):
			self.graft(new_branches, overwrite)

		# return branches
		# no need to copy here; graft takes care of that.
		return new_branches

	def mask(self, mask, key_or_keys=None, apply_mask=False):
		"""return any requested branches with mask applied.
		if apply_mask, irreversibly applies boolean mask to all stored arrays"""
		
		# calculate mask if it's not given as an array
		if callable(mask):
			mask = mask(self)

		# determine keys of branches to apply mask to
		# applying mask: use all keys
		if apply_mask:
			keys_mask = self.keys
		# not applying mask: just use keys determined by key_or_keys
		else:
			if key_or_keys:
				keys_mask = key_or_keys
				if type(keys_mask) is str:
					keys_mask = {keys_mask}
			else:
				print("WARNING: no branches requested, and apply_mask is False. This call will do nothing.")
				keys_mask=set()

		# create bud function for applying the mask
		mask_branches = lambda manager:{key:manager[key][mask] for key in keys_mask}

		# bud, keep and overwrite if apply_mask
		masked_branches = self.bud([mask_branches],keep=apply_mask,overwrite=apply_mask)

		# return cases based on type(key_or_keys)
		# None: return
		if key_or_keys is None:
			return
		# str: single branch
		if type(key_or_keys) is str:
			return masked_branches[key_or_keys]
		# set: dict of key:branch
		elif type(key_or_keys) is set:
			return {key:value for key,value in masked_branches.items() if key in key_or_keys}
		# other iterable: list of branches in order they appear in key_or_keys
		else:
			return [masked_branches[_] for _ in key_or_keys]




# built-in mask methods

def mask_range(lo=0,hi=np.inf):
	"""convenient and computationally efficient method for selecting literal range in data
	does not cut on "entry" branch; rather, cuts on index of arrays first axes
	which will be different if any cuts have previously been applied"""
	def mask(manager):
		lo_enf = max([lo,0])
		hi_enf = min([hi,len(manager)])
		pieces = []
		if lo_enf > 0:
			pieces.append(np.zeros(lo_enf))
		if hi_enf > lo_enf:
			pieces.append(np.ones(hi_enf-lo_enf))
		if hi_enf < len(manager):
			pieces.append(np.zeros(len(manager)-hi_enf))
		return np.concatenate(pieces).astype(bool)
	return mask

def mask_all(*masks):
	"""logical_all of masks"""
	def mask(manager):
		return np.all(np.stack([_(manager) for _ in masks], axis=0), axis=0)
	return mask

def mask_any(*masks):
	"""logical_any of masks"""
	def mask(manager):
		return np.any(np.stack([_(manager) for _ in masks], axis=0), axis=0)
	return mask

def cut(branch, lo=-np.inf, hi=np.inf):
	"""lo<branch<hi"""
	def mask(manager):
		branch_array = manager[branch]
		ftr_lo = np.ones(branch_array.shape[0]) if lo is -np.inf else branch_array>lo
		ftr_hi = np.ones(branch_array.shape[0]) if hi is  np.inf else branch_array<hi
		return np.logical_and(ftr_lo,ftr_hi)
	return mask




# built-in bud methods

def bud_function(input_branch, output_branch, callable, args=[], kwargs={}):
	"""callable(input_branch, *args, **kwargs) -> output_branch"""
	def bud(manager):
		return {output_branch:callable(manager[input_branch], *args, **kwargs)}
	return bud

def differentiate_branch(branch, suffix="deriv"):
	"""calculates difference between each entry and the previous
	first entry in the new branch is difference between first and last entries in the input"""
	def bud(manager):
		return {add_suffix(branch,suffix):manager[branch]-np.roll(manager[branch],1)}
	return bud

def convolve_branch(branch, kernel, suffix=None):
	"""convolve a branch with a kernel. integerl kernel will be converted to normalized flat kernel."""
	if type(kernel) is int:
		kernel = np.ones(kernel)/kernel
	def bud(manager):
		return {add_suffix(branch,suffix):np.convolve(manager[branch], kernel, mode='same')}
	return bud

def rectify_branch(branch, radius, suffix=None):
	"""apply rectification algorith to a branch"""
	def bud(manager):
		return {add_suffix(branch,suffix):rectify(manager[branch],radius)}
	return bud

def rectify_scaler(radius=12, suffix=None, match="scaler_"):
	"""apply recfification to all branches whose name starts with "scaler_" (or different string if specified.)"""
	def bud(manager):
		return {add_suffix(_,suffix):rectify(manager[_],radius) for _ in manager.keys if _.startswith(match)}
	return bud

def fix_monotonic_branch(branch, suffix):
	"""fixes instances where value of branch is smaller than that of previous entry"""
	def bud(manager):
		return {add_suffix(branch,suffix):rectify(manager[branch],radius)}
	return bud

def fix_monotonic_timestamp(suffix=None, match="timestamp_"):
	"""apply monotonic fix to all branches starting with "timestamp_" (or different string if specified)"""
	def bud(manager):
		return {add_suffix(_,suffix):fix_monotonic(manager[_]) for _ in manager.keys if _.startswith(match)}
	return bud

def count_passing(mask,new_key):
	"""returns a branch with an integer index tracking how many events up to this point passed the mask"""
	def bud(manager):
		passing = mask(manager)
		return {new_key:np.cumsum(passing)}
	return bud

def subtract_first(branch,suffix=None):
	def bud(manager):
		return {add_suffix(branch,suffix):manager[branch]-manager[branch][0]}
	return bud

def localize_timestamp(suffix=None, match="timestamp_"):
	"""subtract first entry from all branches starting with "timestamp_" (or different string if specified)"""
	def bud(manager):
		return {add_suffix(_,suffix):manager[_]-manager[_][0] for _ in manager.keys if _.startswith(match)}
	return bud

def bud_entry(manager):
	return {"entry":np.arange(len(manager))}

