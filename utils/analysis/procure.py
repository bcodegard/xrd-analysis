"""
"""






class Procure(object):

	# list of branches which are constructed by the manager after loading.
	BRANCHES_CONSTRUCT = ['entry']

	def __init__(self, spc, immediate=True):
		self.spc = spc
		
		if immediate:
			self.fulfill()

	def fulfill(self):

		# find specified dataset, and open an interface for it
		self.get_dataset()

		# using context acquired from dataset, apply shorthand conversions
		# to all functions of dataset contents
		self.apply_shorthand()

		# compile expressions into functions
		self.compile_expressions()

		# determine which branches need to be acquired from the dataset
		# and load those branches
		self.load_dataset_branches()

		# construct special branches, which are any branch included 
		# in Procure.BRANCHES_CONSTRUCT
		self.construct_special_branches()

		# apply specific adjustments to some branches, including
		# scaler rectification, timestamp fixes, and timestamp localization
		self.apply_branch_adjustments()

		# create new branches by evaluating branch definitions on 
		# existing branches. then discard any branches that are no
		# longer needed.
		self.evaluate_defs()

		# calculate cut arrays and use them to filter branches.
		# then discard unneeded branches.
		self.apply_cuts()

		# if binning is defined, create arrays for bin edges and centers,
		self.calculate_binning()

		# if binning is defined, calculate histogram counts.
		self.calculate_counts()






	def get_dataset(self):
		""""""

		# defines:
		# 
		# self.dfi
		# self.context
		# 

	def apply_shorthand(self):
		""""""

		# modifies values in self.spc
		# 
		# replaces shortand with expanded forms for all functions
		# of dataset contents





