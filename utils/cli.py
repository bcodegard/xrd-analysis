"""
functions, methods, classes, etc. relating to command line interface behavior
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"

import argparse



def infer_dest(option_string, prefix_chars='-'):
	return option_string.lstrip(prefix_chars).replace('-','_')



# custom argparse.Action action classes and associated objects

class MultipleDestAction(argparse.Action):
	def __init__(self, option_strings, dest, **kwargs):
		if isinstance(dest, str):
			raise ValueError("MultipleDestAction requirest iterable of strings for dest")
		else:
			true_dest = infer_dest(option_strings[0])
			self.separate_dests = dest
		super(MultipleDestAction, self).__init__(option_strings, true_dest, **kwargs)
		
	def __call__(self, parser, namespace, values, option_string=None):
		setattr(namespace, self.dest, values)
		value_ignore='.'
		for ival,val in enumerate(values):
			if val == value_ignore:
				continue
			setattr(
				namespace,
				self.separate_dests[ival],
				self.const[ival](val),
			)


class MergeAction(argparse.Action):
	"""casts values using callables in const[0]
	appends from list of defaults const[1] if len(values) < len(defaults)
	defaults are cast using callables iff const[2]
	supplying more values than there are defaults is forbidden iff const[3]"""

	def __init__(self, option_strings, dest, **kwargs):
		super(MergeAction, self).__init__(option_strings, dest, **kwargs)
	
	def __call__(self, parser, namespace, values, option_string=None):
		setattr(namespace, self.dest, merge(values, *self.const))


class MergeAppendAction(argparse.Action):
	"""same as merge action, but appends to list instead of overwriting
	when argument is supplied multiple times

	casts values using callables in const[0]
	appends from list of defaults const[1] if len(values) < len(defaults)
	defaults are cast using callables iff const[2]
	supplying more values than there are defaults is forbidden iff const[3]"""

	def __init__(self, option_strings, dest, **kwargs):
		super(MergeAppendAction, self).__init__(option_strings, dest, **kwargs)
	
	def __call__(self, parser, namespace, values, option_string=None):

		if getattr(namespace, self.dest) == self.default:
			setattr(namespace, self.dest, [merge(values, *self.const)])
		else:
			getattr(namespace, self.dest).append(merge(values, *self.const))


class FunctionAction(argparse.Action):
	"""calls function self.const with values, but not any previous attribute self.dest
	sets attribute self.dest to return from function call"""

	def __init__(self, option_strings, dest, **kwargs):
		super(FunctionAction, self).__init__(option_strings, dest, **kwargs)

	def __call__(self, parser, namespace, values, option_string=None):
		setattr(namespace, self.dest, self.const(values))


class FunctionChangeAction(argparse.Action):
	"""calls function self.const with values and previous attribute self.dest
	sets attribute self.dest to return from function"""

	def __init__(self, option_strings, dest, **kwargs):
		super(FunctionChangeAction, self).__init__(option_strings, dest, **kwargs)
	
	def __call__(self, parser, namespace, values, option_string=None):
		setattr(namespace, self.dest, self.const(values, getattr(namespace, self.dest)))


class FunctionAppendAction(argparse.Action):
	"""calls function self.const with values
	appends result to attribute self.dest"""

	def __init__(self, option_strings, dest, **kwargs):
		super(FunctionAppendAction, self).__init__(option_strings, dest, **kwargs)
		
	def __call__(self, parser, namespace, values, option_string=None):
		if getattr(namespace, self.dest) == self.default:
			setattr(namespace, self.dest, [self.const(values)])
		else:
			getattr(namespace, self.dest).append(self.const(values))


def merge(values, callables, defaults=(), cast_defaults=False, no_extra_values=False, value_ignore='.'):
	"""cast values using callables, padding out with defaults as needed"""

	vlist = []
	n_values    = len(values)
	n_callables = len(callables)
	n_defaults  = len(defaults)

	# if specified, require that there are not more specified values than there are defaults
	if no_extra_values:
		assert n_values <= n_defaults

	# iterate through all values present
	for i in range(max([n_values, n_defaults])):
		
		# index of callable to use for this entry
		# stops at end of callables, so that all entries
		# past that point are cast using the last callable
		i_callable = min([i, n_callables - 1])

		# value supplied
		if (i < n_values) and (values[i] != value_ignore):
			vlist.append(callables[i_callable](values[i]))

		# value not supplied
		else:
			vlist.append(callables[i_callable](defaults[i]) if cast_defaults else defaults[i])

	return vlist
