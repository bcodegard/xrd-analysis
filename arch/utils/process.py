""""""

__author__ = "Brunel Odegard"
__version__ = "0.1"


import math
import numpy as np



def shape_dims(ndims, this_index, this_size):
	shape = np.ones(n_axes)
	shape[this_index] = this_size
	return shape

def add_dims(arrays):
	return [ar.reshape(shape_dims(len(arrays), iar, ar.size)) for iar,ar in enumerate(arrays)]



class binned_axis(object):
	"""Docstring for binned_axis class"""

	def __init__(self,
			name  = None,
			units = None,
			edges     = None,
			left      = None,
			right     = None,
			midpoints = None,
			widths    = None,
			weights = None,
			ndims = None,
			idim  = None,
			):
		
		# assign name and units
		self.name = name
		self.units = units

		# assign bin characteristics
		self.left      = left
		self.right     = right
		self.midpoints = midpoints
		self.widths    = widths

		# assign bin weights
		self.weights = weights

		# if edges supplied, take left and right from them,
		# Edges are assumed to be contiguous if supplied this way. To use
		# non-contiguous bins, supply left and right edges separately.
		if edges:
			self.left = edges[:-1]
			self.right = edges[1:]

		# calculate midpoints and widths from edges, if not supplied.
		if self.midpoints is None:
			self.midpoints = 0.5 * (self.left + self.right)
		if self.widths is None:
			self.widths = self.right - self.left

		# midpoints and widths are guaranteed to be defined, so we can
		# get the number of bins using them.
		self.nbins = self.midpoints.size

		# set dimensionality if specified.
		self.set_dims(ndims, idim)

	def set_dims(self, ndims, idim):
		"""Define the number of dimensions, and which one belongs to
		this axis. Reshape bin arrays to match."""

		# assign ndims and idim
		self.ndims = ndims
		self.idim = idim

		# if ndims is defined, reshape bin arrays to match ndims and idim
		# otherwise, reshape to flat (1d)
		if self.ndims:
			nd_shape = shape_dims(self.ndims, self.idim, self.nbins)
		else:
			nd_shape = (self.nbins, )

		# reshape all defined arrays
		if self.left is not None:
			self.left = self.left.reshape(nd_shape)
		if self.right is not None:
			self.right = self.right.reshape(nd_shape)
		if self.midpoints is not None:
			self.midpoints = self.midpoints.reshape(nd_shape)
		if self.widths is not None:
			self.widths = self.widths.reshape(nd_shape)
		
		if self.weights is not None:
			self.weights = self.weights.reshape(nd_shape)

	def as_dims(self, ndims, idim):
		"""Return a copy of the object with the specified dimensionality"""
		return binned_axis(
			name      = self.name     ,
			units     = self.units    ,
			edges     = self.edges    ,
			left      = self.left     ,
			right     = self.right    ,
			midpoints = self.midpoints,
			widths    = self.widths   ,
			weights   = self.weights  ,
			ndims = ndims,
			idim  = idim ,
		)

	# def __repr__(self, ):
	# 	return "axis {}; left={}; right={}".format([
	# 		str(self),
	# 		self.left,
	# 		self.right,
	# 	])

	def __str__(self, ):
		components = []
		if self.name:
			components.append(self.name)
		if self.units:
			if components:
				components.append("({})".format*self.units)
			else:
				components.append(self.units)
		return " ".join(components)




class projection_integrator(object):
	"""Docstring for projection_integrator"""


	def __init__(self, func, axes, default_integrate = False):

		# The function being evaluated. Should take arguments:
		# func(param, axes)
		# Should return array whose axes are the given axes, in order.
		self._func = func

		# assign axes and set their dimensionality
		# Note that this assignment means that the given axes are changed
		# by this object. This means that new axis objects should be used
		# for each instance of this class unless it's known that the
		# dimensionalities will match exactly.
		self.axes = axes
		for iax,ax in enumerate(self.axes):
			ax.set_dims(len(self.axes), iax)

		# Default list of axes to integrate over when evaluating.
		# If list is not specified, this default value is used.
		self.default_integrate = default_integrate

	def __call__(self, param, integrate = "auto"):
		"""Evaluate result for the given param, integrating over the
		specified axes, or the default list of axes if unspecified."""

		# evaluate function with supplied param
		data = self._func(param, self.axes)

		# apply weights
		if self._weights is not None:
			data *= self._weights

		# Sum over all axes specified.
		# integrate can be a list of axes, None, or False.
		# If it's false, no integration will be performed.
		# If it's None, then all axes will be integrated over.
		# If it's a list, then all specified axes will be integrated over.
		if integrate == "auto":
			integrate = self.default_integrate
		if integrate is not False:			
			# numpy handles this automatically :)
			data = np.sum(data, axis=integrate)

		# return the resulting data
		return data









