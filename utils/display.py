"""
methods for creating visual displays of data
"""

# __all__ = ...
__author__ = "Brunel Odegard"
__version__ = "0.0"


import math



def bin_count_from_n_data(n_data, factor=4, min_bins=25, max_bins=1000):
	n_bins = math.ceil(factor*math.sqrt(n_data))
	if min_bins is not None:
		n_bins = max([n_bins,min_bins])
	if max_bins is not None:
		n_bins = min([n_bins,max_bins])
	return n_bins



# # matplotlib utilities
# # figure dumping / loading breaks some things
# # deprecated until issue can be resolved

# def stack_axes(donors, recipient, inherit_legend=True, inherit_labels=0, inherit_title=0):
# 	"""copies and merges contents of donors into recipient. assumes recipient starts empty."""

# 	# copy lines
# 	for donor in donors:

# 		print(len(donor.get_lines()))

# 		for line in donor.get_lines():

# 			# recipient.plot(*line.get_data())
# 			# cline = recipient.get_lines()[-1]

# 			# cline = matplotlib.lines.Line2D(*line.get_data(orig=False))
# 			cline = matplotlib.lines.Line2D(*line.get_data(orig=True),axes=recipient)
# 			recipient.add_line(cline)

# 			cline.set_linestyle(line.get_linestyle())
# 			cline.set_linewidth(line.get_linewidth())
# 			cline.set_color(line.get_color())
# 			cline.set_markersize(line.get_markersize())
# 			cline.set_markerfacecolor(line.get_markerfacecolor())
# 			cline.set_markerfacecoloralt(line.get_markerfacecoloralt())
# 			cline.set_markeredgecolor(line.get_markeredgecolor())
# 			cline.set_markeredgewidth(line.get_markeredgewidth())
# 			cline.set_dash_capstyle(line.get_dash_capstyle())
# 			cline.set_dash_joinstyle(line.get_dash_joinstyle())
# 			cline.set_solid_capstyle(line.get_solid_capstyle())
# 			cline.set_solid_joinstyle(line.get_solid_joinstyle())
# 			cline.set_marker(line.get_marker())
# 			cline.set_drawstyle(line.get_drawstyle())

# 	# copy legends
# 	if inherit_legend:
# 		labels_found = []
# 		handles = []
# 		labels = []
# 		for donor in donors:
# 			this_handles,this_labels = donor.get_legend_handles_labels()
# 			for il,l in enumerate(this_labels):
# 				if l not in labels_found:
# 					handles.append(this_handles[il])
# 					labels.append(l)
# 					labels_found.append(l)
# 		recipient.legend(handles, labels)

# 	# copy labels
# 	if inherit_labels is not None:
# 		donor = donors[inherit_labels]
# 		recipient.set_xlabel(donor.get_xlabel())
# 		recipient.set_ylabel(donor.get_ylabel())

# 	# copy title
# 	if inherit_title is not None:
# 		donor = donors[inherit_title]
# 		recipient.set_title(donor.get_title())

# 	# return modified recipient
# 	return recipient
