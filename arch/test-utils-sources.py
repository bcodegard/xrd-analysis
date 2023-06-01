"""
testing functionality of utils.sources
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"


import utils.sources as src


if __name__ == "__main__":

	# __str__ is defined but __repr__ is not
	test_str_repr = False
	if test_str_repr:
		print(repr(src.Ba133))
		print(src.Ba133)
		print("source: {}".format(src.Ba133))

	# errors
	test_name_set = False
	if test_name_set:
		src.Ba133.name = "huh"

	# does not change object's attribute
	test_name_reference_set = False
	if test_name_reference_set:
		name = src.Ba133.name
		name = "huh"
		print(name)
		print(src.Ba133.name)	

	#
	test_emissions_access = True
	if test_emissions_access:

		dec = src.Ba133.emissions
		for _ in dec:
			print("")

			# the property definition
			print(_)
			print(_.energy_kev)

			# an assignment to it
			# behaves same way. goes through propety, and can't be set.
			print(_.e, _.e_err)
			# _.e = 4
		
		for _ in src.Am241.emissions:
			print("")
			print(_)
			print("i", _.i, _.i_err)
			print("e", _.e, _.e_err)

