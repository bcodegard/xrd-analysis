"""
script for testing functionality of utils.config
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"

import os
import utils.config as config

if __name__ == "__main__":

	print("testing function deep_merge")
	print("merge two dicts with priority for the first argument")
	print("recursively merge any shared dict values")
	d1 = {
		"a":13,
		"b":"spam",
		"c":None,
		"d": {
			"1":"one",
			"2":"two",
			"deep":{
				"learning":"yes",
				"blue":"nah",
			},
			"type_mismatch":{1:2},
		},
		"g":True,
	}
	d2 = {
		"a":31,
		"d":{
			"1":"wun",
			"3":"three",
			"deep":{
				"learning":"totally",
				"green":"yah",
			},
			"type_mismatch":4
		},
		"e":2.71828,
		"g":False,
	}
	d3 = {"a":7, "q":-1}
	print("d1 = ", d1)
	print("d2 = ", d2)
	print("d2 = ", d3)
	print("merge(d1,d2) -> ", config.deep_merge(d1,d2))
	print("merge(d2,d1) -> ", config.deep_merge(d2,d1))
	print("merge(d3,d2,d1) -> ", config.deep_merge(d3,d2,d1))


	print("")
	print("load config file test1.yaml")
	cfg = config.load("test1", verbosity=3)
	print(cfg)
	
	print("")
	print("load config file common.yaml")
	cfg = config.load(verbosity=3)
	print(cfg)
	print(os.sep.join(cfg['data_directory']['base']))

	print("")
	print("load multiple configs (test1, test2)")
	cfg = config.load(["test1","test2"], verbosity=3)
	print(cfg)


	# print("")
	# config.load("spam", verbosity=3)

	# print("")
	# config.load(["spam", "eggs", "baked_beans"], verbosity=3)

	# print("")
	# config.load(["sausages", "spam"], "/custom/config/path", verbosity=3)

