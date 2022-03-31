"""
script used for testing features during development
don't actually use this for anything else
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"

import sys
import argparse
import numpy as np

import utils.cli as cli




def main(args):
	max_key_length = max([len(_) for _ in args.keys()])
	for key,value in args.items():
		print("{:<{l}} : {}".format(key,value,l=max_key_length))
	print("BLIMEY! Those are the results!")

if __name__ == '__main__':

	parser = argparse.ArgumentParser(
		description="testing argparse!",
		)

	# version
	parser.add_argument("--version",action="version",version="%(prog)s {}".format(__version__))

	# custom action class
	parser.add_argument("--merge"          , "--m" , type=str  , nargs="+", default=None, action=cli.MergeAction         , const=((str,int,float),("",1,-np.inf,np.inf)))
	parser.add_argument("--merge-nodefault", "--mn", type=str  , nargs="+", default=None, action=cli.MergeAction         , const=((str,float),))
	parser.add_argument("--merge-append"   , "--ma", type=str  , nargs="+", default=None, action=cli.MergeAppendAction   , const=((str,int),("",0,1,2,3)))
	parser.add_argument("--function"       , "--f" , type=float, nargs="+", default=None, action=cli.FunctionAction      , const=lambda vs:sum(vs))
	parser.add_argument("--function-change", "--fc", type=float,            default=1   , action=cli.FunctionChangeAction, const=lambda vs,o:vs*o)
	parser.add_argument("--function-append", "--fa", type=str  , nargs="+", default=None, action=cli.FunctionAppendAction, const=lambda vs:vs[::-1])

	# multiple merge append actions with same dest
	# 
	# example: model components and parameters
	#          entries are ["function_id", *bounds_or_parameters]
	# 
	# some predefined arguments representing specific choices
	# as well as a general argument that can specify anything
	comp_callables = [float, ]
	comp_defaults  = [0.0, 1.0, ]
	parser.add_argument(
		"--a",
		dest="components",
		type=str,
		nargs="+",
		action=cli.FunctionAppendAction,
		const=lambda values:["a"] + cli.merge(values, comp_callables, comp_defaults)
	)
	parser.add_argument(
		"--b",
		dest="components",
		type=str,
		nargs="+",
		action=cli.FunctionAppendAction,
		const=lambda values:["b"] + cli.merge(values, comp_callables, comp_defaults)
	)
	parser.add_argument(
		"--component", "--comp", "--c",
		dest="components",
		type=str,
		nargs="+",
		action=cli.MergeAppendAction,
		const=([str]+comp_callables, [""]+comp_defaults)
	)



	if len(sys.argv) <= 1:
		print("no arguments supplied. running with example set of arguments.")
		args_ex = "--ma wat -1 0 1 --ma taw --ma twa 1 --m amp 2 0 --f 1 3 4 2 3 4.5 --f -0.5 12 --fc 0.95 --fc 5.0 --fc 0.34 --fa a b c --fa a a c --fa 1 2 3 --mn kernel 1 0 2 -1 1 -2 0 -1"
		args_ex_split = [_ for _ in args_ex.split(' ') if _]
		args = vars(parser.parse_args(args_ex_split))
	else:
		args = vars(parser.parse_args())


	main(args)
