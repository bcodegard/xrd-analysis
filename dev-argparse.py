"""
script used for testing features during development
don't actually use this for anything else
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"

import sys
import argparse




def main(args):

	for key,value in args.items():
		print(key,value)

	print("BLIMEY!")




if __name__ == '__main__':

	parser = argparse.ArgumentParser(
		description="testing argparse!",
		)

	# version
	parser.add_argument("--version",action="version",version="%(prog)s {}".format(__version__))

	# data arguments
	parser.add_argument("run",type=str,help="file location, name, or number")
	parser.add_argument("fit",type=str,help="branch to fit: branch,min,max")
	parser.add_argument("--cut",type=str,action='append',help="branch to cut on: branch,min,max")
	parser.add_argument("--m"  ,type=str,dest="model",help="model to use: model_id,calibration_file")
	parser.add_argument("-t"   ,action='store_true',dest="transform_bounds",help="if specified, transform bounds with model")

	# fitting arguments
	parser.add_argument("--bins",type=int,default=0, help="number of bins to use")
	parser.add_argument("--bg"  ,type=str,default="",dest="background",help="background function: any combination of (p)ower (e)xp (c)onstant (l)ine (q)uadratic")
	parser.add_argument("--g"   ,type=str,action='append',dest="gaus",help="gaussian components: min_mu,max_mu (or) name=min_mu,max_mu")

	# display arguments
	parser.add_argument("--d",type=str,default="drp",dest='display',help="display: any combinration of (d)ata (r)oot (p)eaks (s)cipy (g)uess")
	parser.add_argument("--ylim",type=int,help="upper y limit on plot")
	parser.add_argument("-x",dest="xlog",action="store_true",help="sets x axis of figure to log scale")
	parser.add_argument("-y",dest="ylog",action="store_true",help="sets y axis of figure to log scale")
	parser.add_argument("-s",dest="show",action="store_false",help="don't show figure as pyplot window")

	# output arguments
	parser.add_argument("--out",type=str,help="location to save fit results as csv file (appends if file exists)")
	parser.add_argument("--fig",type=str,help="location to save figure as png image (overwrites if file exists)")
	parser.add_argument("-v",action='count',default=0,help="verbosity")

	args = vars(parser.parse_args())
	main(args)
