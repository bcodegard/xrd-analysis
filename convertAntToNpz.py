"""
process .ant pulse files into .npz files
"""

__author__ = "Brunel Odegard"
__version__ = "0.1"

import os
import argparse

import xrd.core.cli    as cli
import xrd.core.config as config

import xrd.process.ant as ant


# todo: put these in config file
INDIR  = os.sep.join([".", "data", "rpi", "ant"])
OUTDIR = os.sep.join([".", "data", "rpi", "npz"])
FI_TEMPLATE = "Run{}_dirpipulses.ant"
FO_TEMPLATE = "Run{}.npz"


def compose_parser():
	
	parser = argparse.ArgumentParser(
		description="process one or more .ant pulse files into .npz files",
	)

	# version
	parser.add_argument("--version",action="version",version="%(prog)s {}".format(__version__))

	# the main argument: which runs to process.
	# each argument can be a single number, or a range indicated by 
	# a dash, E.G. 1000-1500 (inclusive on both sides.)
	parser.add_argument(
		"runs",
		type=str,
		nargs="+",
		help="",
	)

	parser.add_argument(
		"--exclude",
		type=str,
		nargs="+",
		default=[],
	)

	# input directory
	parser.add_argument(
		"--indir", "--in", "--i",
		type=str,
		default=INDIR,
		help="directory in which to look for input .ant files",
	)

	# output directory
	parser.add_argument(
		"--outdir", "--out", "--o",
		type=str,
		default=OUTDIR,
		help="directory in which to put .npz files",
	)

	# input filename template
	parser.add_argument(
		"--infile", "--if", "--fi",
		type=str,
		default=FI_TEMPLATE,
		help="filename template for input files, default {}".format(repr(FI_TEMPLATE))
	)

	# output filename template
	parser.add_argument(
		"--outfile", "--of", "--fo",
		type=str,
		default=FO_TEMPLATE,
		help="filename template for output files, default {}".format(repr(FO_TEMPLATE))
	)

	# reprocess
	# files whose output .npz file already exists will be ignored,
	# unless this flag is given.
	parser.add_argument(
		"-r",
		dest="reprocess",
		action="store_true",
		help="reprocess runs. unless this flag is given, runs with already-existing .npz files will be skipped."
	)

	# quiet
	parser.add_argument("-q",action='store_true',dest="quiet",help="quiet mode. if given, results per run will not be printed.")

	# old pulse file format
	parser.add_argument("-p",action='store_true',dest="pulse_fmt_old",help="use old pulse file format")

	return parser


def main(args):

	# compose list of runs to be processed, not yet checking for 
	# the existence of associated .npz files
	runs = get_runs_to_process(args)

	# compose input and output file path templates
	fi_full_template = os.sep.join([args.indir , args.infile ])
	fo_full_template = os.sep.join([args.outdir, args.outfile])

	for ir,run in enumerate(runs):

		# compose input and output file paths for this run
		fi = fi_full_template.format(run)
		fo = fo_full_template.format(run)

		# skip run if input file missing
		if not os.path.exists(fi):
			if not args.quiet:
				print("run {} skipped; input file missing".format(run))
			continue

		# skip if input file empty
		if not os.path.getsize(fi):
			if not args.quiet:
				print("run {} skipped; input file empty".format(run))
			continue

		# skip run if output file already exists, unless -r flag
		if (not args.reprocess) and os.path.exists(fo):
			if not args.quiet:
				print("run {} skipped; processed file already exists".format(run))
			continue

		# try to process the run
		try:
			if args.pulse_fmt_old:
				ant.convert_pulses(fi,fo)
			else:
				ant.convert_dirpipulses(fi,fo)
			if not args.quiet:
				print("run {} success".format(run))
		except Exception as e:
			print("run {} failed: {}".format(run,e))


def get_runs_to_process(args):
	incl = set()
	for t in args.runs:
		incl |= runs_from_token(t)	
	for t in args.exclude:
		incl -= runs_from_token(t)
	return sorted(incl)

def runs_from_token(token):
	if "-" in token:
		start,stop = token.split('-')
		return set(range(int(start),int(stop)+1))
	return {int(token)}


if __name__ == "__main__":
	parser = compose_parser()
	args = parser.parse_args()
	main(args)
