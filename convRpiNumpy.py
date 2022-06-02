"""
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"


import os
import sys
import time
import ROOT
import uproot

import awkward
import numpy as np
import matplotlib.pyplot as plt



RPI_DATA_DIR = "/home/bode/Documents/GitHub/xrd-analysis-refactor/data/rpi"
RPI_DATA_FILENAME = "Run{}.txt"
RPI_DATA_FILE = os.sep.join([RPI_DATA_DIR, RPI_DATA_FILENAME])
RPI_RUNS = [399]
RPI_DATA_FILES = [RPI_DATA_FILE.format(_) for _ in RPI_RUNS]


INDEX_MIN = 1024
INDEX_MAX = 1030
def passes_primary(index):
	return (index >= INDEX_MIN) and (index <= INDEX_MAX)
def passes_all(index):
	return True

def main(file_in, file_out, one_per_event = True):
	
	primary_pulses = {1:[], 2:[]}
	n_events = {1:0, 2:0}

	passes = passes_primary if one_per_event else passes_all

	with open(file_in, "r") as file:


		line = file.readline()
		iline = 0
		
		while line:

			# if (iline%1000) == 0:
			# print("")
			# print(iline, line)
			command, _, arguments = line.partition(':')


			if command.startswith("AREA"):

				channel = int(command[4:])
				n_events[channel] += 1
				
				primary_values = []
				if arguments.strip():
					args = list(map(float, arguments.strip().split(" ")))
					values  = args[0::2]
					indices = args[1::2]
					for iind, ind in enumerate(indices):
						if passes(ind):
							primary_values.append(values[iind])
							if one_per_event:
								break
				
				if primary_values:
					primary_pulses[channel]+=primary_values
				elif one_per_event:
					primary_pulses[channel].append(0)

			line = file.readline().strip()
			iline += 1

	print(iline)
	print(n_events)
	print([len(_) for _ in primary_pulses.values()])

	nbins = 100
	max_max = max([np.max(_) for _ in primary_pulses.values()])
	min_min = min([np.min(_) for _ in primary_pulses.values()])
	x_edges = np.linspace(min_min, max_max, nbins+1)

	plt.hist(primary_pulses[1], bins=x_edges, histtype='step', label="channel 1", )
	plt.hist(primary_pulses[2], bins=x_edges, histtype='step', label="channel 2", )
	plt.legend()
	plt.title('{}'.format(file_in.rpartition(os.sep)[2]))
	plt.xlabel("area in pVs?")
	plt.ylabel("counts")
	plt.yscale('log')
	plt.show()


	# np.savez_compressed(file_out, **{"area_{}".format(k):v for k,v in primary_pulses.items()})
	np.savez(file_out, **{"area_{}".format(k):v for k,v in primary_pulses.items()})







if __name__ == "__main__":

	for file_in in RPI_DATA_FILES:
		
		file_out = file_in.rpartition(".")[0] + ".npz"

		main(file_in, file_out)
