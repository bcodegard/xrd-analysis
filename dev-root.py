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


SIM_DIR = '/home/bode/Documents/GitHub/xrd-analysis/data/root/simulation' 
TEST_FILENAME = "Co57.root"
TEST_FILE = os.sep.join([SIM_DIR, TEST_FILENAME])

ROOT_DIR = '/home/bode/root/root_v6.26.00.Linux-fedora34-x86_64-gcc11.2/root/bin'
BUILD_DIR = '/home/bode/geant/libraries/beamRadSimXRay/build'

LIB_BEAM = os.sep.join([BUILD_DIR, 'libBeamRadCore.so'])


def timed(fn, *args, **kwargs):
	t1 = time.time()
	ans = fn(*args, **kwargs)
	t2 = time.time()
	return (t2-t1), ans

def pprint(stuff):
	for _ in stuff:
		print(_)

def print_columns(frame):
	names = frame.GetColumnNames()
	# longest_name = max([len(_) for _ in names])
	pprint(["{:<36} - {:<}".format(str(_),frame.GetColumnType(_)) for _ in names])

def main_pyroot():

	os.chdir(BUILD_DIR)
	print("cwd: {}".format(os.getcwd()))
	ROOT.gSystem.Load("libPhysics")
	ROOT.gSystem.Load(LIB_BEAM)
	# ROOT.gInterpreter.GenerateDictionary('/home/bode/geant/libraries/beamRadSimXRay/include/brROOTEvent.hh')
	# ROOT.gInterpreter.GenerateDictionary('/home/bode/geant/libraries/beamRadSimXRay/include/brGammaTrack.hh')
	print("loaded stuff")
	print("")


	if False:
		frame = ROOT.RDataFrame("Events",'/home/bode/Documents/GitHub/xrd-analysis/data/root/simulation/Am241.root')
		# pprint(frame.GetColumnNames())
		# print("")
		# pprint(dir(frame))
		# print("")



		# f2 = frame.Define("pve","GammaTracks.initialEnergy_MeV[GammaTracks.trackID < 1]")
		# f2 = frame.Define("pve","GammaTracks.initialEnergy_MeV")
		# f2 = frame.Define("pve", "return Map(GammaTracks, [](const auto &vec) { return vec->trackID(); })")

		# "auto lambda0 = [](ROOT::VecOps::RVec<brGammaTrack*>& var0){return var0.at(0)->initialEnergy_MeV"
		# f2 = frame.Define("pve","")

		f1 = frame.Filter("Edep_MeV_Si1>0")
		# f2 = f1.Define(
		# 	"gt0","GammaTracks.at(0)").Define(
		# 	"gt0e","gt0->at(4)")
		# print_columns(f2)

		f2 = f1.Define("pve","Edep_MeV_Si1 * 1000.0")
		f2 = f1.Define("pve","GammaTracks.at(0)->trackID")
		hist = f2.Histo1D("pve")
		hist.Draw()
		input()


		sys.exit(0)

		# n = 100000
		# few = frame.Filter("eventID < {}".format(n))
		# t,ids = timed(few.AsNumpy, ["eventID"])
		# print("fetched {} ids in {:>6.3f} ms".format(n,t*1000))

		frame_filt = frame.Filter("Edep_MeV_Si1>0")
		# hist = frame_filt.Histo1D("Edep_MeV_Si1")
		# hist = frame_filt.Histo1D("NbOfGammaTracks")
		hist = frame_filt.Histo1D("GammaTracks.parentID")

		# frame_filt = frame.Filter("GammaTracks->initialEnergy_MeV>>h(100,0,0.55")
		# hist = frame_filt.Histo1D("GammaTracks->parentID==0")
		hist.Draw()
		input()


	# tree interface
	if True:
		file = ROOT.TFile.Open(TEST_FILE)
		tree = file.Get("Events")

		# tree.Draw("GammaTracks->initialEnergy_MeV>>h(100,0,0.55)","GammaTracks->parentID==0","")
		tdraw, ans = timed(
			tree.Draw,"GammaTracks->initialEnergy_MeV","GammaTracks->parentID==0",""
		)
		# ans = tree.Draw("Edep_MeV_Si1","Edep_MeV_Si1>0","")
		print(ans)
		
		tget, v1 = timed(tree.GetV1)

		tbuff, arr = timed(np.frombuffer, v1, count=ans)

		print("took {:>.3f} seconds to Draw".format(tdraw))
		print("took {:>.3f} seconds to GetV1".format(tget))
		print("took {:>.3f} seconds to frombuffer".format(tbuff))

		# for k in range(100):
		# 	print(v1[k])
		input()

		
		# eventID = np.zeros(1)
		# tree.SetBranchAddress("eventID", eventID)
		# ent = tree.GetEntry(1484)
		# print(ent)
		# print(eventID)

		# for entry in tree:
		# 	print("\nentry")
		# 	print(entry)
		# 	print(entry.ROOTEvent)
		# 	print(entry.Edep_MeV_Si1)
		# 	break
		# print(tree)
		# print(tree.Dump())
		# # pprint(tree.GetListOfBranches())
		# # pprint(dir(tree))
		sys.exit(0)


	



def main_uproot():

	os.chdir(BUILD_DIR)
	print("cwd: {}".format(os.getcwd()))
	ROOT.gSystem.Load("libPhysics")
	ROOT.gSystem.Load(LIB_BEAM)
	print("loaded stuff")
	print("")

	root_file = uproot.open(TEST_FILE)
	
	tree = root_file['Events;24']
	print("tree has keys:")
	print(tree.keys())
	print("")
	# for _ in dir(tree):
	# 	print(_)
	# print("")

	
	print(tree.arrays("eventID"))
	sys.exit(0)


	test = root_file['Events;24/ROOTEvent/GammaTracks']
	print("")
	print(test)
	for _ in dir(test):print(_)





	# # raises error CannotBeAwkward: arbitray pointer
	if False:
		tracks = tree.arrays("GammaTracks","eventID<10")

	# This starts running with no errors, but is increadibly slow
	# I think it tries to load 100% of the dataset instead of just
	# loading the first 10 events.
	# 
	# the library="np" tag makes the previous error not occur, but
	# makes the process incredibly slow.
	if False:
		tracks = tree.arrays("GammaTracks","eventID<10",library="np")



	# iterator with cut on nonzero energy deposit in channel 1
	if False:
		iterator = uproot.iterate(tree, ["ROOTEvent/eventID"], cut="ROOTEvent/Edep_MeV_Si1>0", library="np")
		# iterator = tree.iterate(step_size = 10, cut="ROOTEvent/Edep_MeV_Si1>0", library="np")

		print("")
		print("iterating through sets of 10 events until non-empty response")
		batch = []
		tried = 0
		while not batch:
			tried+=1
			t, batch = timed(next, iterator)
			print("{:>4} fetching 10 events took {:>8.3f} ms".format(tried,t*1000))
		print("found {} events in the last batch".format(len(batch)))


	if False:
		print(type(tree))
		lazy = uproot.lazy(
			# {TEST_FILE},
			tree,
			# ["ROOTEvent/GammaTracks"],
			["ROOTEvent/eventID"],
			library="ak",
			recursive=True,
			full_path=True
		)
		print(lazy)








if __name__ == '__main__':
	print("cwd: {}".format(os.getcwd()))
	# main_uproot()
	main_pyroot()
