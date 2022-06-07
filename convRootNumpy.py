"""
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"


import os
import sys
import time
import ROOT

# import awkward
import numpy as np
import matplotlib.pyplot as plt




def timed(fn, *args, **kwargs):
	t1 = time.time()
	ans = fn(*args, **kwargs)
	t2 = time.time()
	return (t2-t1), ans

def pprint(stuff):
	for _ in stuff:
		print(_)

# def print_columns(frame):
# 	names = frame.GetColumnNames()
# 	# longest_name = max([len(_) for _ in names])
# 	pprint(["{:<36} - {:<}".format(str(_),frame.GetColumnType(_)) for _ in names])

def col2key(col):
	# return col.replace('->','.')
	return col




ROOT_DIR = '/home/bode/root/root_v6.26.00.Linux-fedora34-x86_64-gcc11.2/root/bin'
BUILD_DIR = '/home/bode/geant/libraries/beamRadSimXRay/build'
LIB_BEAM = os.sep.join([BUILD_DIR, 'libBeamRadCore.so'])
def trawl(tree, columns, filters=""):

	# make sure all data are retrievable
	# tree.SetEstimate(tree.GetEntries())
	tree.SetEstimate(tree.GetEntries())

	# cast filters to list if single string supplied
	if type(filters) is str:
		filters = [filters]

	# cast columns to dict
	return_flat = False
	if type(columns) is str:
		columns = [columns]
		return_flat = True
	# if type(columns) in (list, tuple):
	# 	columns = {col2key(_):_ for _ in columns}

	ncols = len(columns)
	# keys = list(columns.keys())
	print("there are {} columns".format(len(columns)))

	# draw_var = ":".join([columns[_] for _ in keys])
	draw_var = ":".join(columns)
	draw_sel = "&&".join(filters)
	draw_opt = "goff"
	print("draw varexp is {}".format(draw_var))
	print("draw selection is {}".format(draw_sel))
	print("draw option is {}".format(draw_opt))
	# print("")

	tdraw, ans = timed(tree.Draw, draw_var, draw_sel, draw_opt)
	print("draw command finished in {:.3f} seconds".format(tdraw))
	print("return {} as answer".format(ans))
	# print("")

	# arrays = {}
	arrays=[]
	# for i,key in enumerate(keys):
	for i,key in enumerate(columns):
		t, arr = timed(np.frombuffer, tree.GetVal(i), count=ans)
		print("extracting array {} ({}; {} {}) from buffer took {:.3f} ns".format(i, key, arr[0], arr[1], t*1e6))
		# arrays[key] = arr
		arrays.append(arr)
	# print("")

	if return_flat:
		# return arrays[keys[0]]
		return arrays[0]
	else:
		return arrays




# default tree
TREE_NAME = "Events"

# default list of top level leaves to save
# the rest are empty >:(
LEAVES_EVENT = [
	# "ROOTEvent",
	# "fUniqueID",
	# "fBits",
	"eventID",
	# "GammaTracks",
	"NbOfGammaTracks",
	# "NeutronTracks",
	# "NbOfNeutronTracks",
	# "PhotonTracks",
	# "NbOfPhotonTracks",
	# "MuonTracks",
	# "NbOfMuonTracks",
	# "PMTHits",
	# "NbOfPMTHits",
	# "ScintRHits",
	# "NbOfScintRHits",
	# "nbOfCerenkovPhotons",
	# "nbOfScintillationPhotons",
	# "absorptionCount",
	# "boundaryAbsorptionCount",
	# "totalNbOfPEPMT",
	# "pmtsAboveTrigger",
	# "totalEDepInCrytals",
	# "totalNREDepInCrytals",
	# "energyEnterScinti_MeV",
	# "energyExitScinti_MeV",
	"Edep_MeV_Si1",
	"Edep_MeV_Si2",
	"Edep_MeV_Si3",
	"Edep_MeV_Si4",
	# "Edep_MeV_Si5",
	# "Edep_MeV_Abs1",
	# "Edep_MeV_Abs2",
	# "Edep_MeV_Abs3",
	# "Edep_MeV_Abs4",
	# "Edep_MeV_ScintVeto",
	# "barHit",
	# "gammaOutScintillator",
	# "muonTrig",
	# "scintToPMT",
]

# default list of gamma track leaves
LEAVES_GAMMA = [
	"GammaTracks->trackID",
	"GammaTracks->initialTime_ns",
	"GammaTracks->finalTime_ns",
	"GammaTracks->initialEnergy_MeV",
	"GammaTracks->finalEnergy_MeV",
	"GammaTracks->energyDeposit_MeV",
	"GammaTracks->totalEnergy_MeV",
	"GammaTracks->parentID",
	"GammaTracks->initialPositionX_m",
	"GammaTracks->finalPositionX_m",
	"GammaTracks->initialPositionY_m",
	"GammaTracks->finalPositionY_m",
	"GammaTracks->initialPositionZ_m",
	"GammaTracks->finalPositionZ_m",
	"GammaTracks->totalTrackLength_m",
	"GammaTracks->gammaOutScint",
	"GammaTracks->fUniqueID",
	"GammaTracks->fBits",
]

def root_extract(file_in, file_out, leaves_event=None, leaves_gamma=None, ):
	
	# if tree_name is None:
	tree_name = TREE_NAME

	print("\nsetting up root environment...")
	os.chdir(BUILD_DIR)
	print("cwd: {}".format(os.getcwd()))
	ROOT.gSystem.Load("libPhysics")
	ROOT.gSystem.Load(LIB_BEAM)
	print("loaded stuff")

	print("\nloading tree {} from root file {}".format(tree_name, file_in))
	file = ROOT.TFile.Open(file_in)
	tree = file.Get(tree_name)


	if leaves_gamma is None:
		leaves_gamma = LEAVES_GAMMA


	if leaves_event is None:
		leaves_event = LEAVES_EVENT
	elif leaves_event is all:
		leaves_event = [_.GetName() for _ in tree.GetListOfLeaves()]
		if leaves_gamma:
			raise ValueError("Should not get all event-level leaves when sub-leaves exist")
		# for leaf in leaves_event:
		# 	print(leaf.GetName())
		# print(dir(leaf))
		# sys.exit(0)




	print("\nextracting data from event level leaves")
	arrays_event = trawl(tree, leaves_event, "")

	print("creating activity filter")
	ftr_deposit = np.any(np.stack([arrays_event[_] for _ in [2,3,4,5]], axis=0) > 0, axis=0)
	print("{} / {} have any activity".format(ftr_deposit.sum(), ftr_deposit.size))
	
	print("\nfiltering event level arrays...")
	arrays_save = {}
	for ib,b in enumerate(leaves_event):
		print("{}/{}".format(ib+1,len(leaves_event)))
		arrays_save[b] = arrays_event[ib][ftr_deposit]

	if leaves_gamma:
		print("\nextracting data from GammaTracks leaves")
		arrays_gamma = trawl(tree,leaves_gamma,"GammaTracks->parentID == 0")

	print("\nfiltering GammaTracks arrays...")
	for ib,b in enumerate(leaves_gamma):
		print("{}/{}".format(ib+1,len(leaves_gamma)))
		arrays_save[b.replace('->','.')] = arrays_gamma[ib][ftr_deposit]

	print("\nsaving filtered arrays...")
	t,ans = timed(np.savez, file_out, **arrays_save)
	print("saving arrays took {:.3f} ms".format(t*1000))







def main_pyroot(get_flat=None, get_ft=None, ):

	os.chdir(BUILD_DIR)
	print("cwd: {}".format(os.getcwd()))
	ROOT.gSystem.Load("libPhysics")
	ROOT.gSystem.Load(LIB_BEAM)
	# ROOT.gInterpreter.GenerateDictionary('/home/bode/geant/libraries/beamRadSimXRay/include/brROOTEvent.hh')
	# ROOT.gInterpreter.GenerateDictionary('/home/bode/geant/libraries/beamRadSimXRay/include/brGammaTrack.hh')
	# os.chdir(ROOT_DIR)
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


	# test trawl
	if True:

		file = ROOT.TFile.Open(TEST_FILE)
		tree = file.Get("Events")
		# # pprint(dir(tree))
		# # print("")
		# pprint(tree.GetListOfLeaves())
		# sys.exit(0)



		if get_flat is None:
			# the rest are empty >:(
			get_flat = [
				# "ROOTEvent",
				# "fUniqueID",
				# "fBits",
				"eventID",
				# "GammaTracks",
				"NbOfGammaTracks",
				# "NeutronTracks",
				# "NbOfNeutronTracks",
				# "PhotonTracks",
				# "NbOfPhotonTracks",
				# "MuonTracks",
				# "NbOfMuonTracks",
				# "PMTHits",
				# "NbOfPMTHits",
				# "ScintRHits",
				# "NbOfScintRHits",
				# "nbOfCerenkovPhotons",
				# "nbOfScintillationPhotons",
				# "absorptionCount",
				# "boundaryAbsorptionCount",
				# "totalNbOfPEPMT",
				# "pmtsAboveTrigger",
				# "totalEDepInCrytals",
				# "totalNREDepInCrytals",
				# "energyEnterScinti_MeV",
				# "energyExitScinti_MeV",
				"Edep_MeV_Si1",
				"Edep_MeV_Si2",
				"Edep_MeV_Si3",
				"Edep_MeV_Si4",
				# "Edep_MeV_Si5",
				# "Edep_MeV_Abs1",
				# "Edep_MeV_Abs2",
				# "Edep_MeV_Abs3",
				# "Edep_MeV_Abs4",
				# "Edep_MeV_ScintVeto",
				# "barHit",
				# "gammaOutScintillator",
				# "muonTrig",
				# "scintToPMT",
			]
		arrays_flat = trawl(tree, get_flat, "")
		ftr_deposit = np.any(np.stack([arrays_flat[_] for _ in [2,3,4,5]], axis=0) > 0, axis=0)
		print("deposit filter: {} / {} have any activity".format(ftr_deposit.sum(), ftr_deposit.size))
		print("filtering flat arrays...")
		arrays_save = {}
		for ib,b in enumerate(get_flat):
			print("{}/{} flat".format(ib+1,len(get_flat)))
			arrays_save[b] = arrays_flat[ib][ftr_deposit]


		if get_gt is None:
			get_gt = [
				"GammaTracks->trackID",
				"GammaTracks->initialTime_ns",
				"GammaTracks->finalTime_ns",
				"GammaTracks->initialEnergy_MeV",
				"GammaTracks->finalEnergy_MeV",
				"GammaTracks->energyDeposit_MeV",
				"GammaTracks->totalEnergy_MeV",
				"GammaTracks->parentID",
				"GammaTracks->initialPositionX_m",
				"GammaTracks->finalPositionX_m",
				"GammaTracks->initialPositionY_m",
				"GammaTracks->finalPositionY_m",
				"GammaTracks->initialPositionZ_m",
				"GammaTracks->finalPositionZ_m",
				"GammaTracks->totalTrackLength_m",
				"GammaTracks->gammaOutScint",
				"GammaTracks->fUniqueID",
				"GammaTracks->fBits",			
			]
		arrays_gt = trawl(tree,get_gt,"GammaTracks->parentID == 0")

		print('| '.join(['{:<12}'.format(_) for _ in get_gt]))
		for i in range(16):
			print('| '.join(['{:<12.3e}'.format(_[i]) for _ in arrays_gt]))

		for k,v in enumerate(arrays_gt):
			print("{} {} - {}".format(k,get_gt[k],v.shape))


		print("\nfiltering gt arrays...")
		for ib,b in enumerate(get_gt):
			print("{}/{} gt".format(ib+1,len(get_gt)))
			arrays_save[b.replace('->','.')] = arrays_gt[ib][ftr_deposit]

		print("\nsaving filtered arrays...")
		t,ans = timed(np.savez, "/home/bode/Documents/GitHub/xrd-analysis-refactor/Co57.npz", **arrays_save)
		print("saving arrays took {:.3f} ms".format(t*1000))



	# tree interface
	if False:
		file = ROOT.TFile.Open(TEST_FILE)
		tree = file.Get("Events")

		tree.SetEstimate(tree.GetEntries())

		# draw_col = "GammaTracks->initialEnergy_MeV"
		# draw_ftr = "GammaTracks->parentID==0"
		# draw_opt = ""

		n_col = 1
		draw_col = "GammaTracks->initialEnergy_MeV:GammaTracks->finalEnergy_MeV"
		# draw_col = "GammaTracks->initialEnergy_MeV - GammaTracks->finalEnergy_MeV"
		draw_ftr = "GammaTracks->parentID==0"# && GammaTracks->totalEnergy_MeV>0.1"
		draw_opt = "goff"#"candle"#"para"


		# tree.Draw("GammaTracks->initialEnergy_MeV>>h(100,0,0.55)","GammaTracks->parentID==0","")
		tdraw, ans = timed(tree.Draw,draw_col,draw_ftr,draw_opt)
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

		print("array has shape {}".format(arr.shape))

		plt.hist(arr, bins=100)
		plt.show()

		
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



def test_use():

	arr_src = {}
	for src in SOURCES:
		tload, arr = timed(np.load,"./data/sim/{}.npz".format(src))
		print("loading data for {} took {:.3f} ns".format(src, tload*1e6))
		arr_src[src] = arr


	pve_leaf = "GammaTracks.initialEnergy_MeV"
	pve_filters = {}
	for src in SOURCES:
		print("\nsrc {}".format(src))
		pve_filters[src] = []
		for lo,hi in PVE[src]:
			this_pve = arr_src[src][pve_leaf] * 1000
			this_ftr = np.logical_and(this_pve > lo, this_pve < hi)
			pve_filters[src].append(this_ftr)
			print("\tfilter {}/{}".format(this_ftr.sum(), this_ftr.size))
	print("")

	# keys = list(arr.keys())
	keys = ["GammaTracks.initialEnergy_MeV"]
	# keys = ["Edep_MeV_Si1"]
	for br in keys:
		print(br)

		fig,ax = plt.subplots(2,3,sharex=False,sharey=False)
		fig.subplots_adjust(
		    top=0.94,
		    bottom=0.06,
		    left=0.06,
		    right=0.94,
		    hspace=0.2,
		    wspace=0.2,
		)
		plt.suptitle(br)
		fig.set_size_inches(15,10)
		fig.set_dpi(120)

		for i,src in enumerate(SOURCES):

			plt.subplot(2,3,i+1)
			# plt.subplot(1+(i//3),1+(i%3),1)
			# this_ax = ax[i+1]

			this_leaf = arr_src[src][br]
			if not np.any(this_leaf>0):
				print(src, "all zeros")
				continue
			print(src)

			blo = this_leaf[this_leaf>0].min()
			bhi = this_leaf.max()
			nbins = 1000
			this_bins = np.linspace(blo,bhi,nbins + 1)

			plt.hist(this_leaf, bins=this_bins, histtype='stepfilled', label='total', color=(0,0,0,0.1), edgecolor=(0,0,0,0.25))
			for iftr,ftr in enumerate(pve_filters[src]):
				plt.hist(this_leaf[ftr], bins=this_bins, histtype='step', label='{:.0f} KeV'.format(0.5*sum(PVE[src][iftr])))

			plt.xlabel(br)
			plt.title(src)
			plt.legend()
			plt.yscale('log')
		
		plt.savefig('./figs/2022-05-18 pve/{}.png'.format(br))
		# plt.clf()
		plt.show()




SIM_DIR = '/home/bode/Documents/GitHub/xrd-analysis/data/root/simulation'
SIM_OUT_DIR = '/home/bode/Documents/GitHub/xrd-analysis-refactor/data/sim'

EXP_DIR = '/home/bode/Documents/GitHub/xrd-analysis/data/root/scintillator'
EXP_OUT_DIR = '/home/bode/Documents/GitHub/xrd-analysis-refactor/data/exp'

SOURCES = [
	"Am241",
	"Cd109",
	"Ba133",
	"Co57",
	# "Cs137",
	"Mn54",
	"Na22",
]

PVE = {
	"Am241": [[ 13, 15], [ 17, 19], [ 59, 60]           ] ,
	"Cd109": [[ 21, 23], [ 24, 26], [ 86, 90]           ] ,
	"Ba133": [[ 30, 32], [ 80, 82], [302,304], [355,357]] ,
	"Co57" : [[ 13, 15], [121,123], [135,137]           ] ,
	"Mn54" : [[834,836]                                 ] ,
	"Na22" : [[510,512]                                 ] ,
}



if __name__ == '__main__':
	print("cwd: {}".format(os.getcwd()))
	
	if True:

		# sources = SOURCES
		# sources = ["Am241"]
		# sources = ["Ba133","Cd109"]
		sources = ["Co57"]
		files_per_source = {
			"Am241":["Am241x10.root"],
			"Ba133":["Ba133x10.root", "Ba133x10_with53mult.root", "Ba133x10_with53.root"],
			"Cd109":["Cd109x100.root", "Cd109x25.root", "Cd109x25_2.root", "Cd109x25_3.root", "Cd109x25_4.root"],
			"Co57" :["Co57x10.root"],
		}

		# sim_dir = SIM_DIR
		sim_dir = '/home/bode/Documents/GitHub/xrd-analysis/data/root/simulation2'

		sim_out_dir = SIM_OUT_DIR

		# sim data
		for src in sources:
			print("\nsource {}".format(src))

			files = files_per_source.get(src, "{}.root".format(src))
			if type(files) is str:
				files = [files]

			for file in files:
				print("file {}".format(file))
				
				file_in  = os.sep.join([sim_dir    , file])
				file_out = os.sep.join([sim_out_dir, file.replace(".root",".npz")])

				if os.path.exists(file_out):
					print("\toutput already exists for input file {}".format(file))
					print("\tdelete ore move existing file if you want to re-process")
					continue
				
				# file_out = os.sep.join([SIM_OUT_DIR, "{}.npz".format(src)])
				print('\tExtracting data, sim data for src {}'.format(src))
				print('\toutput file is {}'.format(file))

				root_extract(file_in, file_out)

		# # experimental  data
		# for run in [4291, 4293, 4292, 4294, 4225, 4226]:
		# 	print('\n\n\nExtracting data, experimental data for run {}\n'.format(run))
		# 	file_in  = os.sep.join([EXP_DIR    , "Run{}.root".format(run)])
		# 	file_out = os.sep.join([EXP_OUT_DIR, "Run{}.npz".format(run)])
		# 	root_extract(file_in, file_out, leaves_event=all, leaves_gamma=[])

	if False:
		main_pyroot()

	if False:
		test_use()
