"""
script used for testing features during development
don't actually use this for anything else
"""

import numpy
import random
import math
import numpy as np

import matplotlib.pyplot as plt

import utils.fileio as fileio
import utils.data   as data
import utils.model  as model



def main():

	run = 4041
	test_file = '../xrd-analysis/data/root/scintillator/Run{}.root'.format(run)

	branches_use = {"timestamp_3046_1","area_3046_1","vMax_3046_1","tMax_3046_1","scaler_3046_1"}

	branches = fileio.load_branches(test_file, {"timestamp_3046_1","area_3046_1","vMax_3046_1","tMax_3046_1","scaler_3046_1","area_3046_1","tMax_3046_2","scaler_3046_2"})
	bm = data.BranchManager(branches, export_copies=False, import_copies=False, )

	bm.bud([data.bud_entry])
	bm.bud([data.fix_monotonic_timestamp()],overwrite=True)
	bm.bud([data.subtract_first("timestamp_3046_1")],overwrite=True)
	bm.bud([data.differentiate_branch("timestamp_3046_1")])
	bm.bud([data.count_passing(data.cut("timestamp_3046_1_deriv",10),"burst_index")])
	
	t  = bm["timestamp_3046_1"]
	dt = bm["timestamp_3046_1_deriv"]
	bi = bm["burst_index"]

	n_bursts = bi.max() + 1

	burst_nev  = []
	burst_i    = []
	burst_fit  = []
	
	# 4039
	# peak_model = model.constant([[0,np.inf]]) + model.gaus([[0,np.inf],[35000,37800],[4000,5000]]) + model.gaus([[0,np.inf],[39000,43000],[2500,3500]])
	# bins = np.linspace(28000,50000,50)
	peak_model = model.constant([[0,np.inf]]) + model.gaus([[0,np.inf],[39000,43000],[2500,3500]])
	bins = np.linspace(38000,45000,100)

	# # 4029
	# peak_model = model.constant([[0,np.inf]]) + model.gaus([[0,np.inf],[20000,23000],[1800,2400]])
	# bins = np.linspace(17500,25500,100)

	for i in range(n_bursts):

		mask = data.cut("burst_index", i - 0.1, i + 0.1)
		masked_branches = bm.mask(mask, branches_use)

		this_t    = masked_branches["timestamp_3046_1"]
		this_area = masked_branches["area_3046_1"]
		this_vmax = masked_branches["vMax_3046_1"]
		this_tmax = masked_branches["tMax_3046_1"]

		this_nev = this_t.size
		
		# 4039
		# p0 = [0.0, 2079 * this_nev / 1000000, 36391.0, 4517.0, 17688.0 * this_nev / 1000000, 40846.0, 2972.0]
		p0 = [0.0, 6067 * this_nev / 1000000, 40846.0, 2972.0]

		# # 4029
		# p0 = [0.0, 2859 * this_nev / 600000, 21403.0, 2177.0]

		counts, edges = np.histogram(this_area, bins=bins)
		midpoints = (edges[1:] + edges[:-1])*0.5

		try:
			popt, pcov, chi2, ndof = peak_model.fit(midpoints, counts, p0=p0)

			if not (i%50):
				plt.step(midpoints, counts, where='mid', color="k", label="data")
				plt.plot(midpoints, peak_model(midpoints, *popt), 'g-', label="best fit")
				plt.title("run {}, burst {}\nchi2/ndof={:.2f}/{}={:.2f}".format(run,i,chi2,ndof,chi2/ndof))
				plt.show()

			burst_nev.append(this_t.size)
			burst_i.append(i)
			burst_fit.append([popt, pcov, chi2, ndof])
			print("{:<3} - {} - {} - {} - {} - {}".format(i, this_t.size, list(popt), list(pcov), chi2, ndof))

		except:
			print("{} - {} - fit failed".format(i, this_nev))

	burst_i   = np.array(burst_i  )
	burst_nev = np.array(burst_nev)

	plt.plot(burst_i, burst_nev, 'k.', )
	plt.xlabel('burst index')
	plt.ylabel('number of events')
	plt.title('number of events per burst\nRun {}'.format(run))
	plt.show()

	burst_popt = np.stack([_[0] for _ in burst_fit],axis=0)
	burst_pcov = np.stack([_[1] for _ in burst_fit],axis=0)

	# poi = 5
	poi = 2
	pm_const = model.constant()
	pm_popt, pm_pcov, pm_chi2, pm_ndof = pm_const.fit(burst_i, burst_popt[:,poi], yerr=burst_pcov[:,poi])

	colors = ['k','g','b','darkred','tab:brown','magenta','c','r']
	for j in range(peak_model.npars):
		if j!=poi:
			continue
		plt.errorbar(burst_i, burst_popt[:,j], burst_pcov[:,j], color=colors[j], ls='', marker='.', label=peak_model.pnames[j])
		plt.plot(burst_i, pm_const(burst_i, *pm_popt), 'k-', label="fit")
	plt.xlabel('burst index')
	plt.ylabel('parameter values')
	plt.title("fit parameter values per burst, run {}\nfit chi2/dof = {:.2f}/{} = {:.2f}".format(run, pm_chi2, pm_ndof, pm_chi2/pm_ndof))
	plt.legend()
	plt.show()



	# print(t[:10])
	# print(dt[:10])
	# print(bi[:10])

	# plt.plot(bm["timestamp_3046_1"][1:], bm["timestamp_3046_1_deriv"][1:], 'k-', label="dt")
	# plt.plot(bm["timestamp_3046_1"][1:], bm["burst_index"][1:], 'r.', label="batch index")
	# plt.xlabel("timestamp")
	# plt.legend()
	# plt.show()

	# plt.plot(bm["entry"], bm["timestamp_3046_1"], 'k-')
	# plt.plot(bm["entry"], bm["burst_index"], 'r.')
	# plt.xlabel('entry')
	# plt.show()


if __name__ == '__main__':
	main()
