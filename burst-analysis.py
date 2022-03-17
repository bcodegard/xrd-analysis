"""
script used for testing features during development
don't actually use this for anything else
"""

import argparse
import sys
import os

import random
import math
import numpy as np

import matplotlib.pyplot as plt

import utils.fileio  as fileio
import utils.data    as data
import utils.model   as model
import utils.display as display




ROOT_FILE_DEFAULT = '../xrd-analysis/data/root/scintillator/Run{}.root'
FIG_LOC = "./figs/{}.png"



def procure_cluster_data(args, extra_branches=set()):
	""""""

	# extra_branches -> set
	if not (type(extra_branches) is set):
		extra_branches = set(extra_branches)

	# run -> path to file
	if os.sep in args.run:
		root_file = args.run
	else:
		root_file = ROOT_FILE_DEFAULT.format(args.run)
	assert os.path.exists(root_file)

	# compose list of needed branches
	# get all keys present in root file
	branches_all = fileio.get_keys(root_file)
	branches_use = extra_branches
	# cluster branch
	cluster_branch = next(_ for _ in sorted(branches_all) if _.startswith(args.cluster_branch))
	branches_use |= {cluster_branch}
	# fit
	branches_use |= {args.fit}
	# cuts
	cut_branches = {_.partition(',')[0] for _ in args.cut}
	branches_use |= cut_branches
	
	# load branches and create manager instance
	branches = fileio.load_branches(root_file, branches_use)
	bm = data.BranchManager(branches, export_copies=False, import_copies=False)

	# apply cuts
	# if args.cut:
	cuts = [data.cut(*data.split_with_defaults(_,[None,-np.inf,np.inf],[str,float,float])) for _ in args.cut]
	cuts.append(data.cut(args.fit, args.fit_lo, args.fit_hi))
	bm.mask(data.mask_all(*cuts),apply_mask=True)

	# fixes and tweaks
	bm.bud([data.bud_entry])
	bm.bud([data.fix_monotonic_timestamp()],overwrite=True)
	bm.bud([data.localize_timestamp()],overwrite=True)

	# differentiate if requested
	if args.cluster_diff:
		if args.cluster_diff > 1:
			print("WARNING: differentiation with kernel size > 1 not yet implemented")
		bm.bud([data.differentiate_branch(cluster_branch,suffix="deriv")])
		cluster_branch_final = '_'.join([cluster_branch,"deriv"])
	else:
		cluster_branch_final = cluster_branch

	# make cluster index branch
	bm.bud([data.count_passing(data.cut(cluster_branch_final,args.cluster_threshold),"cluster_index")])

	return bm





def analyze_fourier(args):


	branch_use = "vMax_3046_1"
	branches_use = {branch_use, "timestamp_3046_1"}
	bm = procure_cluster_data(args, branches_use)

	bcut = bm.mask(data.cut(branch_use,10,900), branch_use)

	# bcut = np.random.normal(np.linspace(500,505,bcut.size), 50, bcut.shape)

	# plt.hist(bcut, bins=np.linspace(0,1000,500))
	# plt.show()

	dur = bm["timestamp_3046_1"][-1] - bm["timestamp_3046_1"][0]
	sep = dur / bcut.size

	print(dur, sep)

	freq = np.fft.rfftfreq(bcut.size, sep)
	bfft = np.fft.rfft(bcut)

	# print(freq[:10])
	# print(freq[-10:])

	# plt.plot(1/(freq[1:]), abs(bfft[1:]), 'k.')
	# plt.xlabel("1/frequency")
	# plt.ylabel("amplitude")
	# plt.xscale('log')
	# plt.yscale('log')
	# plt.show()

	T = 1/(freq[1:])
	A = abs(bfft[1:])
	print(T[:10])
	print(A[:10])
	nbins = 200
	display.pairs2d(
		[T,A],
		[np.logspace(math.log(min(T),10),math.log(max(T),10),nbins), np.logspace(math.log(min(A),10),math.log(max(A),10),nbins)],
		[True,True],
		["period","amplitude"],
	)
	plt.show()



def show_clustering(args):
	bm=procure_cluster_data(args)
	t = bm["timestamp_3046_1"]; tm = t.max()
	f = bm[args.fit]          ; fm = f.max()
	c = bm["cluster_index"]   ; cm = c.max()
	plt.plot(bm["entry"], bm["timestamp_3046_1"] / tm, marker='' , ls='-', color='k',       label='time / {:.2f}'.format(tm))
	# plt.plot(bm["entry"], bm[args.fit]           / fm, marker='.', ls='' , color='g',       label='{} / {:.2f}'.format(args.fit,fm))
	plt.plot(bm["entry"], bm["cluster_index"]    / cm, marker='.', ls='' , color='darkred', label='cluster / {}'.format(cm))

	plt.xlabel('entry')
	plt.legend()
	plt.show()

def analyze_drift(args, ):

	# branches_use = {"timestamp_3046_1","area_3046_1","vMax_3046_1","tMax_3046_1","scaler_3046_1"}
	bm = procure_cluster_data(args)

	ci = bm["cluster_index"]
	n_clusters = ci.max() + 1

	cluster_nev  = []
	cluster_i    = []
	cluster_fit  = []
	
	nc = 5 if args.delta_length else 3
	fig,ax = plt.subplots(figsize=(nc*5,5))
	fig.subplots_adjust(
	    top=0.981,
	    bottom=0.049,
	    left=0.04,
	    right=0.96,
	    hspace=0.2,
	    wspace=0.2,
	)

	# list of components for fit model
	fit_model_components = []
	# add background components
	if "q" in args.bg:
		fit_model_components.append(model.quadratic())
	elif "l" in args.bg:
		fit_model_components.append(model.line())
	elif "c" in args.bg:
		fit_model_components.append(model.constant([[0,np.inf]]))
	if "e" in args.bg:
		fit_model_components.append(model.exponential())
	# store number of parameters used by background components
	n_bg_parameters = sum([_.npars for _ in fit_model_components])
	# add gaussians
	gaus_names = []
	GAUS_DEFAULTS  = [-np.inf,np.inf,0.0,np.inf,0.0,np.inf]
	for ig,g in enumerate(data.split_with_defaults(args.gaus,GAUS_DEFAULTS,[float]*6,name_delimiter="=")):
		gaus_names.append(g[0])
		# re-arrange so as to have mu bounds specified first
		this_bounds = [[g[5],g[6]], [g[1],g[2]], [g[3],g[4]]]
		fit_model_components.append(model.gaussian(this_bounds))
	# compose model
	fit_model = fit_model_components[0]
	for component in fit_model_components[1:]:
		fit_model = fit_model + component
	# calculate bins
	# bins = np.linspace(args.fit_lo, args.fit_hi, args.bins)
	bins = np.linspace(bm[args.fit].min(), bm[args.fit].max(), args.bins+1)


	for i in range(n_clusters):

		mask = data.cut("cluster_index", i - 0.1, i + 0.1)
		masked_branches = bm.mask(mask, {"timestamp_3046_1",args.fit})

		this_t    = masked_branches["timestamp_3046_1"]
		this_data = masked_branches[args.fit]

		this_nev = this_t.size

		counts, edges = np.histogram(this_data, bins=bins)
		midpoints = (edges[1:] + edges[:-1])*0.5

		popt, pcov, chi2, ndof = fit_model.fit(midpoints, counts, p0=popt if i>0 else None)
		# try:
		# 	popt, pcov, chi2, ndof = fit_model.fit(midpoints, counts, p0=popt if i>0 else None)
		# except Exception as E:
		# 	print(E)
		# 	print("{} - {} - fit failed".format(i, this_nev))
		# 	# plt.step(midpoints, counts, where='mid', color="k", label="data")
		# 	# plt.plot(midpoints, fit_model(midpoints, *popt), 'g-', label="last good fit")
		# 	# plt.title("erroring cluster\nusing last good popt")
		# 	# plt.show()
		# 	continue

		if not i:
			plt.subplot(1,nc,1)
			plt.step(midpoints, counts, where='mid', color="k", label="data")
			plt.plot(midpoints, fit_model(midpoints, *popt), 'g-', label="best fit")
			plt.title("run {}, cluster {}\nchi2/ndof={:.2f}/{}={:.2f}".format(args.run,i,chi2,ndof,chi2/ndof))
			# plt.show()

		cluster_nev.append(this_t.size)
		cluster_i.append(i)
		cluster_fit.append([popt, pcov, chi2, ndof])
		print("{:<3} - {} - {} - {} - {} - {}".format(i, this_t.size, [round(_,3) for _ in popt], [round(_,3) for _ in pcov], chi2, ndof))


	cluster_i   = np.array(cluster_i  )
	cluster_nev = np.array(cluster_nev)

	plt.subplot(1,nc,2)
	plt.plot(cluster_i, cluster_nev, 'k.', )
	plt.xlabel('cluster index')
	plt.ylabel('number of events')
	plt.title('number of events per cluster\nRun {}'.format(args.run))
	# plt.show()

	cluster_popt = np.stack([_[0] for _ in cluster_fit],axis=0)
	cluster_pcov = np.stack([_[1] for _ in cluster_fit],axis=0)


	# plot parameters
	poi = args.poi if args.poi>=0 else fit_model.npars+args.poi
	qopt = cluster_popt[:,poi]
	qcov = cluster_pcov[:,poi]
	
	pm_const = model.constant()
	pm_popt, pm_pcov, pm_chi2, pm_ndof = pm_const.fit_with_errors(cluster_i, qopt, xerr=None, yerr=qcov)

	plt.subplot(1,nc,3)
	colors = ['k','g','b','darkred','tab:brown','m','c','tab:red',"peru","orange","olive","teal","tab:purple"]
	for j in range(fit_model.npars):
		if j!=poi:
			continue
		this_popt, this_pcov, this_chi2, this_ndof = pm_const.fit_with_errors(cluster_i, cluster_popt[:,j], xerr=None, yerr=cluster_pcov[:,j])	
		this_label = "{}: {:.3f}".format(fit_model.pnames[j], this_chi2/this_ndof)
		plt.errorbar(cluster_i, cluster_popt[:,j], cluster_pcov[:,j], color=colors[j], ls='', marker='.', label=this_label)
		plt.plot(cluster_i, pm_const(cluster_i, *this_popt), color=colors[j], ls='--')

	plt.xlabel('cluster index')
	plt.ylabel('parameter values')
	plt.title("fit parameter values per cluster, run {}\n{} fit chi2/dof = {:.2f}/{} = {:.2f}".format(args.run, fit_model.pnames[poi], pm_chi2, pm_ndof, pm_chi2/pm_ndof))
	plt.legend()
	# plt.show()



	chi_model = model.gaus([[0,np.inf],[-np.inf,np.inf],[0,np.inf]])


	# analyze parameters over cluster index
	if args.delta_length:

		isep  = args.delta_length
		d     = (qopt[isep:] - qopt[:-isep]      )
		d_var = (qcov[isep:]**2 + qcov[:-isep]**2)
		d_std = np.sqrt(d_var)
		d_ind = cluster_i[isep:]

		fit_d = pm_const.fit_with_errors(d_ind, d, xerr=None, yerr=np.sqrt(d_var))
		
		chi2_d_zero = ((d**2) / d_var).sum()
		ndof_d_zero = d.size

		# plt.errorbar(cluster_i, qopt-np.mean(qopt), qcov, color='k', ls='', marker='.', label='mu[i] - avg(mu)')
		plt.subplot(1,nc,4)
		plt.errorbar(d_ind, d, d_std, color='darkred', ls='', marker='.', label="{0}[i] - {0}[i-{1}]".format(fit_model.pnames[poi],isep))
		plt.plot(d_ind, pm_const(d_ind,*fit_d[0]), color='r', ls='--', marker='', label='constant fit to difference')
		plt.axhline(0, color='b', ls='-', label='zero')
		plt.xlabel("cluster index")
		plt.ylabel("difference between clusters")
		plt.title("c=0: chi2/dof = {:.2f} / {} = {:.3f} \nc={:.3f}: chi2/dof = {:.2f} / {} = {:.3f}".format(
			chi2_d_zero, ndof_d_zero, chi2_d_zero/ndof_d_zero,
			fit_d[0][0],fit_d[2],fit_d[3],fit_d[2]/fit_d[3],
		))
		plt.legend()
		# plt.show()

		# analyze distribution of chi
		d_chi = d / d_std

		# max_abs_chi = abs(d_chi).max()
		bins_chi = np.linspace(d_chi.min()-1,d_chi.max()+1,20+1)

		counts_chi, edges_chi = np.histogram(d_chi, bins=bins_chi)
		midpoints_chi = (edges_chi[1:] + edges_chi[:-1])*0.5

		popt_chi, perr_chi, chisq_chi, ndof_chi = chi_model.fit(midpoints_chi, counts_chi)


		plt.subplot(1,nc,5)
		plt.step(midpoints_chi, counts_chi, where='mid', color="k", label="data")
		plt.plot(midpoints_chi, chi_model(midpoints_chi, *popt_chi), 'g-', label="best fit")
		popt_chi_string = ", ".join(["{}={:.2f}\xb1{:.3f}".format(_,popt_chi[i], perr_chi[i]) for i,_ in enumerate(chi_model.pnames)])
		plt.title("run {}, change in {}, chi2/ndof={:.2f}/{}={:.3f}\n{}".format(args.run,fit_model.pnames[poi],chisq_chi,ndof_chi,chisq_chi/ndof_chi,popt_chi_string))
		plt.xlabel("chi = delta_ij / err(delta_ij)")
		plt.ylabel("counts")


	plt.tight_layout()

	if args.fig:
		
		# just filename: save in ./figs/
		if not (os.sep in args.fig):
			fig_file = FIG_LOC.format(args.fig)
		else:
			fig_file = args.fig

		# save the figure to an image file
		plt.savefig(fig_file)

	plt.show()






def main(args):

	print(args)
	routine = args.do[0]

	if routine == "drift":
		print("performing drift analysis")
		analyze_drift(args)

	elif routine == "fourier":
		print("performing fourier analysis")
		analyze_fourier(args)

	elif routine == "show":
		print("showing clustering results")
		show_clustering(args)

	else:
		print("unrecognized analysis routine: {}".format(routine))

	return




if __name__ == '__main__':

	# main(None)
	# sys.exit(0)

	parser = argparse.ArgumentParser(
		description="analysis using cluster identification to separate dataset into subsets",
	)


	# dataset specification
	parser.add_argument("run",type=str,help="file location, name, or number")
	parser.add_argument("fit",type=str,nargs="*",help="branch to fit, low bound, high bound")
	# parser.add_argument("fit_lo",type=float,default=-np.inf,help="low bound on fit range")
	# parser.add_argument("fit_hi",type=float,default= np.inf,help="high bound on fit range")
	parser.add_argument("--cut",type=str,action='append',default=[],help="branch to cut on: branch,min,max")


	# fitting specification
	parser.add_argument("--bins",type=int,default=100,help="number of bins to use")
	parser.add_argument("--bg"  ,type=str,default="c",help="background function: any combination of (p)ower (e)xp (c)onstant (l)ine (q)uadratic")
	parser.add_argument("--g"   ,type=str,default=[],action='append',dest="gaus",help="gaussian components: min_mu,max_mu (or) name=min_mu,max_mu")


	# clustering details
	parser.add_argument("--cluster-branch"   ,"--cb",type=str  ,default="timestamp_",help="branch used to determine clusters")
	parser.add_argument("--cluster-diff"     ,"--cd",type=int  ,default=1           ,help="differentiate cluster branch? 0 = no; int>0 = kernel size")
	parser.add_argument("--cluster-threshold","--ct",type=float,default=10.0        ,help="minimum value for cluster boundary")
	parser.add_argument("--cluster-size-min" ,"--cs",type=int  ,default=0           ,help="minimum number of datapoints before ending cluster")


	# analysis routine
	parser.add_argument("--do",type=str,nargs="+",default=["drift"],help="what analysis to perform, and any extra arguments it needs")
	parser.add_argument("--poi",type=int,default=-2,help="which parameter from fit to analyze. negative = count back from end of list.")
	parser.add_argument("--delta-length","--d",type=int,default=0,help="analyze difference between clusters with indices separated by this amount")


	# output
	parser.add_argument("--fig",type=str,default="",help="location to save figure as png image (overwrites if file exists)")


	args = parser.parse_args()

	# jank
	merge = lambda ltop,lbot:ltop if len(ltop) >= len(lbot) else ltop + lbot[len(ltop):]
	for i in range(1,len(args.fit)):
		args.fit[i] = float(args.fit[i])
	args.fit, args.fit_lo, args.fit_hi = merge(args.fit, [None, -np.inf, np.inf])

	main(args)
