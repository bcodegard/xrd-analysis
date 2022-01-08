import matplotlib
import matplotlib.figure

import sys
import numpy as np

import utils.fileio as fileio
import utils.display as display

if __name__ == "__main__":

	file_out = sys.argv[1]
	files_in = sys.argv[2:]

	if len(files_in) > 1:

		figures = [fileio.load_figure(_) for _ in files_in]

		# l1 = figures[0].get_axes()[0].get_lines()[0]
		# d1 = l1.get_xdata()
		# print(l1.get_transform().transform([3,3]))
		# print(figures[0].get_axes()[0].have_units())
		# print(d1)
		# print(type(d1))
		# print(d1.dtype)
		# print("")

		fcom = matplotlib.figure.Figure()

		# # for now assume all figures have exactly one axes object
		# acom = fcom.add_subplot(1,1,1)
		# display.stack_axes([_.get_axes()[0] for _ in figures], acom)

		# l2 = fcom.get_axes()[0].get_lines()[0]
		# d2 = l2.get_xdata()
		# print(l2.get_transform().transform([3,3]))
		# print(acom.have_units())
		# print(d2)
		# print(type(d2))
		# print(d2.dtype)
		# print("")


		# acom.set_xscale("log")
		# acom.set_yscale("log")

		fcom.savefig(file_out)

	else:

		fileio.load_figure(files_in[0]).savefig(file_out)


	# if True:
	# 	x1 = np.linspace(0,1,100)
	# 	x2 = np.linspace(2,4,10)

	# 	y1a = x1*3
	# 	y1b = 2*x1**2 - x1

	# 	y2a = np.cos(x2)
	# 	y2b = np.cos(x2*1.1)

	# 	f1 = matplotlib.figure.Figure()
	# 	a1 = f1.add_subplot(1,1,1)
	# 	a1.errorbar(x1,y1a,y1a**0.5,color="r",ls='--',label="shmer")
	# 	a1.plot(x2,y2a,color='b', ls='--', marker='^',label="gmuh")
	# 	a1.axvline(1.5, color='k', linestyle='-',label="fuh")
	# 	a1.set_xlabel("x")
	# 	a1.set_ylabel("y")
	# 	a1.set_title("donor 1")
	# 	a1.legend()

	# 	f2 = matplotlib.figure.Figure()
	# 	a2 = f2.add_subplot(1,1,1)
	# 	a2.plot(x1,y1b,"r-",label="squawk")
	# 	a2.plot(x2,y2b,color='b', ls='-', marker='s',label="meep")
	# 	a2.set_xlabel("x axis")
	# 	a2.set_ylabel("y axis")
	# 	a2.set_title("donor 2")
	# 	a2.legend()

	# 	f1.savefig("./test_f1.png")
	# 	f2.savefig("./test_f2.png")

	# 	f3 = matplotlib.figure.Figure()
	# 	a3 = f3.add_subplot(1,1,1)
	# 	display.stack_axes([a1,a2],a3)
	# 	a3.set_xlabel("all x")
	# 	a3.set_ylabel("all y")
	# 	f3.savefig("./test_f3.png")
