"""
script used for testing features during development
don't actually use this for anything else
"""

import numpy
import math
import sys
import numpy as np

import utils.interact    as interact
import matplotlib.pyplot as plt


M_KEY_PRESS_BRIEF   = "press   {}" 
M_KEY_RELEASE_BRIEF = "release {}" 

def on_key_press(event):
	# print("press key {} at {},{}".format(event.key, event.xdata, event.ydata))
	print(M_KEY_PRESS_BRIEF.format(event.key))

def on_key_release(event):
	# print("release key {} at {},{}".format(event.key, event.xdata, event.ydata))
	print(M_KEY_RELEASE_BRIEF.format(event.key))


if __name__ == "__main__":

	xaxis = np.linspace(10, 350, 200)
	
	ydist = 15.0 + xaxis*0.075 + 35 / (1 + ((xaxis - 175) / 12)**2)
	ydata = np.random.poisson(ydist)


	fig, ax = plt.subplots()


	# fig.canvas.mpl_connect('key_press_event', on_key_press)
	# fig.canvas.mpl_connect('key_release_event', on_key_release)
	mpli = interact.FitMPLI(fig, ax, verbosity = 3)


	ax.step(xaxis, ydata, 'k-' , where='mid', )
	ax.step(xaxis, ydist, 'r--', where='mid', )
	ax.set_xlabel("x axis")
	ax.set_ylabel("generated counts")
	plt.show()




