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



def add_vline(event):
	if event.inaxes:
		event.inaxes.axvline(event.xdata, color='k', linestyle='--', label="{:.1f}".format(event.xdata))
		event.canvas.draw()

def alert(event):
	print("alert: event {}".format(event))


def release(press, release):
	print("just released")
	print("press   : {}".format(press))
	print("release : {}".format(release))

	release.inaxes.axvspan(
		press.xdata,
		release.xdata,

		ec = "#40d030ff",
		fc = "#40d03033",

		# color = 'g',
		# alpha = 0.15,
	)

	release.canvas.draw()




if __name__ == "__main__":

	xaxis = np.linspace(10, 350, 200)
	
	ydist = 15.0 + xaxis*0.075 + 35 / (1 + ((xaxis - 175) / 12)**2)
	ydata = np.random.poisson(ydist)


	fig, ax = plt.subplots()

	fi = interact.FigureInteractor(fig, verbosity = 1)
	fi.bind_key("a", lambda event:print("you pressed a"))
	fi.bind_key("V", add_vline)

	fi.bind_button_press(1, alert)
	fi.bind_button_press(1, alert, "alt")
	fi.bind_button_press(3, alert, ["alt", "shift"])

	fi.bind_button_release(1, release, "alt")

	ax.step(xaxis, ydata, 'k-' , where='mid', label='generated sample')
	ax.step(xaxis, ydist, 'r--', where='mid', label='true distribution')
	ax.set_xlabel("x axis")
	ax.set_ylabel("counts")
	plt.legend()
	plt.show()




