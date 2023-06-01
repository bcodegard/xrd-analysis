"""
script used for testing features during development
don't actually use this for anything else
"""

import numpy
import math
import sys

import numpy as np

from PyQt6 import QtGui, QtCore
import PyQt6.QtWidgets as Widgets

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

	vspan = release.inaxes.axvspan(
		press.xdata,
		release.xdata,

		ec = "#40d030ff",
		fc = "#40d03033",

		# color = 'g',
		# alpha = 0.15,
	)

	# make pickable
	vspan.set_picker(True)

	release.canvas.draw()




if __name__ == "__main__":

	xaxis = np.linspace(10, 350, 200)
	
	ydist = 15.0 + xaxis*0.015 + 15 / (1 + ((xaxis - 175) / 21)**2)
	ydata = np.random.poisson(ydist)


	fig, ax = plt.subplots()

	fi = interact.FigureInteractor(fig, verbosity = 1)
	fi.bind_key("a", lambda event:print("you pressed a"))
	fi.bind_key("V", add_vline)

	fi.bind_button_press(1, alert)
	fi.bind_button_press(1, alert, "alt")
	fi.bind_button_press(3, alert, ["alt", "shift"])

	fi.bind_button_release(1, release, "alt")


	# qt stuff!?
	toolbar = fig.canvas.toolbar
	
	line_edit = Widgets.QLineEdit()
	fig.canvas.toolbar.addWidget(line_edit)
	line_edit.setStyleSheet("background-color: #101040")
	line_edit.editingFinished.connect(lambda *args:print(args))

	# tree = Widgets.QTreeWidget()
	# fig.canvas.toolbar.addWidget(tree)
	# tree.setStyleSheet("background-color: #606080")
	# tree.addTopLevelItem(Widgets.QTreeWidgetItem(("what", "the")))


	# pickable artists!
	# 
	# In theory, could have tree view widget that gives access data of
	# any picked artist, and allows for editing.
	# Could have binding for selecting artist to edit, E.G. alt + left
	# 
	# different artists could also have individual functionality,
	# for example click to move vlines, vspans, legend, etc.
	# Maybe also a key combo for moving, E.G. (shift + left) drag & drop
	# 
	# key combo for removing
	# 
	s1, = ax.step(xaxis, ydata, 'k-' , where='mid', label='generated sample')
	s2, = ax.step(xaxis, ydist, 'r--', where='mid', label='true distribution')
	s1.set_picker(True)
	s2.set_picker(True)
	fig.canvas.mpl_connect('pick_event', lambda event: print(event.artist))


	# label and show
	ax.set_xlabel("x axis")
	ax.set_ylabel("counts")
	plt.legend()
	plt.show()




