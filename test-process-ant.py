""""""

import process.ant as ant



if __name__ == "__main__":
	f0 = "/home/bode/Documents/python/xrd-scope-pulses/ant/Run6000_pulses.ant"
	f1 = "/home/bode/Documents/python/xrd-scope-pulses/ant/Run6141_pulses.ant"

	ant.convert_ant_to_npz(f0, "./Run6000.npz")

