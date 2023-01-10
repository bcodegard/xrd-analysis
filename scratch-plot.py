import numpy as np
import matplotlib.pyplot as plt




xres = 10000

plot_width_us = 250
time_us = np.linspace(0, plot_width_us, xres)



beam_envelope = np.zeros(xres)
env_start_us = 59
env_width_us = 150
beam_envelope[(time_us>env_start_us)&(time_us<(env_start_us+env_width_us))] = 1

plt.plot(time_us, beam_envelope, 'k--', label="beam envelope (150us)")


beam_duty = np.zeros(xres)
beam_duty_offset_us = -env_start_us
beam_duty_on_ns  = 290
beam_duty_off_ns = 70
beam_duty_tot_ns = beam_duty_off_ns+beam_duty_on_ns
beam_duty[((time_us+beam_duty_offset_us)%(beam_duty_tot_ns/1000)) < (beam_duty_on_ns/1000)] = 1

plt.plot(time_us, beam_envelope*beam_duty, "c-", label="beam duty cycle (360ns)")


event_window_us = 1.800
plt.axvspan(env_start_us, env_start_us + event_window_us, color='g', label="DRS event window")

plt.ylabel("")
plt.xlabel("time (microseconds)")
plt.legend()
plt.show()

