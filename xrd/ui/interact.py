"""
Class for handling user input in matplotlib window
"""



def button_token(button, modifiers):
	if modifiers is None:
		modifiers = ()
	if isinstance(modifiers, str):
		modifiers = (modifiers,)
	return (button,) + tuple(sorted(modifiers))



class FigureInteractor(object):

	def __init__(self, figure, verbosity=0):
		self.verbosity = verbosity
		self._fig = figure
		self._mpl_connect()
		self._setup()

	def _mpl_connect(self):
		self._fig.canvas.mpl_connect('key_press_event'  , self._on_key_press  )
		self._fig.canvas.mpl_connect('key_release_event', self._on_key_release)

		self._fig.canvas.mpl_connect('button_press_event'  , self._on_button_press  )
		self._fig.canvas.mpl_connect('button_release_event', self._on_button_release)
		# self._fig.canvas.mpl_connect('motion_notify_event' , self._on_motion_notify )


	def _setup(self):
		
		self._key_bindings = {}

		self._button_press_bindings = {}
		self._button_release_bindings = {}

		self._last_button_press = {}


	def bind_key(self, key, recipient):
		if key not in self._key_bindings:
			self._key_bindings[key] = []
		self._key_bindings[key].append(recipient)

	def unbind_key(self, key):
		del self._key_bindings[key]

	def unbind_all_keys(self):
		self._key_bindings.clear()

	def _on_key_press(self, event):
		if self.verbosity >= 2:
			print("key pressed: {}".format(event.key))
		for recipient in self._key_bindings.get(event.key, ()):
			# if a recipient returns something truthy, this indicates
			# that the event should not be given to any further
			# recipients.
			if recipient(event):
				break

	def _on_key_release(self, event):
		if self.verbosity >= 2:
			print("key released: {}".format(event.key))


	def bind_button_press(self, button, recipient, modifiers = None):
		token = button_token(button, modifiers)
		if token not in self._button_press_bindings:
			self._button_press_bindings[token] = []
		self._button_press_bindings[token].append(recipient)

	def bind_button_release(self, button, recipient, modifiers = None):
		token = button_token(button, modifiers)
		if token not in self._button_release_bindings:
			self._button_release_bindings[token] = []
		self._button_release_bindings[token].append(recipient)

	def unbind_button_press(self, button, modifiers = None):
		token = button_token(button, modifiers)
		del self._button_press_bindings[token]

	def unbind_button_release(self, button, modifiers = None):
		token = button_token(button, modifiers)
		del self._button_release_bindings[token]

	def _on_button_press(self, event):

		# store as the last button press event for this button
		# so that release events have access to where they started
		self._last_button_press[event.button.value] = event

		# press must be in axes
		if not event.inaxes:
			return

		# must not have any mode selected in navigation toolbar
		if event.canvas.toolbar.mode != "":
			return

		# construct token
		token = button_token(event.button.value, event.modifiers)
		# print(token)
		# print("button_press   {} {:.1f} {:.1f} {}".format(event.button, event.xdata, event.ydata, event.modifiers))

		# call recipients
		for recipient in self._button_press_bindings.get(token,()):
			if recipient(event):
				break

	def _on_button_release(self, event):
		
		# fetch and then clear last button press with same button value
		press = self._last_button_press.get(event.button.value, None)
		self._last_button_press[event.button.value] = None

		# release must be in axes
		if not event.inaxes:
			return

		# must not have mode selected in navigation toolbar
		# print(repr(event.canvas.toolbar.mode))
		if event.canvas.toolbar.mode != "":
			return

		# press and release must be in same axes
		if press.inaxes is not event.inaxes:
			return

		# construct token for button and modifiers
		token = button_token(event.button.value, event.modifiers)

		# press and release must have identical token
		if token != button_token(press.button.value, press.modifiers):
			return
		
		# print(token)
		# print("button_release {} {:.1f} {:.1f} {}".format(event.button, event.xdata, event.ydata, event.modifiers))

		# call recipients
		for recipient in self._button_release_bindings.get(token,()):
			if recipient(press, event):
				break

	# def _on_motion_notify(self, event):
	# 	if not event.inaxes:
	# 		return
	# 	print("motion_notify  {} {:.1f} {:.1f} {}".format(event.button, event.xdata, event.ydata, event.key))

