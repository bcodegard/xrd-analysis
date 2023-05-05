"""
Class for handling user input in matplotlib window
"""


# message templates for debug info on key press events
MSG_KEY_PRESS   = "+   {:>24}"
MSG_KEY_RELEASE = "-   {:<24}"


# delimiter character for multi-key events,
# E.G. "ctrl+shift+a"
KEY_COMBO_DELIMITER = "+"

# rename keys before processing combinations.
# necessary because ctrl shows up as "control" when not part of a combo
# but as "ctrl" while part of a combo. who thought that was a good idea?
KEY_RENAME = {
	"control" : "ctrl"  , # I cannot fathom the reason for this
	" "       : "space" , # understandable, but I prefer 'space'
	"meta"    : "alt"   , # seriously, why
}

COMBO_KEYS = {
	"ctrl" ,
	"alt"  ,
}

def parse_keys(event):
	# print("\n\n")
	# print(repr(event.key))
	keys_raw = event.key.split(KEY_COMBO_DELIMITER)
	keys = [KEY_RENAME.get(_,_) for _ in keys_raw]

	# since '+' is both the delimiter and a key, and that's not escaped,
	# we have a headache. for now, just print a warning if there are
	# empty strings in keys, as that's how the '+' key is currently parsed.
	if not all(keys):
		print("Warning: empty key found in event {}".format(event.key))

	# I can't find any documentation on what the order represents, but
	# best as I can tell, the last key listed is the one being changed,
	# and the preceeding keys are the ones currently being held down.
	changed = keys[-1]
	held    = keys[:-1]

	return changed, held






class MPLInteract(object):


	def __init__(self, figure, verbosity=0):
		
		self._fig = figure

		self.verbosity = verbosity

		self._setup()
		self._connect()

	def _setup(self):

		# key combo tracking
		self._in_combo  = False # whether there's a combo ongoing
		self._combo_key = None  # key that started current combo
		self._combo     = None  # list of events in current combo

	def _connect(self):
		self._fig.canvas.mpl_connect('key_press_event'  , self._on_key_press  )
		self._fig.canvas.mpl_connect('key_release_event', self._on_key_release)




	def _on_key_press(self, event):
		# parse event data
		changed, held = parse_keys(event)

		# print("press {}".format(repr(event.key)))
		# print("press     {:>16}   holding {}".format(repr(changed), repr(held)))

		# if not in combo, and pressed key is a combo key, start a combo
		if not self._in_combo:
			if changed in COMBO_KEYS:
				self._start_combo(changed)

		# if still not in combo, return
		if not self._in_combo:
			return

		# append the event to the current combo
		self._combo.append((changed, held))


	def _on_key_release(self, event):
		# parse event data
		changed, held = parse_keys(event)

		# print("release {}".format(repr(event.key)))
		# print("release   {:<16}   holding {}".format(repr(changed), repr(held)))

		# if not in combo, we don't care
		if not self._in_combo:
			return

		# if changed key is not a combo key, we don't care
		if changed not in COMBO_KEYS:
			return

		# if released combo key is not the one that started the combo, 
		# do nothing.
		if changed != self._combo_key:
			return

		# end the combo when the starter key is released.
		self._end_combo()




	def _start_combo(self, combo_key):
		# print("\nstart combo with modifier {}".format(combo_key))
		self._in_combo  = True
		self._combo_key = combo_key
		self._combo     = []

	def _end_combo(self):
		# print("end combo with modifier {}".format(self._combo_key))
		
		# run the combo
		self._run_combo(self._combo_key, self._combo)

		# clean up and return to not in combo state
		self._in_combo  = False
		self._combo_key = None
		self._combo     = None

	def _run_combo(self, combo_key, combo):
		print("base class just prints combo information upon combo completion")
		print("combo key: {}".format(combo_key))
		print("combo: {}".format(combo))
		print("")







class FitMPLI(MPLInteract):
	"""docstring for FitMPLI"""
	
	def __init__(self, figure, axes, verbosity=3):
		super(FitMPLI, self).__init__(figure, verbosity)
		self._ax = axes

