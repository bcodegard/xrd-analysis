"""
Handles configurations, which are stored as .yaml files
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"

import os
import shutil
from yaml import safe_load




# directory to look for config files, if not specified
_DIR_CONFIG = os.sep.join([os.path.dirname(__file__), '..', '..', 'config'])

# directory to look for default config files, if not specified
_SUBDIR_DEFAULT_CONFIG = "default-config"
_DIR_DEFAULT_CONFIG    = os.sep.join([_DIR_CONFIG, _SUBDIR_DEFAULT_CONFIG])

# file extension for config files
_CFG_EXT = ".yaml"




# utility functions

def deep_merge(*dicts):
	"""recursively merge any number of dicts.
	priority given to earliest entry per key"""
	
	if len(dicts) == 1:
		return dicts[0]

	elif len(dicts) == 2:
		d1,d2 = dicts

		if not (isinstance(d1, dict) and isinstance(d2, dict)):
			return d1

		k1 = d1.keys()
		k2 = d2.keys()
		keys_or  = k1|k2
		keys_and = k1&k2
		
		d={}
		for key in keys_or:
			if key in keys_and:
				if isinstance(d1[key], dict):
					d[key] = deep_merge(d1[key], d2[key])
				else:
					d[key] = d1[key]
			else:
				d[key] = d1.get(key, d2.get(key))

		return d

	else:
		return deep_merge(deep_merge(*dicts[:-1]), dicts[-1])

def _finish_path(name, directory, extension):

	# add "." to start of extension if missing
	if not (extension.startswith('.')):
		extension = '.' + extension

	# apply the extension if missing
	if not (name.endswith(extension)):
		name = name + extension

	# remove trailing separator from directory
	if directory.endswith(os.sep):
		directory = directory.rpartition(os.sep)[0]

	# join os.sep with directory and name
	return os.sep.join([directory, name])

def _vprint(min_level, level, what):
	if level >= min_level:
		print(what)




# i/o functions

def _load_yaml(path):
	with open(path, 'rb') as stream:
		config_yaml = safe_load(stream)
	return config_yaml

def _load_multiple(files, dir_config, verbosity=1):
	"""load each file from a list and return list of files' contents"""
	# load all config files
	configs = [_load_yaml(_finish_path(_,dir_config,_CFG_EXT,)) for _ in files]
	# deep merge and return
	return deep_merge(*configs)

_ERR_NO_FILE_OR_DEFAULT = "Could not find config or default for file '{file}'"
def _copy_default_if_missing(files, dir_config, dir_default_config, verbosity=1):
	"""check that each file exists; try to copy associated default file
	if it does not."""
	for file in files:
		dir_file = _finish_path(file, dir_config, _CFG_EXT)

		if not os.path.exists(dir_file):
			_vprint(1,verbosity,"config file {} not found; looking for default...".format(dir_file))
			dir_file_default = _finish_path(file, dir_default_config, _CFG_EXT)
	
			if os.path.exists(dir_file_default):
				_vprint(1,verbosity,"found default config file {}; copying it.".format(dir_file_default))
				shutil.copyfile(dir_file_default, dir_file)
			else:
				_vprint(1,verbosity,"did not find default config file {}".format(dir_file_default))
				raise FileNotFoundError(_ERR_NO_FILE_OR_DEFAULT.format(file=file))




# the "main" function of this module
# this is the only one that external scripts should call.

def load(*file_or_files, dir_config=None, dir_default_config=None, verbosity=1):
	"""Load one or more configuration files. If more than one, priority
	for overlapping fields is given to the earliest specified entry.
	
	File names cannot be full directories. load a config file from a
	direcotry other than the hard-coded one, specify it for dir_config.
	"""

	# no config files specified -> just load common
	if not file_or_files:
		file_or_files = ["common"]

	# set directories if not specified
	if dir_config is None:
		dir_config = _DIR_CONFIG
	if dir_default_config is None:
		dir_default_config = _DIR_DEFAULT_CONFIG

	# convert file_or_files to list of strings if it's a single string
	if isinstance(file_or_files, str):
		file_or_files = [file_or_files]

	# copy files from dir_default_config
	# if they're missing from dir_config
	_copy_default_if_missing(
		file_or_files,
		dir_config,
		dir_default_config,
		verbosity,
	)
	
	# load, merge, and return contents of config files
	return _load_multiple(file_or_files,dir_config,verbosity)
