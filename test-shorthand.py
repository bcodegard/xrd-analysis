"""
script used for testing features during development
don't actually use this for anything else
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"

import re
import sys

import utils.config as config



class ObjDict(dict):
	"""Provides recursive attribute-style access to dictionary items"""

	@classmethod
	def convert(cls, obj):
		if isinstance(obj, cls):
			return obj
		if isinstance(obj, dict):
			return cls({k:cls.convert(v) for k,v in obj.items()})
		if isinstance(obj, list):
			return [cls.convert(_) for _ in obj]
		if isinstance(obj, tuple):
			return tuple(cls.convert(_) for _ in obj)
		if isinstance(obj, set):
			return {cls.convert(_) for _ in obj}
		return obj

	@classmethod
	def _convert_dict(cls, obj):
		return {k:cls.convert(v) for k,v in obj.items()}

	def __init__(self, d):
		super(ObjDict, self).__init__(ObjDict._convert_dict(d))

	def __getattr__(self, attr):
		return self[attr]

	def __setattr__(self, attr, value):
		self[attr] = ObjDict.convert(value)

	def __repr__(self):
		return 'ObjDict({})'.format(super(ObjDict, self).__repr__())

def repl_template(template, ctx=None):
	if ctx is None:
		ctx = {}
	def repl(match):
		# print(match, match.string, match.groups())
		return template.format(*match.groups(), **ctx)
	return repl

def apply_shorthand(string, shorthand, context, enforce_word_boundary=True):

	modified = string
	for pattern,template in shorthand.items():

		if enforce_word_boundary:
			pattern = r"\b" + pattern + r"\b"

		modified = re.sub(pattern, repl_template(template, context), modified)

	return modified





if __name__ == "__main__":

	cfg = config.load("common")
	cfg = ObjDict(cfg)

	# print((cfg.figure_directory))
	# sys.exit(0)


	sh = cfg.shorthand.by_file_type.root

	test_original = "I've got _a4 a4_ a4 A4 a4 A4 t1 t_1 t2 t_2"
	modified = test_original

	context = {"drs":2988, "channel":1}
	

	print(apply_shorthand(test_original, sh, context, True))
	sys.exit(0)






