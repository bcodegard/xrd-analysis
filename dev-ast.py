"""
script used for testing features during development
don't actually use this for anything else
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"

import math
import numpy as np

import utils.expression as expr




if __name__ == '__main__':

	line = np.linspace(0,1,21)
	ints = np.arange(0,21)

	print("access names")
	n1s = "(area/vMax)**c1"
	n1 = expr.check_and_compile(n1s)
	print(n1s)
	print(n1.kwargnames)

	print("")
	
	print("comparisons")
	c1s = "(x>0.9)|((x-0.2)**2<0.01)"
	c1 = expr.check_and_compile(c1s)
	print(c1s)
	print(c1(x=7))
	print(c1(x=line))

	c2s = "(x==0)|(x==1)"
	c2 = expr.check_and_compile(c2s)
	print(c2s)
	print(c2(x=3))
	print(c2(x=1))
	print(c2(x=1.0))
	print(c2(x=line))

	print("")

	print("functions")
	f1s = "(x**2 + y**2)**0.5"
	f1 = expr.check_and_compile(f1s)
	print(f1s)
	print(f1(x=3, y=4))
	print(f1(x=line*2, y=line[::-1]))

	f2s = "i>>1"
	f2 = expr.check_and_compile(f2s)
	print(f2s)
	print(f2(i=ints))
	print(f2(i=3*ints))

	f3s = "((i%3)==0)|(x>0.8)"
	f3 = expr.check_and_compile(f3s)
	print(f3s)
	print(f3(i=ints, x=line))

	print("")

	print("disallowed types for constants")
	bad_expressions_type = [
		"x + 'spam'",
		"x + b'eggs'",
	]
	for string in bad_expressions_type:
		try:
			expr.check_and_compile(string)
		except Exception as e:
			print("string {:<18} failed to parse. error: {}".format(string,e))

	print("")

	print("disallowed node types")
	bad_expressions_node = [
		"x.dtype",
		"eval ( 'thing' )",
		"__import__('os')",
		"y()",
		"x is y",
		"x in y",
		"x in [0,1,2]",
		"[0,1,2]",
		"x[0]",
	]
	for string in bad_expressions_node:
		try:
			expr.check_and_compile(string)
		except Exception as e:
			print("string {:<18} failed to parse. error: {}".format(string, e))

