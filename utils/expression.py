"""
compile strings representing mathematical expressions into callable
functions, after checking that they include no forbidden nodes,
importantly neither function calls nor attribute access.
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"

import ast




# allowed types for constants
constant_type_whitelist = [
	int,
	float,
	complex,
]

# types of nodes allowed in expressions
node_type_whitelist = [

	# the expression itself
	ast.Expression,

	# constants and dynamic variables
	ast.Constant,
	ast.Name,
	ast.Load,

	# unary operators
	ast.UnaryOp,
	ast.USub,
	ast.Invert,
	ast.Not,
	
	# binary operators
	ast.BinOp,
	ast.Add,
	ast.Sub,
	ast.Mult,
	ast.Div,
	ast.FloorDiv,
	ast.Mod,
	ast.Pow,
	ast.LShift,
	ast.RShift,
	ast.BitOr,
	ast.BitXor,
	ast.BitAnd,

	# comparisons
	ast.Compare,
	ast.Eq,
	ast.NotEq,
	ast.Lt,
	ast.LtE,
	ast.Gt,
	ast.GtE,

]

# types of nodes explicitly disallowed
# while any type not in the whitelist is disallowed,
# types in this list will raise errors even if they
# are put in to the whitelist.
node_type_blacklist = [
	
	# calling functions is forbidden
	# if support is desired for mathematical functions which do not have
	# dedicated operators, for instance cosine, then a whitelist of 
	# function names will need to be established, and greated care paid
	# to ensuring that the evaluation environment cannot contain any
	# other functions.
	ast.Call,
	ast.keyword,

	# accessing attributes is forbidden
	ast.Attribute,
	
	# boolean operators: these don't work with numpy arrays.
	ast.BoolOp,
	ast.IfExp,

	# boolean operators are forbidden
	# is, is not, in, not in, else
	# these represent access to information which should not be needed
	# for mathematical expressions, and are therefore forbidden.
	ast.Is,
	ast.IsNot,
	ast.In,
	ast.NotIn,

	# subscripting is forbidden
	ast.Subscript,
	ast.Index,
	ast.Slice,
	ast.ExtSlice,

	# storing or deleting names is forbidden
	ast.Store,
	ast.Del,

]


ERR_NODE_NOT_WHITELISTED = "{} is not a whilelisted node type"
ERR_NODE_BLACKLISTED = "{} is a blacklisted node type"
ERR_CONSTANT_TYPE_NOT_WHITELISTED = "{} is not a whitelisted type for constants"

def get_names(expression):
	"""get a list of names in expression"""
	names = set()
	for node in ast.walk(expression):
		if type(node) is ast.Name:
			names.add(node.id)
	return names

def check(expression):
	"""walks through a parsed expression and checks each node
	against a whitelist and a redundant blacklist, as well as
	checking further node properties for some node types.
	"""
	for node in ast.walk(expression):

		# node type must be whitelisted
		if type(node) not in node_type_whitelist:
			raise ValueError(ERR_NODE_NOT_WHITELISTED.format(type(node)))
		
		# additionally, it must not be blacklisted
		if type(node) in node_type_blacklist:
			raise ValueError(ERR_NODE_BLACKLISTED.format(type(node)))

		# if it's a constant, check its type
		if type(node) is ast.Constant:
			if type(node.value) not in constant_type_whitelist:
				raise ValueError(ERR_CONSTANT_TYPE_NOT_WHITELISTED.format(type(node.value)))

def check_and_compile(expression_string):
	"""Parse a string into an expression, check that it has no illegal
	nodes, importantly including function calls and attribute access.
	If it passes, compile it, and return a wrapped function which calls
	eval ensuring that globals is empty, including __builtins__"""

	# parse the expression
	expression = ast.parse(expression_string, "<string>", "eval")

	# check that the parsed expression contains no illegal nodes
	check(expression)

	# compile the expression into code
	compiled_expression = compile(expression, "<string>", "eval")

	# wrap the code to ensure that globals cannot contain anything,
	# including __builtins__, and that locals contains only the items
	# supplied by the user when calling the function.
	@expects(get_names(expression))
	def eval_with_no_builtins(**items):
		return eval(compiled_expression, {"__builtins__":{}}, items)
	return eval_with_no_builtins



def expects(names):
	def wrapper(fn):
		return decorated_function(fn, [], set(names))
	return wrapper

class decorated_function(object):
	def __init__(self, fn, argnames=[], kwargnames=set()):
		self._fn = fn
		self.argnames = argnames
		self.kwargnames = kwargnames
	def __call__(self, *args, **kwargs):
		return self._fn(*args, **kwargs)
