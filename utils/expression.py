"""
compile strings representing mathematical expressions into callable
functions, after checking that they include no forbidden nodes,
importantly neither function calls nor attribute access.
"""

__author__ = "Brunel Odegard"
__version__ = "0.0"

import ast
import numpy as np


# ERR_MAX_ARGS = "function {} takes at most {} args, but {} were given"
# ERR_ARG_FORBIDDEN = "function {} cannot be given kwarg {} (forbidden)"
# def with_fixed(fn, max_args=None, fix_args={}, fix_kwargs={}, forbid_kwargs=set()):
# 	def wrapped(*args,**kwargs):

# 		# ensure that number of args does not exceed maximum,
# 		# to prevent keyword arguments from being supplied as 
# 		# positional arguments
# 		if max_args is not None:
# 			n_args = len(args)+len(args_fix)
# 			if n_args > max_args:
# 				raise ValueError(ERR_MAX_ARGS.format(fn,max_args,n_args))

# 		# merge args with fix_args
# 		args_it = iter(args)
# 		args_use = []
# 		for i in range(len(args)+len(fix_args)):
# 			if i in fix_args:
# 				args_use.append(fix_args[i])
# 			else:
# 				args_use.append(next(args_it))

# 		# check forbidden kwargs
# 		for key in kwargs.keys():
# 			if (key in forbid_kwargs) or (key in fix_kwargs):
# 				raise ValueError(ERR_ARG_FORBIDDEN.format(fn,key))

# 		# merge fixed kwargs
# 		kwargs_use = kwargs|fix_kwargs

# 		# call function with composed args_use and kwargs_use
# 		return fn(*args_use,**kwargs_use)

# 	return wrapped



# this dict is the "globals" argument used when evaluating expressions.
# including "__builtins__":{} ensures that no functions or objects will
# be accessible except those explicitly included here.
# 
# the other pairs in this dict are functions that are available for
# expressions to invoke. note that attribute access is still forbidden.
GLOBALS_EVAL = {
	"__builtins__":{},

	"arange":lambda a,b=None:np.arange(a) if b is None else np.arange(a,b),

	"isnan":lambda a:np.isnan(a),
	
	"flatten":lambda a:a.flatten(),
	
	"at":lambda a,i,ax:np.take_along_axis(a,i.reshape([1 if _i==ax else _ for _i,_ in enumerate(a.shape)]),ax),

	"sum" :lambda a,axis=None:np.sum(a,axis),
	"mean":lambda a,axis=None:np.mean(a,axis),
	"min":lambda a,axis=None:a.min(axis),
	"max":lambda a,axis=None:a.max(axis),
	"std":lambda a,axis=None:np.std(a,axis),
	"pct":lambda a,p,axis=None:np.percentile(a,p,axis),

	"abs":lambda a:np.absolute(a),

	# running mean, via convolution
	"rmean":lambda a,n:np.convolve(a,np.ones(n)/n,"same"),

	# gradient
	"grad":lambda a,ax=None:np.gradient(a, axis=ax),
	
	"cos":lambda x:np.cos(x),
	"sin":lambda x:np.sin(x),

}

# set of names allowed for functions
ALLOWED_FUNC_NAMES = set(GLOBALS_EVAL.keys()) - {"__builtins__"}


# allowed types for constants
constant_type_whitelist = [
	int,
	float,
	complex,
	type(None),
]

# types of nodes allowed in expressions
node_type_whitelist = [

	# the expression itself
	ast.Expression,

	# function calls and keywords
	ast.Call,
	ast.keyword,

	# subscript for slicing
	ast.Subscript,
	ast.Tuple,
	ast.Slice,

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
	
	# # calling functions is forbidden
	# # if support is desired for mathematical functions which do not have
	# # dedicated operators, for instance cosine, then a whitelist of 
	# # function names will need to be established, and greated care paid
	# # to ensuring that the evaluation environment cannot contain any
	# # other functions.
	# ast.Call,
	# ast.keyword,

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

	# subscripting is slice-only
	ast.Index,
	ast.ExtSlice,

	# storing or deleting names is forbidden
	ast.Store,
	ast.Del,

]


ERR_NODE_NOT_WHITELISTED = "node has type {}, which is not whilelisted"
ERR_NODE_BLACKLISTED = "{} is a blacklisted node type"
ERR_CONSTANT_TYPE_NOT_WHITELISTED = "{} is not a whitelisted type for constants"
ERR_NAME_CONTEXT_NOT_LOAD = "Name node with id {} has ctx {}; must be Load"
ERR_FUNC_MUST_BE_NAME = "Call node has func with type {}; must be Name"
ERR_FUNC_NAME_NOT_ALLOWED = "Call node has func with unrecognized name {}"
ERR_VAR_NAME_FORBIDDEN = "non-function Name node has id {}, which is reserved"

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

		# the node's type must be whitelisted
		if type(node) not in node_type_whitelist:
			raise ValueError(ERR_NODE_NOT_WHITELISTED.format(type(node)))
		
		# additionally, its type must not be blacklisted
		if type(node) in node_type_blacklist:
			raise ValueError(ERR_NODE_BLACKLISTED.format(type(node)))

		# if it's a Constant, check that its value's type is whitelisted
		if isinstance(node,ast.Constant):
			if type(node.value) not in constant_type_whitelist:
				raise ValueError(ERR_CONSTANT_TYPE_NOT_WHITELISTED.format(type(node.value)))
		
		# if it's a Name, check that its ctx is Load
		if isinstance(node,ast.Name) and not (type(node.ctx) is ast.Load):
			raise ValueError(ERR_NAME_CONTEXT_NOT_LOAD.format(node.id,type(node.ctx)))

		# if it's a Call, check that the function is a Name and is allowed
		if isinstance(node,ast.Call):
			if type(node.func) is ast.Name:
				if node.func.id not in ALLOWED_FUNC_NAMES:
					raise ValueError(ERR_FUNC_NAME_NOT_ALLOWED.format(node.func.id))
			else:
				raise ValueError(ERR_FUNC_MUST_BE_NAME.format(type(node.func)))

		# if it's not a Call, inspect all child nodes of type Name, with the
		# assumption that they are expected inputs to the expression, and
		# ensure thet they aren't found in GLOBALS_EVAL
		else:
			for fieldname,value in ast.iter_fields(node):
				if isinstance(value, ast.Name) and value.id in GLOBALS_EVAL:
					raise ValueError(ERR_VAR_NAME_FORBIDDEN.format(value.id))

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
	# 
	# additionally, wrap with expects so that other code can easily
	# access the list of kwargs expected by the expression. note that
	# the keys in GLOBALS_EVAL are excluded, so that they are not
	# overwritten or mistakenly procured.
	@expects(get_names(expression) - GLOBALS_EVAL.keys())
	def eval_with_no_builtins(**items):
		return eval(compiled_expression, GLOBALS_EVAL, items)
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
