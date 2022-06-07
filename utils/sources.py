"""
access for common data about radioactive sources and their properties
"""

# __all__ = ...
__author__ = "Brunel Odegard"
__version__ = "0.0"

from utils.data import invert_dictionary




# radioactive sources and their emissions
# 
# data in this section is taken from:
# http://nucleardata.nuclear.lu.se/toi/index.asp
# 
# emissions have been excluded which have no listed value for i,
# or for which the value is less than 1e-6 (which corresponds to
# i(%) < 1e-4)
# 
# peaks which have no uncertainty listed for one of their properties
# have had it filled in with (5)

class source(object):
	
	def __init__(self, name, emissions, half_life_years=None):
		
		self.__name = name
		self.__half_life_years = half_life_years

		self.__emissions = tuple(emission(self, *_) for _ in emissions)

	def __str__(self):
		return self.name

	@property
	def name(self):
		return self.__name

	@property
	def half_life_years(self):
		return self.__half_life_years

	@property
	def emissions(self):
		return self.__emissions




class emission(object):
	__STR_FORMAT = "emission of source {parent}, mode {mode}, \
I {i_pct} ({i_pct_uncertainty}) %, energy {energy_kev} \
({energy_kev_uncertainty}) KeV"
 
	
	def __init__(self, parent, mode, energy_kev, energy_kev_uncertainty, energy_kev_sf, i_pct, i_pct_uncertainty, i_pct_sf):
		self.__parent = parent
		self.__mode = mode
		
		self.__i_pct = float(i_pct)
		if i_pct_sf is None:
			if '.' not in i_pct:
				raise ValueError("cannot automatically calculate significant figures without decimal point in representation")
			i_pct_sf = len(i_pct.rpartition('.')[2])
		self.__i_pct_sf = i_pct_sf
		self.__i_pct_uncertainty = float(i_pct_uncertainty) * (10 ** -i_pct_sf)
		
		self.__energy_kev = energy_kev
		if energy_kev_sf is None:
			if '.' not in energy_kev:
				raise ValueError("cannot automatically calculate significant figures without decimal point in representation")
			energy_kev_sf = len(energy_kev.rpartition('.')[2])
		self.__energy_kev_uncertainty = float(energy_kev_uncertainty) * (10 ** -energy_kev_sf)

	def __str__(self):
		return self.__STR_FORMAT.format(
			parent = self.__parent,
			mode = self.__mode,
			i_pct = self.__i_pct,
			i_pct_uncertainty = self.__i_pct_uncertainty,
			energy_kev = self.__energy_kev,
			energy_kev_uncertainty = self.__energy_kev_uncertainty,
		)

	@property
	def parent(self):
		return self.__parent

	@property
	def mode(self):
		return self.__mode

	@property
	def i_pct(self):
		return self.__i_pct
	@property
	def i_pct_uncertainty(self):
		return self.__i_pct_uncertainty
	@property
	def i_pct_sf(self):
		return self.__i_pct_sf

	@property
	def energy_kev(self):
		return self.__energy_kev
	@property
	def energy_kev_uncertainty(self):
		return self.__energy_kev_uncertainty
	@property
	def energy_kev_sf(self):
		return self.__energy_kev_sf

	# shorthand
	# ""  -> pct
	# ""  -> kev
	# e   -> energy
	# err -> uncertainty
	e      = energy_kev
	energy = energy_kev
	e_err      = energy_kev_uncertainty
	energy_err = energy_kev_uncertainty
	i      = i_pct
	i_err  = i_pct_uncertainty


Ba133 = source(
	"Ba133",
	[
		["e",  "53.161" ,  "1", None,  "2.199", "22", None],
		["e",  "79.6139", "26", None,  "2.62" ,  "6", None],
		["e",  "80.9971", "14", None, "34.06" , "27", None],
		["e", "160.613" ,  "8", None,  "0.645",  "8", None],
		["e", "223.234" , "12", None,  "0.450",  "4", None],
		["e", "276.398" ,  "2", None,  "7.164", "22", None],
		["e", "302.853" ,  "1", None, "18.33" ,  "6", None],
		["e", "356.017" ,  "2", None, "62.05" , "19", None],
		["e", "383.851" ,  "3", None,  "8.94" ,  "3", None],
	],
	10.51,
)

Am241 = source(
	"Am241",
	[
		["a",  "26.3448",   "2", None,  "2.40"   ,  "2", None],
		["a",  "32.183" ,   "5", None,  "0.0174" ,  "4", None],
		["a",  "33.1964",   "3", None,  "0.126"  ,  "3", None],
		["a",  "42.73"  ,   "5", None,  "0.0055" , "11", None],
		["a",  "43.423" ,  "10", None,  "0.073"  ,  "8", None],
		["a",  "55.56"  ,   "2", None,  "0.0181" , "18", None],
		["a",  "57.85"  ,   "5", None,  "0.0052" , "15", None],
		["a",  "59.5412",   "2", None, "35.9"    ,  "4", None],
		["a",  "69.76"  ,   "3", None,  "0.0029" ,  "4", None],
		["a",  "98.97"  ,   "2", None,  "0.0203" ,  "4", None],
		["a", "102.98"  ,   "2", None,  "0.0195" ,  "4", None],
		["a", "123.01"  ,   "2", None,  "0.00100",  "3", None],
		["a", "125.30"  ,   "2", None,  "0.00408",  "9", None],
	],
	432.2,
)

Co57 = source(
	"Co57",
	[
		["e",  "14.41300", "15", None,  "9.16"  , "15", None],
		["e", "122.0614" ,  "4", None, "85.60"  , "17", None],
		["e", "136.4743" ,  "5", None, "10.68"  ,  "8", None],
		["e", "339.54"   , "18", None,  "0.0139",  "3", None],
		["e", "352.36"   ,  "1", None,  "0.0132",  "3", None],
		["e", "366.75"   ,  "1", None,  "0.0013",  "3", None],
		["e", "569.92"   ,  "4", None,  "0.017" ,  "1", None],
		["e", "692.03"   ,  "2", None,  "0.157" ,  "9", None],
		["e", "706.40"   , "20", None,  "0.0253",  "5", None],
	],
	0.74463,
)

Cd109 = source(
	"Cd109",
	[],
	1.267,
)

Cs137 = source(
	"Cs137",
	[],
	None,
)

Na22 = source(
	"Na22",
	[],
	None,
)

Mn54 = source(
	"Mn54",
	[],
	None,
)

Co60 = source(
	"Co60",
	[],
	None,
)

Pb210 = source(
	"Pb210",
	[],
	None,
)




# ease of access objects

source_id_to_name = {
	0:"None",
	1:"Cs137",
	2:"Ba133",
	3:"Na22",
	4:"Mn54",
	5:"Cd109",
	6:"Co57",
	7:"Co60",
	8:"Pb210",
	9:"Am241",
}

source_name_to_id = invert_dictionary(source_id_to_name)
