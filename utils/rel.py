"""
script for testing relative import functionality
"""

# the following work when importing this module from a script in /xrd-analysis/
# but not when running this script directly
#
# import utils.fileio as fileio
from . import fileio
from .analysis import procure

def test():
	print(fileio.ROOTKEY_DEFAULT_MODE)


