#!/usr/bin/python3

'''The main clc Python package

This package doesn't contain any code itself, but all the clc.mmm submodules
export their symbols here for convenience. So any function that can be called as
clc.mmm.fff() can be called as clc.fff() instead. The latter is preferred.

'''

# These will either wrap or import everything in ._clc, as needed
from .wrappers  import *

# Leaving these unimported. The user MUST "import clc.argparse_helpers" and "import clc.bag_interface"
#from .argparse_helpers import *
#from .bag_interface    import *
