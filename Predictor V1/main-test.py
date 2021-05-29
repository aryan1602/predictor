# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 16:54:29 2021

@author: Aryan
"""

### Imports ###
# add imports - classes and defs
import sys
from predictor import predictRuns


"""
sys.argv[1] is the input test file name given as command line arguments

"""
runs = predictRuns('test.csv')
print("Predicted Runs: ", runs)