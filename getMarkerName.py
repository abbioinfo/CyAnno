#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 13:59:59 2021

@author: abhinav kaushik
This script lets to known the marker name o be used in CyAnno for a given FCS file. This helps in choosing the 
list of relevant marker names. From the output of this script, just select the markers you are really interested in 
as lineage markers, as add to CyAnno.py in same syntax.


usage:
    python getMarkerName.py [Path to FCS file]
    
Example:
    python getMarkerName.py ./POISED/LiveCells/export_P015-000-Un.fcs
    
Output:
    ['190BCKG', '138Ba', 'CD16', 'beads', 'beads', None, '133Cs', 'IFNg', 'CD69', 'CXCR3', 
    'IL.17', 'LAP', 'CD27', 'CD40L', 'PD1', 'CD123', 'CD45RA', None, 'CD28', 'GPR15', 'HLA.DR', 
    'CD33', 'CD14', 'CD127', 'CD86', 'LiveDead', 'dna1', 'dna2', 'TCRgd', 'CD19', 'CD49b', 
    'IL.4', 'CD4', 'CD8', 'CD38', 'LAG3', None, '208Pb', 'BC', 'BC', 'BC', 'BC', 'BC', 'BC', 
    'OX40', None, 'CD20', 'CCR4', 'IL9', 'CD3', 'CD11c', 'CCR7', '131Xe', 'CD57', 'IL.10', 
    'integrin', 'CD25', 'CD56', 'CLA', None]

"""
import os
import fcsparser
import pandas as pa
import re 
import sys

#f = './POISED/LiveCells/export_P015-000-Un.fcs'
f = str(sys.argv[1])

if os.path.exists(f):
    match = re.search(r'fcs$', f)
    if match: ## if file is FCS
        panel, exp = fcsparser.parse(f, reformat_meta=True)
        desc = panel['_channels_']['$PnS'] ## marker name instead of meta name
        desc = desc.str.replace('.*_', '', regex=True) ## cleaning marker name 
        desc = desc.str.replace('-', '.', regex=True).tolist() ## cleaning marker name 
        print(desc)
    else:
        print("Please provide valid FCS file with .fcs extention")

