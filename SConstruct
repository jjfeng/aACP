#!/usr/bin/env scons

# Simulate data under various different settings and fit models

import os

from os.path import join
from nestly.scons import SConsWrap
from nestly import Nest
import SCons.Script as sc

# Command line options

sc.AddOption('--output', type='string', help="output folder", default='_output')

env = sc.Environment(
        ENV=os.environ,
        output=sc.GetOption('output'))

sc.Export('env')

env.SConsignFile()

flag = 'simulation_fine_control_bernoulli'
sc.SConscript(flag + '/sconscript', exports=['flag'])

flag = 'simulation_erm_bernoulli'
sc.SConscript(flag + '/sconscript', exports=['flag'])

flag = 'simulation_time_trends'
sc.SConscript(flag + '/sconscript', exports=['flag'])
