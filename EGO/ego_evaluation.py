# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:48:23 2015

@author: wangronin
"""

import os
from mpi4py import MPI

pid = os.getpid()

comm = MPI.Comm.Get_parent()
comm_self = MPI.COMM_WORLD

fitness = comm.bcast(None, root=0)
data = comm.scatter(None, root=0)

index, pars = data

MPI.

# Fitness evaluation in parallel...
y = fitness(pars)

if not isinstance(y, int):
    y = y[0]

# Synchronization...
comm.Barrier()

# Gathering the fitted kriging model back
results = {'index': index, 'y': y}
          
comm.gather(results, root=0)

comm.Disconnect()