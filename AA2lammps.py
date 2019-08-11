#Covert AA to MD, scaing into the length scale unit l=rhow^(-1/3)
import mdtraj
import numpy as np
import sys

trajfile = sys.argv[1]
topfile = sys.argv[2]

sigma = 0.3107408102611562*10 #the length scale convention I'm using, rhow^(-1/3)
L = 4

traj = mdtraj.load(trajfile, top=topfile)
#traj.image_molecules(inplace=True)

traj.xyz /= sigma
#traj.unitcell_lenths /= sigma
traj.unitcell_vectors /= sigma

traj.save('.'.join(trajfile.split('.')[:-1]) + '.lammpstrj')
