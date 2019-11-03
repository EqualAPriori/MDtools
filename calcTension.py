# Assuming Salt Free SDS system
# my netcdf files got corrupted... have to jerry-rig to use pdb files instead (10x less data, ~180 frames)
# note: pdb is reported to only 5~6 sig figs! i.e. limited resolution for doing test area rescaling,
# where roundoff is same order as small perturbations

# System stuff
from sys import stdout
import sys,getopt
import time
import numpy as np
import argparse as ap

# OpenMM Imports
import simtk.openmm as mm
import simtk.openmm.app as app

# ParmEd & MDTraj Imports
from parmed import gromacs
gromacs.GROMACS_TOPDIR = "/home/kshen/mylib/ff"
from parmed.openmm.reporters import NetCDFReporter
from parmed import unit as u
import parmed as pmd
import mdtraj
from pymbar import timeseries
#import MDAnalysis as mda

# Command line inputs
ewldTol = 1e-6

#=== parse: ===
parser = ap.ArgumentParser(description="Calculate surface tension using perturbation. Not suitable for square well potential.")
parser.add_argument('-top', type=str, help="gromacs topology file")
parser.add_argument('-traj', type=str, help="trajectory file, expect includes box info")
parser.add_argument('-box', type=str, help="box file")
parser.add_argument('-pdb', type=str, help="pdb file to read in trajectory")
parser.add_argument('-outprefix', default='', type=str, help="prefix for output files")
parser.add_argument('-dAfrac', type=float, help="fractional perturbation of area")
parser.add_argument('-ewldTol', default=1e-6, type=float, help="ewld tolerance. default is 1e-6")
parser.add_argument('-axis', default=2, choices=[0,1,2], type=int, help="long axis, 0:x, 1:y, 2:z")
parser.add_argument('-stride',default=1, type=int, help="stride for reading trajectory")
parser.add_argument('-volumeChange', action='store_true', help="whether or not the box size changes")
args = parser.parse_args()

ewldTol = args.ewldTol
dAfrac = args.dAfrac
axis = args.axis
print(sys.argv)
print(args)

# files
coord_file  = args.traj # 'output.dcd'
top_file = args.top # gromacs topology file
box_file = args.box # 'box.gro', in future read from trajectory
pdb_file = args.pdb # pdb file to read in trajectory
defines = {}

# Test area parameters
#dAfrac   = 0.00001
scaleZ1     = 1.0/(1.0 + dAfrac)
scaleXY1    = (1.0 + dAfrac)**0.5
scaleZ2     = 1.0/(1.0 - dAfrac)
scaleXY2    = (1.0 - dAfrac)**0.5

# === Parameters, assumed, eventually should read in from mdparse ===
T               = 298.
NPT             = False
useLJPME        = True  #True
LJcut           = 12    #Angstroms
tail            = False#False
Temp            = T     #K
Pressure        = 1     #bar
barostatfreq    = 25    #time steps
fric            = 1.0   #1/ps
dt              = 2.0   #fs
rigidH2O        = True

# === Start making system ===
gro = gromacs.GromacsGroFile.parse(box_file)
pdb = pmd.load_file(pdb_file)

box = gro.box #note stored in Angstrom
if axis == 2:
    scaling1 = [scaleXY1, scaleXY1, scaleZ1, 1.0, 1.0, 1.0]
    scaling2 = [scaleXY2, scaleXY2, scaleZ2, 1.0, 1.0, 1.0]
elif axis == 1:
    scaling1 = [scaleXY1, scaleZ1, scaleXY1, 1.0, 1.0, 1.0]
    scaling2 = [scaleXY2, scaleZ2, scaleXY2, 1.0, 1.0, 1.0]
elif axis == 0:
    scaling1 = [scaleZ1, scaleXY1, scaleXY1, 1.0, 1.0, 1.0]
    scaling2 = [scaleZ2, scaleXY2, scaleXY2, 1.0, 1.0, 1.0]
box1 = gro.box*scaling1
box2 = gro.box*scaling2


top = gromacs.GromacsTopologyFile(top_file, defines=defines, box=box)
top1= gromacs.GromacsTopologyFile(top_file, defines=defines, box=box1)
top2= gromacs.GromacsTopologyFile(top_file, defines=defines, box=box2)
#top.positions = pdb.positions


def makeSystem(topology):#, Temp, useLJPME, LJcut, tail=True, NPT=False, Pressure=1, barostatfreq=25, rigidH2O=True)
    if useLJPME:
        nbm = mm.NonbondedForce.LJPME
    else:
        nbm = mm.NonbondedForce.PME

    system = topology.createSystem(nonbondedMethod=app.PME,
                nonbondedCutoff=LJcut*u.angstroms,
                constraints=app.HBonds, rigidWater=rigidH2O, ewaldErrorTolerance=ewldTol) #default ewTol=5e-4

    ftmp = [f for ii, f in enumerate(system.getForces()) if isinstance(f,mm.NonbondedForce)]
    fnb = ftmp[0]
    fnb.setNonbondedMethod(nbm)
    print("Nonbonded method, use LJPME: {}".format(fnb.getNonbondedMethod()) )

    if (not tail) or (useLJPME):
        print("Turning off tail correction...")
        fnb.setUseDispersionCorrection(False)
        print("Check dispersion flag: {}".format(fnb.getUseDispersionCorrection()) )

    if NPT:
        barostat = mm.MonteCarloBarostat(Pressure*u.bar, Temp*u.kelvin, barostatfreq)
        system.addForce(barostat)
         
    integrator = mm.LangevinIntegrator(
		    Temp*u.kelvin,       # temperature of heat bath
		    fric/u.picoseconds,  # friction coefficient
		    dt*u.femtoseconds, # time step
    )
    return system,integrator

system0, integrator0 = makeSystem(top)#, Temp=Temp, useLJPME=useLJPME, LJcut=LJcut, tail=tail, NPT=NPT, Pressure=Pressure, barostatfreq=barostatfreq, rigidH2O=rigidH2O)
system1, integrator1 = makeSystem(top1)
system2, integrator2 = makeSystem(top2)
properties = {'OpenCLPrecision': 'double'}
platform = mm.Platform.getPlatformByName('OpenCL')
platform.setPropertyDefaultValue('Precision','double')
sim0 = app.Simulation(top.topology,  system0, integrator0, platform)
sim1 = app.Simulation(top1.topology, system1, integrator1, platform)
sim2 = app.Simulation(top2.topology, system2, integrator2, platform)
#sim0 = app.Simulation(top.topology,  system0, integrator0, platformProperties=properties)
#sim1 = app.Simulation(top1.topology, system1, integrator1, platformProperties=properties)
#sim2 = app.Simulation(top2.topology, system2, integrator2, platformProperties=properties)

#simulation.context.setPositions(pdb.positions)
#simulation.context.applyConstraints(1e-12)


# === CALCULATE ENERGIES ===
def getEnergy(simulation,positions):
    simulation.context.setPositions(positions)
    state = simulation.context.getState( getEnergy=True )
    return state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)

traj = mdtraj.load(coord_file,top=pdb_file, stride=args.stride)
energies = np.zeros([traj.n_frames, 3])

print("===== starting energy calculations =====")
start = time.time()
for iframe,ts in enumerate(traj):
    if( np.mod(iframe,10)==0 ):
        print("On frame {}".format(iframe))
    
    if args.volumeChange:
        box = ts.openmm_boxes(0)
        print("Changing box... {}".format(box))
        sim0.context.setPeriodicBoxVectors(box[0], box[1], box[2])
        sim1.context.setPeriodicBoxVectors(box[0]*scaling1[0], box[1]*scaling1[1], box[2]*scaling1[2])
        sim2.context.setPeriodicBoxVectors(box[0]*scaling2[0], box[1]*scaling2[1], box[2]*scaling2[2])
        print(sim2.context.getState().getPeriodicBoxVectors())

    pos = ts.xyz[0] #pdb in A, but openMM default seems to be in nm
    en0 = getEnergy(sim0,pos)

    natoms = len(top.atoms)
    reference = np.zeros([natoms,3])
    for res in top.residues:
        atomids = [atom.idx for atom in res.atoms]
        com = np.mean( pos[atomids,:], 0 )
        reference[atomids,:] = com[None,:]


    #calculate larger box
    if axis == 2:
        newpos = pos + reference*np.array( [scaleXY1-1.0, scaleXY1-1.0, scaleZ1-1.0] )    
    elif axis == 1:
        newpos = pos + reference*np.array( [scaleXY1-1.0, scaleZ1-1.0, scaleXY1-1.0] )    
    elif axis == 0:
        newpos = pos + reference*np.array( [scaleZ1-1.0, scaleXY1-1.0, scaleXY1-1.0] )    
    en1    = getEnergy(sim1,newpos)    

    #calculate smaller box
    if axis == 2:
        newpos = pos + reference*np.array( [scaleXY2-1.0, scaleXY2-1.0, scaleZ2-1.0] )
    elif axis == 1:
        newpos = pos + reference*np.array( [scaleXY2-1.0, scaleZ2-1.0, scaleXY2-1.0] )
    elif axis == 0:
        newpos = pos + reference*np.array( [scaleZ2-1.0, scaleXY2-1.0, scaleXY2-1.0] )
    en2    = getEnergy(sim2, newpos)

    energies[iframe,:] = np.array([en0, en1, en2])

end = time.time()
print("Took {} seconds".format(end-start))

#np.savetxt('energies.txt',energies,delimiter=',',header="dA% = {} area change; Col0: Reference, Col1: +dA, Col2: -dA".format(dAfrac))


# === detect correlations in energies ===
[nequil,g,Neff_max] = timeseries.detectEquilibration(energies[:,0])
indices = timeseries.subsampleCorrelatedData(energies[:,0], g=g)


# === ESTIMATE SURFACE TENSION ===
# finite differencing with area change 2dA, account for two interfaces
kb  = 1.3806504e-23     #1.38064852e-23    #J/K
kT  = kb*Temp           #J
Nav = 6.02214179e23     #6.022e23
kTkJmol = kT*Nav/1000   #kT in kJ/mol

dE1 = energies[indices,1] - energies[indices,0]
dE2 = energies[indices,2] - energies[indices,0]

BdE1 = dE1/kTkJmol
BdE2 = dE2/kTkJmol

dF1 = -kT * np.log( np.exp(-BdE1).mean() ) #J
dF2 = -kT * np.log( np.exp(-BdE2).mean() ) #J

dF1error = kT/np.abs(np.exp(-BdE1).mean())*np.exp(-BdE1).std()/np.sqrt(np.size(BdE1))
dF2error = kT/np.abs(np.exp(-BdE2).mean())*np.exp(-BdE2).std()/np.sqrt(np.size(BdE2))


if axis == 2:
    da = 2 * dAfrac * (gro.box[0]/10.0) * (gro.box[1]/10.0)  # nm
elif axis == 1:
    da = 2 * dAfrac * (gro.box[0]/10.0) * (gro.box[2]/10.0)  # nm
elif axis == 0:
    da = 2 * dAfrac * (gro.box[1]/10.0) * (gro.box[2]/10.0)  # nm

tension = (dF1 - dF2)/2/da * 1e18 #convert from J/nm^2 to  J/m^2 aka N/m
tensionError = np.sqrt(dF1error**2 + dF2error**2)/2/da*1e18
print('tension: {} +/- {}N/m'.format(tension, tensionError)) 

np.savetxt('energies.txt',energies,delimiter=',',
        header="dA% = {} area change, gamma={} N/m; Col0: Reference, Col1: +dA, Col2: -dA".format(dAfrac,tension))

import datetime
with open('results.txt',"a") as f:
    f.write('{}\n'.format(datetime.datetime.now()) )
    f.write('{}\n'.format(sys.argv))
    f.write('T={}K, dA % = {} area change, gamma = {}+/-{} N/m\n'.format(Temp,dAfrac, tension, tensionError)),
    f.write('dFfwd = {}+/-{}J/mol, dFback={}+/-{}J/mol\n'.format(dF1, dF1error, dF2, dF2error))
    f.write('or dFfwd = {}+/-{}kT, dFback = {}+/-{}kT\n'.format(dF1/kT, dF1error/kT, dF2/kT, dF2error/kT))



