#/usr/bin/env python

#=== Functions:
# OpenMM:
#   - create_omm_system
#       --> create topology
#       --> register force fields
#   - create_openMM_simulation
#       --> creates platform
#       --> creates integrator, simulation object
#
# TODO:
#   - parse gromacs, AA system
#

import os
import numpy as np
import parsevalidate
import parmed

VERY_VERBOSE = False
# === make an error class ===
class OMMError(Exception):
  def __init__(self, Msg):
    self.Msg = Msg
  def __str__(self):
    return str(self.Msg)

# === preamble ===
UNITS = "DimensionlessUnits"
CONSTRAINT_TOLERANCE = 1e-6
try:
    from simtk import openmm, unit
    #from simtk.unit import *
    from simtk.openmm import app
    import numpy as np
    import simtk.openmm.app.topology as topology
    import mdtraj

    import sys, time, os
    #========================================
    ###DEFINE ENERGY, LENGTH & MASS SCALES###
    #Try to use OMM's built-in unit definitions to the extent possible#
    #========================================
    epsilon  = 1.0 * unit.kilojoules_per_mole     #kJ/mole
    sigma = 1.0 * unit.nanometer                #nm
    tau = 1.0*unit.picoseconds                  #ps
    mass = tau**2/sigma**2*epsilon              #dalton = g/mole

    #N_av = 6.022140857*10**23 /unit.mole
    #kb = 1.380649*10**(-23)*unit.joules/unit.kelvin* N_av #joules/kelvin/mol
    N_av = unit.constants.AVOGADRO_CONSTANT_NA #6.02214179e23 / mole
    kB = unit.constants.BOLTZMANN_CONSTANT_kB #1.3806504e-23 * joule / kelvin
    Tref = epsilon/kB/N_av

    #from openmmtools.constants import ONE_4PI_EPS0 
    ONE_4PI_EPS0 = 138.935456       #in OMM units, 1/4pi*eps0, [=] kT/mole*nm/e^2
    q_factor = ONE_4PI_EPS0**-0.5    #s.t. electrostatic energy (1/4pi eps0)*q^2*qF^2/r = 1kB*Tref*Nav/mole = 1kJ/mole = 1epsilon

except:
    raise ImportError( "numpy, simtk, or mdtraj did not import correctly" )


# ============================== #

def createOMMSys(my_top, verbose = False):
    """
    Creates an OpenMM system
    Parameters
    ----------
    my_top : chemlib topology object, should also include general system parameters (i.e. temperatures, etc.)
    verbose : bool
        verbosity of output

    Returns
    -------
    system : OpenMM system
    topOMM : OpenMM topology
    topMDtraj : MDtraj topology

    Notes
    -----
    Todo:
    1) topology.BoxL
    """

    if UNITS != "DimensionlessUnits": raise ValueError( "Danger! OMM export currently only for dimensionless units, but {} detected".format(Units) )
    "takes in Sim 'Sys' object and returns an OpenMM System and Topology"
    print("=== OMM export currently uses LJ (Dimensionless) Units ===")
    print("mass: {} dalton".format(mass.value_in_unit(unit.dalton)))
    print("epsilon: {}".format(epsilon))
    print("tau: {}".format(tau))

    system_options = parsevalidate.parseSys( my_top.system_specs ) #my_top.system_specs["SystemOptions"]
    #try:
    #    parsevalidate.parseSys(system_options) 
    #    parsevalidate.parseFF(my_top)

    # --- box size ---
    Lx, Ly, Lz = parsevalidate.parseBox( system_options )
    
    #===================================
    # Create a System and its Box Size #
    #===================================
    print("=== Creating OMM System ===")
    system = openmm.System()

    # Set the periodic box vectors:
    box_edge = [Lx,Ly,Lz]
    box_vectors = np.diag(box_edge) * sigma
    system.setDefaultPeriodicBoxVectors(*box_vectors)


    #==================
    # Create Topology #
    #==================
    #Topology consists of a set of Chains 
???LINES MISSING
???LINES MISSING
???LINES MISSING
???LINES MISSING
???LINES MISSING
???LINES MISSING
???LINES MISSING
???LINES MISSING
???LINES MISSING
            reference = np.zeros([natoms,3])
            for res in top.residues:
                atomids = [atom.idx for atom in res.atoms]
                com = np.mean( pos[atomids,:], 0 )
                reference[atomids,:] = com[None,:]

            newpos = pos + reference* ( scalings - [1.,1.,1.] )
            simulation.context.setPositions(newpos)
            Enew = simulation.context.getState(getEnergy=True).getPotentialEnergy()

            #--- Monte Carlo Acceptance/Rejection ---
            print(tension*deltaArea)
            print(alpha*( (Lnew-L0)**2.0 - (Lold-L0)**2.0 ))
            w = Enew - Eold - tension*deltaArea + alpha*( (Lnew-L0)**2.0 - (Lold-L0)**2.0 )
            betaw = w/kBT
            print('... MC transition energy: {}'.format(betaw))
            if betaw > 0 and np.random.random_sample() > np.exp(-betaw):
                #Reject the step
                print('... Rejecting Step')
                simulation.context.setPeriodicBoxVectors( oldbox[0], oldbox[1], oldbox[2] )
                simulation.context.setPositions(pos)
            else:
                #Accept step
                print('... Accepting Step')
                TotalAcceptances = TotalAcceptances + 1
            TotalMCMoves += 1

            #--- Print out final state ---
            print('... box state after MC move:')
            print( simulation.context.getState().getPeriodicBoxVectors() )
            print('... acceptance rate: {}'.format(np.float(TotalAcceptances)/np.float(TotalMCMoves)))
            print('  ')

        #finish membrane barostating


        if tension_dimless > 0.0 and np.mod(iblock,100) != 0:
            continue
        else:
            #simulation.saveState(checkpointxml)
            positions = simulation.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions()
            app.PDBFile.writeFile(simulation.topology, positions, open(checkpointpdb, 'w')) 
            np.savetxt('boxdimensions.dat',box_sizes)

# ======================
# function to test OMM energies...



#/usr/bin/env python

#=== Functions:
# OpenMM:
#   - create_omm_system
#       --> create topology
#       --> register force fields
#   - create_openMM_simulation
#       --> creates platform
#       --> creates integrator, simulation object
#
# TODO:
#   - parse gromacs, AA system
#

import os
import numpy as np
import parsevalidate
import parmed

VERY_VERBOSE = False
# === make an error class ===
class OMMError(Exception):
  def __init__(self, Msg):
    self.Msg = Msg
  def __str__(self):
    return str(self.Msg)

# === preamble ===
UNITS = "DimensionlessUnits"
CONSTRAINT_TOLERANCE = 1e-6
try:
    from simtk import openmm, unit
    #from simtk.unit import *
    from simtk.openmm import app
    import numpy as np
    import simtk.openmm.app.topology as topology
    import mdtraj

    import sys, time, os
    #========================================
    ###DEFINE ENERGY, LENGTH & MASS SCALES###
    #Try to use OMM's built-in unit definitions to the extent possible#
    #========================================
    epsilon  = 1.0 * unit.kilojoules_per_mole     #kJ/mole
    sigma = 1.0 * unit.nanometer                #nm
    tau = 1.0*unit.picoseconds                  #ps
    mass = tau**2/sigma**2*epsilon              #dalton = g/mole

    #N_av = 6.022140857*10**23 /unit.mole
    #kb = 1.380649*10**(-23)*unit.joules/unit.kelvin* N_av #joules/kelvin/mol
    N_av = unit.constants.AVOGADRO_CONSTANT_NA #6.02214179e23 / mole
    kB = unit.constants.BOLTZMANN_CONSTANT_kB #1.3806504e-23 * joule / kelvin
    Tref = epsilon/kB/N_av

    #from openmmtools.constants import ONE_4PI_EPS0 
    ONE_4PI_EPS0 = 138.935456       #in OMM units, 1/4pi*eps0, [=] kT/mole*nm/e^2
    q_factor = ONE_4PI_EPS0**-0.5    #s.t. electrostatic energy (1/4pi eps0)*q^2*qF^2/r = 1kB*Tref*Nav/mole = 1kJ/mole = 1epsilon

except:
    raise ImportError( "numpy, simtk, or mdtraj did not import correctly" )


# ============================== #

def createOMMSys(my_top, verbose = False):
    """
    Creates an OpenMM system
    Parameters
    ----------
    my_top : chemlib topology object, should also include general system parameters (i.e. temperatures, etc.)
    verbose : bool
        verbosity of output

    Returns
    -------
    system : OpenMM system
    topOMM : OpenMM topology
    topMDtraj : MDtraj topology

    Notes
    -----
    Todo:
    1) topology.BoxL
    """

    if UNITS != "DimensionlessUnits": raise ValueError( "Danger! OMM export currently only for dimensionless units, but {} detected".format(Units) )
    "takes in Sim 'Sys' object and returns an OpenMM System and Topology"
    print("=== OMM export currently uses LJ (Dimensionless) Units ===")
    print("mass: {} dalton".format(mass.value_in_unit(unit.dalton)))
    print("epsilon: {}".format(epsilon))
    print("tau: {}".format(tau))

    system_options = parsevalidate.parseSys( my_top.system_specs ) #my_top.system_specs["SystemOptions"]
    #try:
    #    parsevalidate.parseSys(system_options) 
    #    parsevalidate.parseFF(my_top)

    # --- box size ---
    Lx, Ly, Lz = parsevalidate.parseBox( system_options )
    
    #===================================
    # Create a System and its Box Size #
    #===================================
    print("=== Creating OMM System ===")
    system = openmm.System()

    # Set the periodic box vectors:
    box_edge = [Lx,Ly,Lz]
    box_vectors = np.diag(box_edge) * sigma
    system.setDefaultPeriodicBoxVectors(*box_vectors)


    #==================
    # Create Topology #
    #==================
    #Topology consists of a set of Chains 
???LINES MISSING
???LINES MISSING
???LINES MISSING
???LINES MISSING
???LINES MISSING
???LINES MISSING
???LINES MISSING
???LINES MISSING
???LINES MISSING
            reference = np.zeros([natoms,3])
            for res in top.residues:
                atomids = [atom.idx for atom in res.atoms]
                com = np.mean( pos[atomids,:], 0 )
                reference[atomids,:] = com[None,:]

            newpos = pos + reference* ( scalings - [1.,1.,1.] )
            simulation.context.setPositions(newpos)
            Enew = simulation.context.getState(getEnergy=True).getPotentialEnergy()

            #--- Monte Carlo Acceptance/Rejection ---
            print(tension*deltaArea)
            print(alpha*( (Lnew-L0)**2.0 - (Lold-L0)**2.0 ))
            w = Enew - Eold - tension*deltaArea + alpha*( (Lnew-L0)**2.0 - (Lold-L0)**2.0 )
            betaw = w/kBT
            print('... MC transition energy: {}'.format(betaw))
            if betaw > 0 and np.random.random_sample() > np.exp(-betaw):
                #Reject the step
                print('... Rejecting Step')
                simulation.context.setPeriodicBoxVectors( oldbox[0], oldbox[1], oldbox[2] )
                simulation.context.setPositions(pos)
            else:
                #Accept step
                print('... Accepting Step')
                TotalAcceptances = TotalAcceptances + 1
            TotalMCMoves += 1

            #--- Print out final state ---
            print('... box state after MC move:')
            print( simulation.context.getState().getPeriodicBoxVectors() )
            print('... acceptance rate: {}'.format(np.float(TotalAcceptances)/np.float(TotalMCMoves)))
            print('  ')

        #finish membrane barostating


        if tension_dimless > 0.0 and np.mod(iblock,100) != 0:
            continue
        else:
            #simulation.saveState(checkpointxml)
            positions = simulation.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions()
            app.PDBFile.writeFile(simulation.topology, positions, open(checkpointpdb, 'w')) 
            np.savetxt('boxdimensions.dat',box_sizes)

# ======================
# function to test OMM energies...




