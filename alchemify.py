################################################################
# Kevin Shen, 2019                                             #
# kevinshen@ucsb.edu                                           #
#                                                              #
################################################################
## Helper functions to alchemize (small, < rcut) molecules in openMM
#  Currently implemented for handling standard OMM forcefields (i.e. the standard nonbonded force; support for custom nonbonded force to come later)#
#  Strategy is as delineated here: https://github.com/choderalab/openmmtools/issues/376
#  1) Turn of charges in standard NonbondedForce
#  2) Add new softCoreForce to turn off LJ interaction with other particles
#  3) Add new soluteCoulForce to turn on intramolecular electrostatics
#  4) Add new soluteLJForce to turn on intramolecular LJ interactions
# 
# Example Usage:
#import alchemify
#soluteIndices = []
#    soluteResidues = [soluteRes] #list of residues to alchemify
#    #parmed gromacs topology
#    for ir,res in enumerate(top.residues):
#        if ir in soluteResidues:
#            for atom in res.atoms:
#                soluteIndices.append(atom.idx)
#    print("Solute residue: {}".format([top.residues[ir].atoms for ir in soluteResidues]))
#    print("Solute Indices: {}".format(soluteIndices))
#    alch = alchemify.alchemist(system,lambdaLJ,lambdaQ)
#    alch.setupSolute(soluteIndices)

import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as u
import numpy as np

class alchemist:
    """A helper class to manage an alchemical simulation
    Notes
    -----
    Basic usage is:
    alch = alchemist()
    alch.setupSolute(soluteIndices)


    """
    def __init__(self,system,lambdaLJ=1.0, lambdaQ=1.0):
        """Initialization
        Parameters
        ----------
        system : openmm system

        Notes
        -----
        Todo:
        1) add type-checking
        """

        self.system = system
        self.q0s = [[0]]*self.system.getNumParticles()
        self.setupFF(lambdaLJ,lambdaQ)
        self.soluteInitialized = False

    def setupFF(self, lambdaLJ=1.0, lambdaQ=1.0):
        """Setup the Alchemical force fields, and store charge vector"""
        #We need to add a custom non-bonded force for the solute being alchemically changed
        #Will be helpful to have handle on non-bonded force handling LJ and coulombic interactions
        #Currently assumes only one Nonbonded force setup in the system
        NBForce = None
        for frc in self.system.getForces():
            if (isinstance(frc, mm.NonbondedForce)):
                NBForce = frc
        self.lambdaLJ = lambdaLJ
        self.lambdaQ = lambdaQ
        print("...alchemify: Using lambdaLJ: {}, lambdaQ: {}".format(self.lambdaLJ,self.lambdaQ))
        
        #Define the soft-core function for turning on/off LJ interactions
        #In energy expressions for CustomNonbondedForce, r is a special variable and refers to the distance between particles
        #All other variables must be defined somewhere in the function.
        #The exception are variables like sigma1 and sigma2.
        #It is understood that a parameter will be added called 'sigma' and that the '1' and '2' are to specify the combining rule.
        softCoreFunction = '4.0*lambdaLJ*epsilon*x*(x-1.0); x = (1.0/reff_sterics);'
        softCoreFunction += 'reff_sterics = (0.5*(1.0-lambdaLJ) + ((r/sigma)^6));'
        softCoreFunction += 'sigma=0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2)'
        #Define the system force for this function and its parameters
        SoftCoreForce = mm.CustomNonbondedForce(softCoreFunction)
        SoftCoreForce.addGlobalParameter('lambdaLJ', self.lambdaLJ) #Throughout, should follow convention that lambdaLJ=1.0 is fully-interacting state
        SoftCoreForce.addPerParticleParameter('sigma')
        SoftCoreForce.addPerParticleParameter('epsilon')

        #Will turn off electrostatics completely in the original non-bonded force
        #In the end-state, only want electrostatics inside the alchemical molecule
        #To do this, just turn ON a custom force as we turn OFF electrostatics in the original force
        ONE_4PI_EPS0 = 138.935456 #in kJ/mol nm/e^2
        soluteCoulFunction = '(1.0-(lambdaQ^2))*ONE_4PI_EPS0*charge/r;'
        soluteCoulFunction += 'ONE_4PI_EPS0 = %.16e;' % (ONE_4PI_EPS0)
        soluteCoulFunction += 'charge = charge1*charge2'
        SoluteCoulForce = mm.CustomNonbondedForce(soluteCoulFunction)
        #Note this lambdaQ will be different than for soft core (it's also named differently, which is CRITICAL)
        #This lambdaQ corresponds to the lambda that scales the charges to zero
        #To turn on this custom force at the same rate, need to multiply by (1.0-lambdaQ**2), which we do
        SoluteCoulForce.addGlobalParameter('lambdaQ', self.lambdaQ) 
        SoluteCoulForce.addPerParticleParameter('charge')
        
        #Also create custom force for intramolecular alchemical LJ interactions
        #Could include with electrostatics, but nice to break up
        #We could also do this with a separate NonbondedForce object, but it would be a little more work, actually
        soluteLJFunction = '4.0*epsilon*x*(x-1.0); x = (sigma/r)^6;'
        soluteLJFunction += 'sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)'
        SoluteLJForce = mm.CustomNonbondedForce(soluteLJFunction)
        SoluteLJForce.addPerParticleParameter('sigma')
        SoluteLJForce.addPerParticleParameter('epsilon')

        #=== Set other interaction parameters ===
        rcut = NBForce.getCutoffDistance() #default in nanometers
        nonbondedMethod = min(NBForce.getNonbondedMethod(),2)
        print("...alchemify: Cutoff method: {}".format(nonbondedMethod))
        print("...alchemify: compare to cutoff nonperiodic: {}".format(mm.CustomNonbondedForce.CutoffPeriodic))

        #Set other soft-core parameters as needed
        SoftCoreForce.setCutoffDistance(rcut)
        SoftCoreForce.setNonbondedMethod(nonbondedMethod)
        #SoftCoreForce.setUseSwitchingFunction(True)
        #SoftCoreForce.setSwitchingDistance(9.0*u.angstroms)
        SoftCoreForce.setUseLongRangeCorrection(True) 

        #Set other parameters as needed - note that for the solute force would like to set no cutoff
        #However, OpenMM won't allow a bunch of potentials with cutoffs then one without...
        #So as long as the solute is smaller than the cut-off, won't have any problems!
        SoluteCoulForce.setCutoffDistance(rcut)
        SoluteCoulForce.setNonbondedMethod(nonbondedMethod)
        #SoluteCoulForce.setUseSwitchingFunction(True)
        #SoluteCoulForce.setSwitchingDistance(9.0*u.angstroms)
        SoluteCoulForce.setUseLongRangeCorrection(False) #DON'T want long-range correction here!

        SoluteLJForce.setCutoffDistance(rcut)
        SoluteLJForce.setNonbondedMethod(nonbondedMethod)
        #SoluteLJForce.setUseSwitchingFunction(True)
        #SoluteLJForce.setSwitchingDistance(9.0*u.angstroms)
        SoluteLJForce.setUseLongRangeCorrection(False) 

        #=== Store the Functions and initial charges ===
        self.SoftCoreForce = SoftCoreForce
        self.SoluteCoulForce = SoluteCoulForce
        self.SoluteLJForce = SoluteLJForce
        self.NBForce = NBForce
        for ind in range(self.system.getNumParticles()):
            #Get current parameters in non-bonded force
            [charge, sigma, epsilon] = NBForce.getParticleParameters(ind)
            self.q0s[ind] = charge


    def setupSolute(self,soluteIndices):
        """Setup force fields and interaction groups to work with designated solute indices
        
        Parameters
        ----------
        soluteIndices : list
            list of atom.idx for atom in residue in solute molecule.
            note that getParticleParameters() is 0-indexed, but have to be careful to call atom.index instead of atom.in (1-based indexing, for pdb) 
        """
        assert not self.soluteInitialized, "Solute previously initialized, can't add force to system again"

        alchemicalParticles = set(soluteIndices)
        chemicalParticles = set(range(self.system.getNumParticles())) - alchemicalParticles
        
        #Loop over all particles and add to custom forces
        #As we go, will also collect full charges on the solute particles
        #AND we will set up the solute-solute interaction forces
        alchemicalCharges = [[0]]*len(soluteIndices)
        for ind in range(self.system.getNumParticles()):
            #Get current parameters in non-bonded force
            [charge, sigma, epsilon] = self.NBForce.getParticleParameters(ind)
            #Make sure that sigma is not set to zero! Fine for some ways of writing LJ energy, but NOT OK for soft-core!
            if sigma/u.nanometer == 0.0:
                newsigma = 0.3*u.nanometer #This 0.3 is what's used by GROMACS as a default value for sc-sigma
            else:
                newsigma = sigma
            #Add the particle to the soft-core force (do for ALL particles)
            self.SoftCoreForce.addParticle([newsigma, epsilon])
            #Also add the particle to the solute only forces
            self.SoluteCoulForce.addParticle([charge])
            self.SoluteLJForce.addParticle([sigma, epsilon])
            #If the particle is in the alchemical molecule, need to set it's LJ interactions to zero in original force
            if ind in soluteIndices:
                newcharge = self.lambdaQ*self.q0s[ind]
                self.NBForce.setParticleParameters(ind, newcharge, sigma, epsilon*0.0)
                #And keep track of full charge so we can scale it right by lambda
                alchemicalCharges[soluteIndices.index(ind)] = charge

        #Now we need to handle exceptions carefully
        for ind in range(self.NBForce.getNumExceptions()):
            [p1, p2, excCharge, excSig, excEps] = self.NBForce.getExceptionParameters(ind)
            #For consistency, must add exclusions where we have exceptions for custom forces
            self.SoftCoreForce.addExclusion(p1, p2)
            self.SoluteCoulForce.addExclusion(p1, p2)
            self.SoluteLJForce.addExclusion(p1, p2)

        #Only compute interactions between the alchemical and other particles for the soft-core force
        self.SoftCoreForce.addInteractionGroup(alchemicalParticles, chemicalParticles)

        #And only compute alchemical/alchemical interactions for other custom forces
        self.SoluteCoulForce.addInteractionGroup(alchemicalParticles, alchemicalParticles)
        self.SoluteLJForce.addInteractionGroup(alchemicalParticles, alchemicalParticles)

        #Now add forces to system. Shouldn't be undone unless we change a system force field in context
        self.system.addForce(self.SoftCoreForce)
        self.system.addForce(self.SoluteCoulForce)
        self.system.addForce(self.SoluteLJForce)
       
        self.chemicalParticles = chemicalParticles
        self.alchemicalParticles = alchemicalParticles
        self.alchemicalCharges = alchemicalCharges
        self.soluteInitialized = True

    #def updateState(self,context,lambdaLJ,lambdaQ):


