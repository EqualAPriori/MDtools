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
    #Each Chain contains a set of Residues, 
    #and each Residue contains a set of Atoms.
    #We take the topology from the `sim System` object. Currently we do 1 residue per chain, and residue is effectively the Molecule class in sim.
    print("--- Creating Topology ---")
    # --- first get atom types and create dummy elements for openMM ---
    sim_atom_types = [value for value in my_top.atom_types.values()]
    elements = {}
    atom_type_index = {}
    atom_name_map = []
    sim_atom_type_map = {}
    for ia,atom_type in enumerate(sim_atom_types):
        newsymbol = 'Z{}'.format(ia)
        if newsymbol not in app.element.Element._elements_by_symbol:
            elements[atom_type.name]=app.element.Element(200+ia, atom_type.name, newsymbol, atom_type.mass * mass)
        else:
            elements[atom_type.name]=app.element.Element.getBySymbol(newsymbol)

        atom_type_index[atom_type.name] = ia
        atom_name_map.append(atom_type.name)
        sim_atom_type_map[atom_type.name] = atom_type

    # --- next get the molecules and bondlist ---
    residues = []
    molecule_atom_list = {}
    molecule_bond_list = {}
    for mname, m in my_top.molecule_types.items():
        molname = m.name
        residues.append(molname)
        #molecule_atom_list[molname] = [my_top.Atoms[ia] for ia in my_top.atoms_in_mol[im]]  #stores names of atoms in molecule
        molecule_atom_list[molname] = [a.name for r in m for a in r]  #stores names of atoms in molecule
        molecule_bond_list[molname] = [ [b[0],b[1]] for b in m.bonds ]
        #molecule_bond_list[molname] = [ [b[0], b[1]] for b in my_top.bonds_in_mol[im] ]    #stores bond indices in molecule

    # --- aggregate stuff, and add atoms to omm system ---
    # Particles are added one at a time
    # Their indices in the System will correspond with their indices in the Force objects we will add later
    atom_list = [aname for aname in my_top.atoms]     #stores atom names
    mol_list = [mname for mname in my_top.molecules]  #stores molecule names
    for a in my_top.atoms:
        system.addParticle(a.mass * mass)
    print("Total number of particles in system: {}".format(system.getNumParticles()))
 

    # --- the actual work of creating the topology ---
    constrained_bonds = False
    
    top = app.topology.Topology()
    mdtrajtop = app.topology.Topology() #so that later can make molecules whole
    constrained_bonds = False
    constraint_lengths = []
    for im,mol in enumerate(my_top.molecules):
        chain = top.addChain() #Create new chain for each molecule
        res = top.addResidue(mol.name, chain) #don't worry about actually specifying residues within a molecule
        mdt_chain = mdtrajtop.addChain() #Create new chain for each molecule
        mdt_res = mdtrajtop.addResidue(mol.name, mdt_chain)
        
        # add the atoms
        atoms_in_this_res = []
        mdt_atoms_in_this_res = []
        #atomsInThisMol = [my_top.AtomTypes[ my_top.Atoms[ia] ] for ia in my_top.AtomsInThisMol[im]] #list of atom objects

        for ir,r in enumerate(mol):
            for atom_ind,a in enumerate(r):
                el = elements[a.name]
                if atom_ind > 0:
                    previous_atom = atom
                atom = top.addAtom( a.name, el, res )
                mdt_atom = mdtrajtop.addAtom( a.name, mdtraj.element.Element.getByAtomicNumber(atom_type_index[a.name]), mdt_res ) #use a dummy element by matching atomic number == cgAtomTypeIndex
                atoms_in_this_res.append(atom)
                mdt_atoms_in_this_res.append(mdt_atom)

        # add the bonds
        for bond_site in mol.bonds: #mol.bonds stores the intramolecular site indices of the bond
            a1 = atoms_in_this_res[bond_site[0]] #the atoms_in_this_res has the current, newly added atoms of this residue
            a2 = atoms_in_this_res[bond_site[1]]
            if verbose: print("Adding bond ({},{}), absolute index {},{}".format(bond_site[0],bond_site[1],a1.index,a2.index))
            #top.addBond( atoms_in_this_res[bond.SType1.AInd], atoms_in_this_res[bond.SType2.AInd] )
            newbond = top.addBond( a1, a2 )
            mdtrajtop.addBond( a1, a2 ) #don't worry about adding constraint to the mdtraj topology

            if bond_site.rigid:
                constrained_bonds = True
                if verbose: print("Adding rigid constraint for {},{}".format(a1,a2))
                system.addConstraint( a1.index, a2.index, bond.length ) #constraint uses 0-based indexing of atoms
                constraint_lengths.append(bond.length)
            else:
                constraint_lengths.append(0.0)

    print("Total number of constraints: {}".format(system.getNumConstraints()))   


    #====================
    ##CREATE FORCEFIELD##
    #====================
    #Currently only allow for some restricted interactions
    #a) harmonic bonds
    #b) Gaussian repulsion
    #c) external force

    #TODO:
    #1) spline/tabulated [would then need to implement a library of function evaluations...]

    #============================
    #create Bonded Interactions #
    #============================
    #Currently only support harmonic bonds, pairwise bonds
    bond_ffs = parsevalidate.parseBond( my_top.system_specs )
    if bond_ffs: #found bonds
        print("---> Found bonded interactions")
        bonded_force = openmm.HarmonicBondForce()
    #Check for bonds

    #Add bonds
    for mol in my_top.molecules:
        for bond in mol.bonds:
            ai = mol.atoms[ bond[0] ]
            aj = mol.atoms[ bond[1] ]

            applicable_potentials = []
            for potential in bond_ffs:
                if potential[0] == 'harmonic':
                    atype_name1, atype_name2, bond_length, K = potential[1:]
                    #print('{},{},{},{}'.format(ai.name,aj.name,atype_name1,atype_name2))
                    if (ai.name, aj.name) in [ (atype_name1, atype_name2) , (atype_name2, atype_name1) ]:
                        applicable_potentials.append(potential)
                        temp_label = 'for sites {},{} in mol {}: absolute atom index {},{}'.format(bond[0],bond[1],mol.name,ai.ind,aj.ind)
                        if K == np.inf:
                            bond.length = bond_length
                            if verbose: print("adding rigid constraint for {}, overriding harmonic bond".format(temp_label))
                            #if bond_site.rigid: #should be True
                            constrained_bonds = True
                            system.addConstraint( ai.ind, aj.ind, bond.length ) #constraint uses 0-based indexing of atoms
                            constraint_lengths.append(bond.length) 
                        else:
                            bonded_force.addBond( ai.ind, aj.ind, bond_length*sigma, 2.0*K*epsilon/sigma/sigma)
                            if verbose: print("adding harmonic bond {}(r-{})^2 for {}".format(K,bond_length, temp_label))
                else:
                    raise ValueError('Bond type {} has not been implemented in openMM export yet'.format(potential))
            # end for potential in bond_ffs
            if not applicable_potentials and not bond.rigid:
                raise OMMError("no bonded potential nor constraint found for bond {} in mol {}".format(bond,mol.name))
    if bond_ffs:                
        system.addForce(bonded_force)


    #=====================================================
    #create custom nonbonded force: Gaussian + ShortRange#
    #=====================================================
    #--- iterate over all atom pair types ---
    #for aname1 in atomNameMap:
    #    for aname2 in atomNameMap:
    #        AType1 = simAtomTypeMap[aname1]
    #        AType2 = simAtomTypeMap[aname2]
    #        #AType1 = [atype for atype in Sys.World.AtomTypes if atype.Name==aname1]
    #        #AType2 = [atype for atype in Sys.World.AtomTypes if atype.Name==aname2]
    #        print([AType1,AType2])
    #
    #TODO: spline aggregation of potentials
    #
    ag, u0, dist0, rcut, indv_gaussians = parsevalidate.parseGaussian( my_top.system_specs )
    if indv_gaussians:
        print("---> Found individual gaussians interactions")
        print("CAUTION, NOT IMPLEMENTED YET!")
    if len(ag) != 0 and len(u0) !=0 and len(dist0) != 0 and rcut is not None:
        print("---> Found gaussians interactions matrix")
        nspec = len(my_top.atom_types)
        if ag.shape[0] != nspec:
            raise ValueError("Gaussian Base widths matrix incorrectly sized")
        if u0.shape[0] != nspec:
            raise ValueError("Gaussian Base prefactor matrix incorrectly sized")

        print("... Detected cutoff: {}".format(rcut))
        GlobalCut = rcut
        nonbondedMethod = 2


        print("... u0 matrix:\n{}".format(u0))
        print("... ag matrix:\n{}".format(ag))
        epsmatrix = np.zeros(ag.shape)
        sigmatrix = np.zeros(ag.shape)
        Bmatrix   = u0/(4*np.pi*ag*ag)**1.5
        kappamatrix = 1/4/ag/ag
        dist0matrix = dist0


        energy_function =  'LJ + Gaussian - CutoffShift;'
        energy_function += 'LJ = 0;'
        #energy_function += 'LJ = 4*eps(type1,type2)*(rinv12 - rinv6);'
        energy_function += 'Gaussian = B(type1,type2)*exp(-kappa(type1,type2)*(r - dist0(type1,type2))^2);'
        #energy_function += 'rinv12 = rinv6^2;'
        #energy_function += 'rinv6 = (sig(type1,type2)/r)^6;'
        energy_function += 'CutoffShift = 4*eps(type1,type2)*( (sig(type1,type2)/{0})^12 - (sig(type1,type2)/{0})^6 ) + B(type1,type2)*exp(-kappa(type1,type2)*({0}-dist0(type1,type2))^2);'.format(GlobalCut)

        fcnb = openmm.CustomNonbondedForce(energy_function)
        fcnb.addPerParticleParameter('type')
        fcnb.setCutoffDistance( GlobalCut )
        fcnb.setNonbondedMethod( nonbondedMethod ) #2 is cutoff periodic

        fcnb.addTabulatedFunction('eps', openmm.Discrete2DFunction(nspec,nspec,epsmatrix.ravel(order='F')) )
        fcnb.addTabulatedFunction('sig', openmm.Discrete2DFunction(nspec,nspec,sigmatrix.ravel(order='F')) )
        fcnb.addTabulatedFunction('B', openmm.Discrete2DFunction(nspec,nspec,Bmatrix.ravel(order='F')) )
        fcnb.addTabulatedFunction('kappa', openmm.Discrete2DFunction(nspec,nspec,kappamatrix.ravel(order='F')) )
        fcnb.addTabulatedFunction('dist0', openmm.Discrete2DFunction(nspec,nspec,dist0matrix.ravel(order='F')) )
        for atom in top.atoms():
            fcnb.addParticle( [atom_type_index[atom.name]] )
        system.addForce(fcnb)


    #===================================== 
    #--- create the external potential ---
    #=====================================
    uext_ffs = parsevalidate.parseUExt( my_top.system_specs )
    if uext_ffs:
        print("---> Found External Force")
    #TODO: possibly allow the period length to be changed
    direction=['x','y','z']

    f_exts=[]

    for potential in uext_ffs:
        print(potential)
        type_names_in_potential = potential[1]
        Uext = potential[2]
        n_period = potential[3]
        axis = potential[4]
        offset = potential[5]

        external={"planeLoc":offset, "ax":axis, "U":Uext*epsilon.value_in_unit(unit.kilojoule/unit.mole), "NPeriod":n_period}

        print("...Adding external potential: Uext={}, NPeriod={}, axis={}".format(external["U"],external["NPeriod"],external["ax"]))
        print("......with atom types: {}".format(type_names_in_potential))
        energy_function = 'U*sin(2*{pi}*NPeriod*(r-{r0})/{L}); r={axis};'.format(pi=np.pi, L=box_edge[external["ax"]], r0=external["planeLoc"], axis=direction[external["ax"]])
        print('...{}'.format(energy_function))
        f_exts.append( openmm.CustomExternalForce(energy_function) )
        f_exts[-1].addGlobalParameter("U", external["U"])
        f_exts[-1].addGlobalParameter("NPeriod", external["NPeriod"])

        for ia,atom in enumerate(top.atoms()):
            if atom.name in type_names_in_potential:
                print('adding atom {} {} to external force'.format(ia,atom.name))
                f_exts[-1].addParticle( ia,[] )
                
        system.addForce(f_exts[-1])
             

    for f in f_exts:
        print("External potential with amplitude U={}, NPeriod={}".format(f.getGlobalParameterDefaultValue(0), f.getGlobalParameterDefaultValue(1)))


    #================================ 
    #--- setup the electrostatics ---
    #================================
    lb, ewald_cut, ewald_tolerance, a_born = parsevalidate.parseElec( my_top.system_specs )

    if lb is not None:
        has_electrostatics = True
    if has_electrostatics and lb > 0.:
        nbfmethod = openmm.NonbondedForce.PME 
        print("To implement in OMM, unit charge is now {}".format(q_factor))
        charge_scale = q_factor * lb**0.5

        nbf = openmm.NonbondedForce()
        nbf.setCutoffDistance( ewald_cut )
        nbf.setEwaldErrorTolerance( ewald_tolerance )
        nbf.setNonbondedMethod( nbfmethod )
        nbf.setUseDispersionCorrection(False)
        nbf.setUseSwitchingFunction(False)

        for i, atom in enumerate(my_top.atoms):
            charge = atom.charge * charge_scale #In dimensionless, EwaldPotential.Coef is typically 1, and usually change relative strength via temperature. But can also scale the coef, which then acts as lB in the unit length
            LJsigma = 1.0
            LJepsilon = 0.0
            nbf.addParticle(charge, LJsigma, LJepsilon)
        
        system.addForce(nbf)
    
    if has_electrostatics and lb > 0. and a_born is not None: #Need explicit analytical portion to cancel out 1/r; tabulation can't capture such a steep potential
        print( "Ewald coefficient is: {}".format(lb))
        #print( "Matrix of smearing correction coefficients is:" )
        #print(smPrefactorMatrix)

        nspec = my_top.num_atom_types
        if nspec != len(a_born):
            raise ValueError("a_born has a different length than num_species!")
        a_born_matrix = np.zeros( (nspec, nspec) )
        for ii in range(nspec):
            for jj in range(nspec):
                a_born_matrix[ii,jj] = np.sqrt( 0.5*a_born[ii]**2. + 0.5*a_born[ii]**2. )

        energy_function = 'coef*q1*q2 * ( (erf(factor*r) - 1)/r - shift );'
        energy_function += 'shift = (erf(factor*rcut) -1)/rcut;'
        energy_function += 'factor = sqrt({:f})/2/aborn(type1,type2);'.format(np.pi)
        energy_function += 'coef = {:f};'.format(lb)
        energy_function += 'rcut = {:f};'.format(ewald_cut)
        fcnb = openmm.CustomNonbondedForce(energy_function)

        fcnb.addPerParticleParameter('type')
        fcnb.addPerParticleParameter('q')
        fcnb.setCutoffDistance( ewald_cut )
        fcnb.setNonbondedMethod( openmm.NonbondedForce.CutoffPeriodic )
        fcnb.addTabulatedFunction('aborn',openmm.Discrete2DFunction(nspec, nspec, a_born_matrix.ravel(order='F')))
        for ia,atom in enumerate(top.atoms()):
            q = my_top.atoms[ia].charge
            print("atom {} has charge {}".format(ia,q))
            fcnb.addParticle( [atom_type_index[atom.name], q] )
        system.addForce(fcnb)


    # === Finished! ===
    return  system,top,mdtrajtop


# ============================== #


def createOMMSimulation( system_specs, system, top, Prefix="", chkfile='chkpnt.chk', verbose=False):
    # --- integration options ---
    # TODO: NPT
    sim_options,run_options = parsevalidate.parseSimulation( system_specs )
    print(sim_options)
    reduced_timestep = sim_options['dt']
    dt = reduced_timestep * tau
   
    if sim_options['T'] is not None:
        reduced_temp = sim_options['T']#1
        temperature = reduced_temp * epsilon/kB/N_av
    
        reduced_Tdamp = sim_options['t_damp']
        friction = 1/(reduced_Tdamp) /tau
    
    pressure = sim_options['P']
    if pressure is not None:
        useNPT = True
        useNVT = False
    else:
        useNPT = False
        useNVT = True
    """
    reduced_pressure = 1
    reduced_Pdamp = 0.1 #time units
    pressure = reduced_pressure * epsilon/(sigma**3) / N_av
    barostatInterval = int(reduced_Pdamp/reduced_timestep)
    """

    #===========================
    ## Prepare the Simulation ##
    #===========================
    print("=== Preparing Simulation ===")
    if useNPT:
        pressure = pressure * epsilon/N_av/sigma/sigma/sigma #convert from unitless to OMM units
        barostatInterval = sim_options['barostat_freq'] #in units of time steps. 25 is OpenMM default
        if sim_options['barostat_axis'] in ['isotropic','iso','all','xyz']:
            my_barostat = openmm.MonteCarloBarostat(pressure, temperature, barostatInterval)
        elif sim_options['barostat_axis'] in [0,'x','X']:
            my_barostat = mm.MonteCarloAnisotropicBarostat(mm.vec3.Vec3(pressure*unit.bar, pressure*unit.bar, pressure*unit.bar), temperature, True, False, False, barostatInterval)
        elif sim_options['barostat_axis'] in [1,'y','Y']:
            my_barostat = mm.MonteCarloAnisotropicBarostat(mm.vec3.Vec3(pressure*unit.bar, pressure*unit.bar, pressure*unit.bar), temperature, False, True, False, barostatInterval)
        elif sim_options['barostat_axis'] in [2,'z','Z']:
            my_barostat = mm.MonteCarloAnisotropicBarostat(mm.vec3.Vec3(pressure*unit.bar, pressure*unit.bar, pressure*unit.bar), temperature, False, False, True, barostatInterval)
        
        system.addForce(my_barostat)
        
        print("Added MC Barostat with P {} (eps/sig^3), T {}, freq {}".format(
            6.02214179e23 * pressure.value_in_unit(unit.kilojoules/unit.nanometer**3),temperature,barostatInterval))
        print("In OMM units, is P={}'bar'".format(pressure.value_in_unit(unit.bar)))

    if sim_options['T'] is None:
        print("This is a constant energy, constant volume (NVE) run.")
        integrator = mm.VerletIntegrator(dt)
    else:
        if sim_options['thermostat'] == 'langevin':
            print("This is a NVT run with langevin thermostat")
            integrator = openmm.LangevinIntegrator(temperature, friction, dt)
        else:
            raise OMMError("Specified temperature but unsupported thermostat {}".format(sim_options['thermostat']))

    if system.getNumConstraints() > 0:
        print("Applying bond constraints before starting")
        integrator.setConstraintTolerance(CONSTRAINT_TOLERANCE)

    # TODO: simulation platform
    #platform = openmm.Platform.getPlatformByName('CUDA')
    #platformProperties = {'Precision': 'mixed'}
    #platform = openmm.Platform.getPlatformByName('CPU')
    #platform = openmm.Platform.getPlatformByName('Reference')
    #simulation = app.Simulation(top,system, integrator, platform)
    #simulation = app.Simulation(top,system, integrator, platform,platformProperties)
    if sim_options['platform'] is None:
        print("Automatically choosing platform")
        simulation = app.Simulation(top,system,integrator)
    else:
        platformName = sim_options['platform']
        device = sim_options['device']
        print("Manually setting platform: {} and device: {}".format(platformName,device))
        platform = openmm.Platform.getPlatformByName(platformName)
        if platformName.lower() in ['CUDA','cuda']:
            platform.setPropertyDefaultValue("CudaDeviceIndex", str(device))
        elif platformName.lower() in ['OpenCL','opencl']:
            platform.setPropertyDefaultValue("OpenCLDeviceIndex", str(device))
        else:
            platform = openmm.Platform.getPlatformByName('CPU')
        simulation = app.Simulation(top,system, integrator, platform)

    chkfile = Prefix + chkfile
    chkpt_freq = 10000
    simulation.reporters.append(app.checkpointreporter.CheckpointReporter(chkfile,chkpt_freq))

    # --- done ---
    #simOptions = {'dt':dt, 'temp':temperature, 'fric':friction}
    sim_options['temp_dimensionful'] = temperature
    print("Parsed simulation options: {}".format(sim_options))
    print("Parsed runtime options: {}".format(run_options))
    return simulation, sim_options, run_options 


# ============================== #


def runOpenMM(my_top, init_xyz = None, Prefix = "", verbose = False, nsteps_min = None, nsteps_equil = None, nsteps_prod = None, write_freq = None, protocol = None, protocolArgs={}, TrajFile = "trj.dcd"):
    "Run OpenMM. Returns OMMFiles (initial pdb, equilibration pdb, final pdb, dcd, thermo log, output log, simulation preamble, return_code)"
    """
    protocol : None, function
        if a function, should have function signature protocol(simulation, sim_options, run_options, **protocolArgs)

    """
    openMM_files = []
    # --- Get OMM System and Simulation ---
    system,top,mdtrajtop = createOMMSys(my_top,verbose)
    simulation,sim_options,run_options = createOMMSimulation(my_top.system_specs, system, top, Prefix,verbose=verbose)

    # --- simulation options ---
    TrajFile = Prefix + TrajFile
    EqLogFile = Prefix + "eqlog.txt"
    LogFile = Prefix + "prodlog.txt"
    openMM_files.append(LogFile)

    if nsteps_min is not None:
        print("Overriding nsteps_min")
        run_options['nsteps_min'] = nsteps_min
    else:
        nsteps_min = run_options['nsteps_min']

    if nsteps_equil is not None: 
        print("Overriding nsteps_equil")
        run_options['nsteps_equil'] = nsteps_equil
    else:
        nsteps_equil = run_options['nsteps_equil']

    if nsteps_prod is not None: 
        print("Overriding nsteps_prod")
        run_options['nsteps_prod'] = nsteps_prod
    else:
        nsteps_prod = run_options['nsteps_prod']

    if write_freq is not None:
        print("Overriding write_freq")
        run_options['write_freq'] = write_freq
    else:
        write_freq = run_options['write_freq']

    # --- Init ---
    print("=== Initializing Simulation ===")
   
    #... need to make molecules whole
    Lx, Ly, Lz = parsevalidate.parseBox( my_top.system_specs['SystemOptions'] )
    unitcell_lengths = np.array([Lx,Ly,Lz])
    unitcell_angles = np.array([90., 90., 90.])
 
    if init_xyz is None:
        if run_options['initial'].lower() in ['rand','random']:
            print("Chose random initial coordinates")
            pos = np.random.random( (my_top.num_atoms, 3) ) * unitcell_lengths
        else:
            pos = mdtraj.load( run_options['initial'] )
    elif type(init_xyz) == np.ndarray:
        print("Overriding run_options with init_xyz array")
        pos = init_xyz.copy()
    elif type(init_xyz) == str:
        print("Overriding run_options with init_xyz file {}".format(init_xyz))
        run_options['initial'] = init_xyz
        pos = mdtraj.load(init_xyz).xyz[0]
   

    mdt_traj = mdtraj.Trajectory(np.array(pos), topology = mdtrajtop, unitcell_lengths = unitcell_lengths, unitcell_angles = unitcell_angles)
    bond_list = np.array([ [b[0].index,b[1].index] for b in top.bonds() ], dtype=np.int32) #should be sorted, as long as the bonds come out sorted from the sim Sys object

    if len(bond_list) > 0:
        print(">0 bonds, making trajectory whole to be safe")
        whole_traj = mdt_traj.make_molecules_whole(inplace=False,sorted_bonds=bond_list)
        simulation.context.setPositions(whole_traj.xyz[0])
    else:
        simulation.context.setPositions(pos)

    #...files
    init_state_file = Prefix + 'output.xml'
    simulation.saveState(init_state_file)
    openMM_files.append(init_state_file)
    initialpdb = Prefix + "initial.pdb"
    initial_positions = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()
    app.PDBFile.writeModel(simulation.topology, initial_positions, open(initialpdb,'w'))
    openMM_files.append(initialpdb)

    #TODO: apply constraint tolerance
    #if system.getNumConstraints() > 0:
    if True:
        simulation.context.applyConstraints(CONSTRAINT_TOLERANCE)
        constrained_pdb = Prefix + "constrained.pdb"
        constrained_positions = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()
        app.PDBFile.writeModel(simulation.topology, constrained_positions, open(constrained_pdb,'w'))
        openMM_files.append(constrained_pdb)


    if nsteps_min > 0:
        print("=== Running energy minimization ===")
        minimizefile = Prefix + "minimized.pdb"
        simulation.minimizeEnergy(maxIterations=nsteps_min)
        if sim_options['T'] is not None:
            simulation.context.setVelocitiesToTemperature(sim_options["temp_dimensionful"]*3)
        minimized_positions = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()
        app.PDBFile.writeModel(simulation.topology, minimized_positions, open(minimizefile,'w'))
        openMM_files.append(minimizefile)

    # --- Equilibrate ---
    print('=== Equilibrating ===')
    simulation.reporters.append(app.StateDataReporter(sys.stdout, write_freq*100, step=True, potentialEnergy=True, kineticEnergy=True, temperature=True, volume=True, density=True, speed=True, separator='\t'))
    simulation.reporters.append(app.StateDataReporter(EqLogFile, write_freq, step=True, potentialEnergy=True, kineticEnergy=True, temperature=True, volume=True, density=True, speed=True, separator='\t'))
    print("Progress will be reported every {} steps".format(write_freq))
    simulation.step(nsteps_equil)
    equilibrate_file = Prefix + "equilibrated.pdb"
    equilibrated_positions = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()
    app.PDBFile.writeModel(simulation.topology, equilibrated_positions, open(equilibrate_file,'w'))
    openMM_files.append(equilibrate_file)
    

    # --- Production ---
    print('=== Production {} steps total, {} tau total ({}) ==='.format(nsteps_prod, nsteps_prod*sim_options['dt'], nsteps_prod*sim_options['dt']*tau ))
    simulation.reporters.pop()
    simulation.reporters.append(app.StateDataReporter(LogFile, write_freq, step=True, potentialEnergy=True, kineticEnergy=True, temperature=True, volume=True, density=True, speed=True, separator='\t'))
    simulation.reporters.append(mdtraj.reporters.DCDReporter(TrajFile,write_freq))
    #simulation.reporters.append(app.checkpointreporter.CheckpointReporter('checkpnt.chk', 5000))
    start = time.time()
    if protocol is not None:
        print("---> Gave a protocol function {}, overriding".format(protocol.__name__))
        run_options['protocol'] = protocol.__name__
        protocol(simulation, sim_options, run_options)
    elif run_options['protocol'] in ['tension']:
        print("---> Running tension protcol")

        tension_options = {}
        tension_options['tension_dimless'] = sim_options['tension'] # kT/sig^2
        tension_options['axis'] = sim_options['tension_axis'] # kT/sig^2
        ax = sim_options['tension_axis']

        #tension_options['temp_dimless'] = sim_options['temp_dimensionful']/unit.kelvin # dimless
        tension_options['temp_dimless'] = sim_options['T']
        tension_options['tension_freq'] = sim_options['tension_freq'] # int
        tension_options['nblocks'] = int( np.round( run_options['nsteps_prod']/tension_options['tension_freq']) )

        tension_options['alpha_scale'] = sim_options['tension_alphascale']
        tension_options['Amax'] = sim_options['tension_Amax']
        

        runTension(simulation, **tension_options)
    elif run_options['protocol'].lower() in ['simple','null']:
        print("---> selected simple protocol")
        simulation.step(nsteps_prod)
    else:
        print("---> Defaulting to simple protocol")
        simulation.step(nsteps_prod)


    end = time.time()
    print("=== Finished production in {} seconds".format(end-start))
    print("run_options script, including overrides: {}".format(run_options))

    # --- Returns ---
    #TODO: validate output for success
    return_code = 0
    return openMM_files, TrajFile, return_code

# ======================
# Custom protocols
"""
dAfrac = 0.001
scaling = 1+dAfrac
scaleNormal = 1/scaling
scaleTangent = scaling**0.5
"""

def runTension( simulation, tension_dimless, axis, temp_dimless, tension_freq, nblocks, dAfrac = 0.001, alpha_scale=0., Amax=3., checkpointpdb = 'tension_checkpoint.pdb'):
    """
    Parameters
    ----------
    Amax : float
        default = 3, s.t. A = A0/2 yields a tension that roughly cancels the applied tension

    Notes
    restoring force is given by alpha * (L-L0)^2
        I set alpha = alpha_scale / (1 - sqrt(1/Amax))
        alpa_scale is magnitude of the tension
        Amax is when the restoring force will cancel out an applied tension = alpha_scale
            similarly, will double applied tension when Amin~ A/A0 ~ Amax/(1-2sqrt(Amax))^2
            i.e. setting Amax ~3. (default value) will yield an 'Amin'~0.494088
    """
    # TODO
    #   1) incoporate limits
    #   2) incorporate axis-selection
    #   3) convert units of tension to CG units
    #   4) L0...
    #parse parameters
    top = parmed.openmm.load_topology(simulation.topology)
    print('...Tension parameters: tension = {}kT/sig^3, axis = {}, tension_freq = {}, dAfrac = {}, restoring alpha_scale = {}kT/sig^3, Aupper/A0 = {}'.format(tension_dimless, axis, tension_freq, dAfrac, alpha_scale, Amax))


    block_steps = tension_freq
    #tension = tension_dimless*unit.bar*unit.nanometer*unit.AVOGADRO_CONSTANT_NA #Need to do CG unit conversion...
    tension = tension_dimless * epsilon/sigma/sigma
    #temp_dimensionful = temp_dimless * unit.kelvin
    
    scaling = 1+dAfrac
    scalieNormal = 1/scaling
    scaleTangent = scaling**0.5

    tangents = np.mod([axis-1, axis+1],2)
    this_box = simulation.context.getState().getPeriodicBoxVectors()
    print(this_box)
    L0 = this_box[tangents[0]][tangents[0]]#assuming z-axis, convert Angstrom to nm
    A0 = this_box[tangents[0]][tangents[0]] * this_box[tangents[1]][tangents[1]]#assuming z-axis
    print('L0: {}, A0: {}'.format(L0,A0)) 
    #Amax = 3.0 #relative to A0
    #alpha_scale = 10.


    #--- Assumes tension_dimless in units of eps/sig^2 ---
    #kBT = unit.AVOGADRO_CONSTANT_NA * unit.BOLTZMANN_CONSTANT_kB * temp_dimless * unit.kelvin
    kBT = epsilon * temp_dimless
    #alpha = alpha_scale/(1-np.sqrt(1/Amax))*unit.bar*unit.nanometer*unit.AVOGADRO_CONSTANT_NA
    alpha = alpha_scale/(1-np.sqrt(1/Amax))*kBT/sigma/sigma

    #if nsteps_prod == 0:
    #    nblocks = int( np.round( nsteps_prod/block_steps ) )
    
    #tension_dimless = sim_options['tension']

    #initialize
    box_sizes = np.zeros([nblocks,3])
    TotalAcceptances = 0
    TotalMCMoves = 0
    
    #run
    for iblock in range(nblocks):
        print("Starting block {}".format(iblock))
        start = time.time()
        simulation.step(block_steps)
        end = time.time()
        print('Took {} seconds for block {}'.format(end-start,iblock))
        this_box = simulation.context.getState().getPeriodicBoxVectors()
        print('Box size: {}'.format(this_box)) 
        box_sizes[iblock,:] = [this_box[0][0].value_in_unit(unit.nanometer), this_box[1][1].value_in_unit(unit.nanometer), this_box[2][2].value_in_unit(unit.nanometer)]

        if tension_dimless > 0:
            print('=== Attempting area change ===')
            #--- Set up ---
            Eold = simulation.context.getState(getEnergy=True).getPotentialEnergy()
            print('... Current energy: {}'.format( Eold.value_in_unit(unit.kilojoule_per_mole) ))
            if np.random.random_sample() < 0.5:
                print('... proposing to shrink area by {}...'.format(scaling))
                tmpf = 1/scaling
            else:
                print('... proposing to expand area by {}...'.format(scaling))
                tmpf = scaling

            scaleNormal = 1.0/tmpf
            scaleTangent = tmpf**0.5

            if axis == 0:
                scalings = [scaleNormal, scaleTangent, scaleTangent]
                tangents = [1,2]
            elif axis == 1:
                scalings = [scaleTangent, scaleNormal, scaleTangent]
                tangents = [0,2]
            elif axis == 2:
                scalings = [scaleTangent, scaleTangent, scaleNormal]
                tangents = [0,1]
            else:
                raise ValueError('Unrecognized tension axis {}'.format(axis))
            scalings = np.array(scalings)

            
            #--- actually scale box ---
            oldbox = [this_box[0], 
                        this_box[1],
                        this_box[2]]
            newbox = [this_box[0]*scalings[0],
                        this_box[1]*scalings[1],
                        this_box[2]*scalings[2]]
            Aold = oldbox[tangents[0]][tangents[0]]*oldbox[tangents[1]][tangents[1]]
            Anew = newbox[tangents[0]][tangents[0]]*newbox[tangents[1]][tangents[1]]
            deltaArea = Aold*(tmpf-1.0)
            Lold = oldbox[tangents[0]][tangents[0]]
            Lnew = newbox[tangents[0]][tangents[0]]
            deltaL = Lold*(scaleTangent - 1)
            simulation.context.setPeriodicBoxVectors( newbox[0], newbox[1], newbox[2] )
            
            natoms = len(top.atoms) #using parmed gromacs topology object
            pos = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
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



