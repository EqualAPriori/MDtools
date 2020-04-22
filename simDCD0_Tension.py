################################################################
# Kevin Shen, 2019                                             #
# kevinshen@ucsb.edu                                           #
#                                                              #
# General purpose openMM simulation script.                    #
# Allows for (verlet, langevin); barostats; LJPME              #
# simulation protocol:                                         #
#   1) equilibrate                                             #
#   2) production run                                          #
#Doesn't write no-water config, unlike simDCD.py               #
#                                                              # 
#This version simulates a variable tension ensemble            #
#  Apply energy penaltly: -gamma(A-A0) + 0.5*alpha(A-A0)^2     #
#Alternative form: (symmetric in L, asymmetric in A)           #
#  -gamma (A-A0) + alpha(L-L0)^2                               #
#Should be used with a barostat, i.e. NPAT, assumes z-axis     #
#Uses Built-in Barostat to handle z-changes                    #
################################################################


# System stuff
from sys import stdout
import time
import os,sys,logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel('INFO')
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(fmt='%(asctime)s - %(message)s', datefmt="%H:%M:%S")
sh.setFormatter(formatter)
import argparse
from collections import namedtuple, defaultdict, OrderedDict
import numpy as np

# OpenMM Imports
import simtk.openmm as mm
import simtk.openmm.app as app

# ParmEd & MDTraj Imports
from parmed import gromacs
#gromacs.GROMACS_TOPDIR = "/home/kshen/SDS"
from parmed.openmm.reporters import NetCDFReporter
from mdtraj.reporters import DCDReporter
from parmed import unit as u
import parmed as pmd
import mdtraj

# Custom Tools
import mdparse



dAfrac = 0.001
scaling = 1+dAfrac
scaleNormal = 1/scaling
scaleTangent = scaling**0.5

def add_barostat(system,args):
    if args.pressure <= 0.0:
        logger.info("This is a constant volume (NVT) run")
    else:
        logger.info("This is a constant pressure (NPT) run at %.2f bar pressure" % args.pressure)
        logger.info("Adding Monte Carlo barostat with volume adjustment interval %i" % args.nbarostat)
        logger.info("Anisotropic box scaling is %s" % ("ON" if args.anisotropic else "OFF"))
        #if args.tension > 0:
        if (args.tension is not None) and args.restoring_scale == 0.:
            logger.info('...detected tension, but no extra nonlinear restoring force, using openMM Membrane Barostat...')
            logger.info('...openMM MembraneBarostat only accepts z-axis, be warned!')
            logger.info('...setting tension {} bar*nm'.format(args.tension))
            #print('...assume tension is given in bar*nm'
            #tension = args.tension * u.bar*u.nanometer
            #tension = tension.value_in_unit(u.bar*u.nanometer)
            XYmode = mm.MonteCarloMembraneBarostat.XYIsotropic
            Zmode = mm.MonteCarloMembraneBarostat.ZFree
            barostat = mm.MonteCarloMembraneBarostat(args.pressure*u.bar, args.tension, args.temperature * u.kelvin, XYmode, Zmode, args.nbarostat)
        elif args.tension is not None and args.restoring_scale != 0.:
            logger.info("Only the Z-axis will be adjusted for NPT moves, keep Volume constant when perturbing area")
            barostat = mm.MonteCarloAnisotropicBarostat(mm.vec3.Vec3(args.pressure*u.bar, args.pressure*u.bar, args.pressure*u.bar), args.temperature*u.kelvin, False, False, True, args.nbarostat)

            '''
            logger.info("Tension={} > 0, using Membrane Barostat with z-mode {}".format(args.tension, args.zmode))
            if args.anisotropic:
                logger.info("XY-axes will change length independently")
                XYmode = mm.MonteCarloMembraneBarostat.XYAnisotropic
            else:
                XYmode = mm.MonteCarloMembraneBarostat.XYIsotropic
            #barostat = mm.MonteCarloMembraneBarostat(args.pressure*u.bar,args.tension*u.bar*u.nanometer,args.temperature*u.kelvin,XYmode,args.zmode,args.nbarostat) 
            barostat = mm.MonteCarloMembraneBarostat(args.pressure*u.bar,args.tension,args.temperature*u.kelvin,XYmode,args.zmode,args.nbarostat) 
            '''
        elif args.anisotropic:
            logger.info("Just a barostat, only the Z-axis will be adjusted")
            barostat = mm.MonteCarloAnisotropicBarostat(mm.vec3.Vec3(args.pressure*u.bar, args.pressure*u.bar, args.pressure*u.bar), args.temperature*u.kelvin, False, False, True, args.nbarostat)
        else:
            barostat = mm.MonteCarloBarostat(args.pressure * u.bar, args.temperature * u.kelvin, args.nbarostat)
        system.addForce(barostat)
    '''
    else:
        args.deactivate("pressure", msg="System is nonperiodic")
        #raise Exception('Pressure was specified but the topology contains no periodic box! Exiting...')
    '''


def set_thermo(system,args):
    '''
    Takes care of thermostat if needed
    '''
    if args.temperature <= 0.0:
        logger.info("This is a constant energy, constant volume (NVE) run.")
        integrator = mm.VerletIntegrator(2.0*u.femtoseconds)
    else:
        logger.info("This is a constant temperature run at %.2f K" % args.temperature)
        logger.info("The stochastic thermostat collision frequency is %.2f ps^-1" % args.collision_rate)
        if args.integrator == "langevin":
            logger.info("Creating a Langevin integrator with %.2f fs timestep." % args.timestep)
            integrator = mm.LangevinIntegrator(args.temperature * u.kelvin, 
                                                args.collision_rate / u.picoseconds, 
                                                args.timestep * u.femtosecond)
        elif args.integrator == "verlet":
            integrator = mm.VerletIntegrator(2.0*u.femtoseconds)
            thermostat = mm.AndersenThermostat(args.temperature * u.kelvin, args.collision_rate / u.picosecond)
            system.addForce(thermostat)
        else:
            logger.warning("Unknown integrator, will crash now")
        add_barostat(system,args)
    return integrator


def main(paramfile='params.in', overrides={}, quiktest=False, deviceid=None, progressreport=True): #simtime=2.0, T=298.0, NPT=True, LJcut=10.0, tail=True, useLJPME=False, rigidH2O=True, device=0, quiktest=False):
    # === PARSE === #
    args = mdparse.SimulationOptions(paramfile, overrides)
    
    
    # Files
    gromacs.GROMACS_TOPDIR = args.topdir
    top_file        = args.topfile
    box_file        = args.grofile
    defines         = {}
    cont            = args.cont


    args.force_active('chkxml',val='chk_{:02n}.xml'.format(cont),msg='first one')
    args.force_active('chkpdb',val='chk_{:02n}.pdb'.format(cont),msg='first one')
    if cont > 0:
        args.force_active('incoord',val='chk_{:02n}.xml'.format(cont-1),msg='continuing')
        args.force_active('outpdb',val='output_{:02n}.pdb'.format(cont),msg='continuing')
        args.force_active('outnetcdf',val='output_{:02n}.nc'.format(cont),msg='continuing')
        args.force_active('logfile',val='thermo.log_{:02n}'.format(cont),msg='continuing')
        args.force_active('outdcd',val='output_{:02n}.dcd'.format(cont),msg='continuing')

    incoord         = args.incoord
    out_pdb         = args.outpdb
    out_netcdf      = args.outnetcdf
    out_dcd         = args.outdcd
    molecTopology   = 'topology.pdb'
    out_nowater     = 'output_nowater.nc'
    out_nowater_dcd = 'output_nowater.dcd'
    logfile         = args.logfile
    checkpointxml   = args.chkxml
    checkpointpdb   = args.chkpdb
    checkpointchk   = 'chk_{:02n}.chk'.format(cont)

    # Parameters
    #Temp            = args.temperature        #K
    #Pressure = 1      #bar
    #barostatfreq    = 25 #time steps
    #fric            = args.collision_rate     #1/ps

    dt              = args.timestep 	      #fs
    if args.use_fs_interval:
        reportfreq = int(args.report_interval/dt)
        netcdffreq = int(args.netcdf_report_interval/dt) #5e4
        dcdfreq    = int(args.dcd_report_interval/dt)
        pdbfreq    = int(args.pdb_report_interval/dt)
        checkfreq  = int(args.checkpoint_interval/dt)
        #simtime    = int( simtime ) #nanoseconds; make sure division is whole... no remainders...
        blocksteps = int(args.block_interval/dt)   #1e6, steps per block of simulation 
        nblocks    = args.nblocks #aiming for 1 block is 1ns
    else:
        reportfreq = args.report_interval
        netcdffreq = args.netcdf_report_interval
        dcdfreq    = args.dcd_report_interval
        pdbfreq    = args.pdb_report_interval
        checkfreq  = args.checkpoint_interval
        blocksteps = args.block_interval
        nblocks    = args.nblocks 

    if quiktest==True:
        reportfreq = 1
        blocksteps = 10
        nblocks = 2

    # === Start Making System === # 
    start = time.time()
    top = gromacs.GromacsTopologyFile(top_file, defines=defines)
    gro = gromacs.GromacsGroFile.parse(box_file)
    top.box = gro.box
    logger.info("Initial Box: {}".format(gro.box))
    L0 = gro.box[0]/10.*u.nanometer #assuming z-axis, convert Angstrom to nm
    A0 = gro.box[0]*gro.box[1]/100.*u.nanometer**2 #assuming z-axis
    print("For restoring tension calculations, A0={}nm^2, L0={}nm".format(A0, L0))
    logger.info("Took {}s to create topology".format(time.time()-start))
    print(top)

    constr = {None: None, "None":None,"HBonds":app.HBonds,"HAngles":app.HAngles,"AllBonds":app.AllBonds}[args.constraints]   
    start = time.time()
    system = top.createSystem(nonbondedMethod=app.NoCutoff, ewaldErrorTolerance = args.ewald_error_tolerance,
                        nonbondedCutoff=args.nonbonded_cutoff*u.nanometers,
                        rigidWater = args.rigid_water, constraints = constr)
    logger.info("Took {}s to create system".format(time.time()-start))
                          
 
    nbm = {"NoCutoff":mm.NonbondedForce.NoCutoff, "CutoffNonPeriodic":mm.NonbondedForce.CutoffNonPeriodic,
                "Ewald":mm.NonbondedForce.Ewald, "PME":mm.NonbondedForce.PME, "LJPME":mm.NonbondedForce.LJPME}[args.nonbonded_method]

    ftmp = [f for ii, f in enumerate(system.getForces()) if isinstance(f,mm.NonbondedForce)]
    fnb = ftmp[0]
    fnb.setNonbondedMethod(nbm)
    logger.info("Nonbonded method ({},{})".format(args.nonbonded_method, fnb.getNonbondedMethod()) )
    if (not args.dispersion_correction) or (args.nonbonded_method=="LJPME"):
        logger.info("Turning off tail correction...")
        fnb.setUseDispersionCorrection(False)

    logger.info("Check dispersion correction flag: {}".format(fnb.getUseDispersionCorrection()) )

    #print("Examining exceptions: ")
    #for ii in range(0,70000,100):
    #    print("{}".format(fnb.getExceptionParameters(ii)))

    # === Integrator, Barostat, Additional Constraints === #
    integrator = set_thermo(system,args)

    if not hasattr(args,'constraints') or (str(args.constraints) == "None" and args.rigid_water == False):
        args.deactivate('constraint_tolerance',"There are no constraints in this system")
    else:
        logger.info("Setting constraint tolerance to %.3e" % args.constraint_tolerance)
        integrator.setConstraintTolerance(args.constraint_tolerance)


    # === Make Platform === #
    logger.info("Setting Platform to %s" % str(args.platform))
    try:
        platform = mm.Platform.getPlatformByName(args.platform)
    except:
        logger.info("Warning: %s platform not found, going to Reference platform \x1b[91m(slow)\x1b[0m" % args.platform)
        args.force_active('platform',"Reference","The %s platform was not found." % args.platform)
        platform = mm.Platform.getPlatformByName("Reference")

    if deviceid is not None or deviceid>=0:
        args.force_active('device',deviceid,msg="Using cmdline-input deviceid")
    if 'device' in args.ActiveOptions and (platform.getName()=="OpenCL" or platform.getName()=="CUDA"):
        device = str(args.device)
        # The device may be set using an environment variable or the input file.
        #if 'CUDA_DEVICE' in os.environ.keys(): #os.environ.has_key('CUDA_DEVICE'):
        #    device = os.environ.get('CUDA_DEVICE',str(args.device))
        #elif 'CUDA_DEVICE_INDEX' in os.environ.keys(): #os.environ.has_key('CUDA_DEVICE_INDEX'):
        #    device = os.environ.get('CUDA_DEVICE_INDEX',str(args.device))
        #else:
        #    device = str(args.device)
        if device != None:
            logger.info("Setting Device to %s" % str(device))
            #platform.setPropertyDefaultValue("CudaDevice", device)
            if platform.getName()=="CUDA":
                platform.setPropertyDefaultValue("CudaDeviceIndex", device)
            elif platform.getName()=="OpenCL":
                print("set OpenCL device to {}".format(device))
                platform.setPropertyDefaultValue("OpenCLDeviceIndex", device)
        else:
            logger.info("Using the default (fastest) device")
    else:
        logger.info("Using the default (fastest) device, or not using CUDA nor OpenCL")

    if "Precision" in platform.getPropertyNames() and (platform.getName()=="OpenCL" or platform.getName()=="CUDA"):
        platform.setPropertyDefaultValue("Precision", args.cuda_precision)
    else:
        logger.info("Not setting precision")
        args.deactivate("cuda_precision",msg="Platform does not support setting cuda_precision.")

    # === Create Simulation === #
    logger.info("Creating the Simulation object")
    start = time.time()
    # Get the number of forces and set each force to a different force group number.
    nfrc = system.getNumForces()
    if args.integrator != 'mtsvvvr':
        for i in range(nfrc):
            system.getForce(i).setForceGroup(i)
    '''
    for i in range(nfrc):
        # Set vdW switching function manually.
        f = system.getForce(i)
        if f.__class__.__name__ == 'NonbondedForce':
            if 'vdw_switch' in args.ActiveOptions and args.vdw_switch:
                f.setUseSwitchingFunction(True)
                f.setSwitchingDistance(args.switch_distance)
    '''

    #create simulation object
    if args.platform != None:
        simulation = app.Simulation(top.topology, system, integrator, platform)
    else:
        simulation = app.Simulation(top.topology, system, integrator)
    topomm = mdtraj.Topology.from_openmm(simulation.topology)
    logger.info("System topology: {}".format(topomm))


    #print platform we're using
    mdparse.printcool_dictionary({i:simulation.context.getPlatform().getPropertyValue(simulation.context,i) for i in simulation.context.getPlatform().getPropertyNames()},title="Platform %s has properties:" % simulation.context.getPlatform().getName())


    logger.info("--== PME parameters ==--")
    ftmp = [f for ii, f in enumerate(simulation.system.getForces()) if isinstance(f,mm.NonbondedForce)]
    fnb = ftmp[0]   
    if fnb.getNonbondedMethod()==3:
        PMEparam = fnb.getPMEParametersInContext(simulation.context)
        logger.info(fnb.getPMEParametersInContext(simulation.context))
    if fnb.getNonbondedMethod() == 5: #check for LJPME
        PMEparam = fnb.getLJPMEParametersInContext(simulation.context)
        logger.info(fnb.getLJPMEParametersInContext(simulation.context))
    #nmeshx = int(PMEparam[1]*1.5)
    #nmeshy = int(PMEparam[2]*1.5)
    #nmeshz = int(PMEparam[3]*1.5)
    #fnb.setPMEParameters(PMEparam[0],nmeshx,nmeshy,nmeshz)
    #logger.info(fnb.getPMEParametersInContext(simulation.context))


    # Print out some more information about the system
    logger.info("--== System Information ==--")
    logger.info("Number of particles   : %i" % simulation.context.getSystem().getNumParticles())
    logger.info("Number of constraints : %i" % simulation.context.getSystem().getNumConstraints())
    for f in simulation.context.getSystem().getForces():
        if f.__class__.__name__ == 'NonbondedForce':
            method_names = ["NoCutoff", "CutoffNonPeriodic", "CutoffPeriodic", "Ewald", "PME", "LJPME"]
            logger.info("Nonbonded method      : %s" % method_names[f.getNonbondedMethod()])
            logger.info("Number of particles   : %i" % f.getNumParticles())
            logger.info("Number of exceptions  : %i" % f.getNumExceptions())
            if f.getNonbondedMethod() > 0:
                logger.info("Nonbonded cutoff      : %.3f nm" % (f.getCutoffDistance() / u.nanometer))
                if f.getNonbondedMethod() >= 3:
                    logger.info("Ewald error tolerance : %.3e" % (f.getEwaldErrorTolerance()))
                logger.info("LJ switching function : %i" % f.getUseSwitchingFunction())
                if f.getUseSwitchingFunction():
                    logger.info("LJ switching distance : %.3f nm" % (f.getSwitchingDistance() / u.nanometer))

    # Print the sample input file here.
    for line in args.record():
        print(line)

    print("Took {}s to make and setup simulation object".format(time.time()-start))

    #============================#
    #| Initialize & Eq/Warm-Up  |#
    #============================#

    p = simulation.context.getPlatform()
    if p.getName()=="CUDA" or p.getName()=="OpenCL":
        print("simulation platform: {}".format(p.getName()) )
        print(p.getPropertyNames())
        print(p.getPropertyValue(simulation.context,'DeviceName'))
        print("Device Index: {}".format(p.getPropertyValue(simulation.context,'DeviceIndex')))


    if os.path.exists(args.restart_filename) and args.read_restart:
        print("Restarting simulation from the restart file.")
        print("Currently is filler")
    else:
        # Set initial positions.
        if incoord.split(".")[-1]=="pdb":
            #print(incoord)
            #t = mdtraj.load(incoord)
            #print(t)
            #simulation.context.setPositions( traj.openmm_positions(0) )
            pdb = app.PDBFile(incoord) #pmd.load_file(incoord)
            simulation.context.setPositions(pdb.positions)
            print('Set positions from pdb, {}'.format(incoord))
            molecTopology = incoord
        elif incoord.split(".")[-1]=="xyz":
            traj = mdtraj.load(incoord, top = mdtraj.Topology.from_openmm(simulation.topology))
            simulation.context.setPositions( traj.openmm_positions(0) )

        elif incoord.split(".")[-1]=="xml":
            simulation.loadState(incoord)
            print('Set positions from xml, {}'.format(incoord))
        else:
            logger.info("Error, can't handle input coordinate filetype")
        
        if args.constraint_tolerance > 0.0:    
            simulation.context.applyConstraints(args.constraint_tolerance) #applies constraints in current frame.
        logger.info("Initial potential energy is: {}".format(simulation.context.getState(getEnergy=True).getPotentialEnergy()) )

        if args.integrator != 'mtsvvvr':
            eda = mdparse.EnergyDecomposition(simulation)
            eda_kcal = OrderedDict([(i, "%10.4f" % (j/4.184)) for i, j in eda.items()])
            mdparse.printcool_dictionary(eda_kcal, title="Energy Decomposition (kcal/mol)")

        # Minimize the energy.
        if args.minimize:
            logger.info("Minimization start, the energy is: {}".format(simulation.context.getState(getEnergy=True).getPotentialEnergy()))
            simulation.minimizeEnergy()
            logger.info("Minimization done, the energy is {}".format(simulation.context.getState(getEnergy=True).getPotentialEnergy()))
            positions = simulation.context.getState(getPositions=True).getPositions()
            logger.info("Minimized geometry is written to 'minimized.pdb'")
            app.PDBFile.writeModel(simulation.topology, positions, open('minimized.pdb','w'))
        # Assign velocities.
        if args.gentemp > 0.0:
            logger.info("Generating velocities corresponding to Maxwell distribution at %.2f K" % args.gentemp)
            simulation.context.setVelocitiesToTemperature(args.gentemp * u.kelvin)
        # Equilibrate.
        logger.info("--== Equilibrating (%i steps, %.2f ps) ==--" % (args.equilibrate, args.equilibrate * args.timestep * u.femtosecond / u.picosecond))
        if args.report_interval > 0:
            # Append the ProgressReport for equilibration run.
            simulation.reporters.append(mdparse.ProgressReport(args, sys.stdout, args.report_interval, simulation, args.equilibrate))
            simulation.reporters[-1].t00 = time.time()
            logger.info("Progress will be reported every %i steps" % args.report_interval)
        # This command actually does all of the computation.
        simulation.step(args.equilibrate)
        if args.report_interval > 0:
            # Get rid of the ProgressReport because we'll make a new one.
            simulation.reporters.pop()
        first = args.equilibrate
    

    #============================#
    #| Production MD simulation |#
    #============================#
    logger.info("--== Production (%i blocks, %i steps total, %.2f ps total) ==--" % (nblocks, nblocks*blocksteps, nblocks*blocksteps * args.timestep * u.femtosecond / u.picosecond))

    #===========================================#
    #| Add reporters for production simulation |#
    #===========================================#   
    print("===== registering reporters and runnning =====")

    if args.report_interval > 0:
        logger.info("Thermo and Progress will be reported every %i steps" % args.report_interval)
        #simulation.reporters.append(ProgressReport(sys.stdout, args.report_interval, simulation, args.production, first))
        mdparse.bak(logfile)
        simulation.reporters.append(app.StateDataReporter(logfile, reportfreq, step=True,
                potentialEnergy=True, kineticEnergy=True, temperature=True, volume=True, density=True, speed=True))
        #simulation.reporters.append(app.StateDataReporter(stdout, reportfreq, step=True,
        #        potentialEnergy=True, kineticEnergy=True, temperature=True, volume=True, density=True, speed=True))
        if progressreport:
            simulation.reporters.append(mdparse.ProgressReport(args, sys.stdout, reportfreq, simulation, nblocks*blocksteps, first=args.equilibrate))
            Prog = simulation.reporters[-1]
        

    if args.pdb_report_interval > 0:
        mdparse.bak(out_pdb)
        logger.info("PDB Reporter will write to %s every %i steps" % (out_pdb, pdbfreq))
        simulation.reporters.append(app.PDBReporter(out_pdb, pdbfreq))

    if args.netcdf_report_interval > 0:
        mdparse.bak(out_netcdf)
        logger.info("netcdf Reporter will write to %s every %i steps" %(out_netcdf, netcdffreq))
        simulation.reporters.append(NetCDFReporter(out_netcdf, netcdffreq, crds=True, vels=args.netcdf_vels, frcs=args.netcdf_frcs))
        '''
        mdparse.bak(out_nowater)
        logger.info("netcdf Reporter will write a no-water coordinate file %s every %i steps" %(out_nowater,netcdffreq))
        #toptraj = mdtraj.load(molecTopology)
        #top = toptraj.top
        top = mdtraj.Topology.from_openmm(simulation.topology)
        sel = [atom.index for residue in top.residues for atom in residue.atoms if (residue.name!="SOL") and (residue.name!="HOH")]
        simulation.reporters.append(mdtraj.reporters.NetCDFReporter(out_nowater, netcdffreq, atomSubset = sel))
        '''
    if args.dcd_report_interval > 0:
        mdparse.bak(out_dcd)
        logger.info("dcd Reporter will write to %s every %i steps" %(out_dcd, dcdfreq))
        simulation.reporters.append(mdtraj.reporters.DCDReporter(out_dcd, dcdfreq))
        '''
        mdparse.bak(out_nowater_dcd)
        logger.info("dcd Reporter will write a no-water coordinate file %s every %i steps" %(out_nowater_dcd, dcdfreq))
        #toptraj = mdtraj.load(molecTopology)
        #top = toptraj.top
        top = mdtraj.Topology.from_openmm(simulation.topology)
        sel = [atom.index for residue in top.residues for atom in residue.atoms if (residue.name!="SOL") and (residue.name!="HOH")]
        simulation.reporters.append(mdtraj.reporters.DCDReporter(out_nowater_dcd, dcdfreq, atomSubset = sel))

        #write out a nowater.pdb as topology input
        top2 = top.subset(sel)
        xyz0 = np.zeros([len(sel),3])
        traj2 = mdtraj.Trajectory(xyz0,topology=top2)
        traj2.save('output_nowater_top.pdb')
        top2omm = top2.to_openmm()
        '''
    if args.checkpoint_interval > 0: 
       simulation.reporters.append(app.CheckpointReporter(checkpointchk, checkfreq))
    #simulation.reporters.append(app.DCDReporter(out_dcd, writefreq))
    #simulation.reporters.append(mdtraj.reporters.HDF5Reporter(out_hdf5, writefreq, velocities=True))
    

    #============================#
    #| Finally Run!             |#
    #============================#
    t1 = time.time()
    if progressreport:
        Prog.t00 = t1
    #simulation.step(args.production)

    boxsizes = np.zeros([nblocks,3])
    TotalAcceptances = 0
    TotalMCMoves = 0
    for iblock in range(0,nblocks):
        logger.info("Starting block {}".format(iblock))
        start = time.time()
        simulation.step(blocksteps)
        end = time.time()
        logger.info('Took {} seconds for block {}'.format(end-start,iblock))
        thisbox = simulation.context.getState().getPeriodicBoxVectors()
        logger.info('Box size: {}'.format(thisbox)) 
        boxsizes[iblock,:] = [thisbox[0][0].value_in_unit(u.nanometer), thisbox[1][1].value_in_unit(u.nanometer), thisbox[2][2].value_in_unit(u.nanometer)]


        #if args.tension > 0:

        if args.tension is not None and args.restoring_scale != 0.:
            logger.info('=== Attempting area change manually ===')
            #--- Assumes args.tension in units of bar*nm ---
            kBT = u.AVOGADRO_CONSTANT_NA * u.BOLTZMANN_CONSTANT_kB * args.temperature * u.kelvin
            tension = args.tension*u.bar*u.nanometer*u.AVOGADRO_CONSTANT_NA
            Amax = 3.0 #relative to A0
            alphascale = args.restoring_scale #1140.
            alpha = alphascale/(1-np.sqrt(1/Amax))*u.bar*u.nanometer*u.AVOGADRO_CONSTANT_NA

            #--- Set up ---
            Eold = simulation.context.getState(getEnergy=True).getPotentialEnergy()
            logger.info('... Current energy: {}'.format( Eold.value_in_unit(u.kilojoule_per_mole) ))
            if np.random.random_sample() < 0.5:
                logger.info('... proposing to shrink area by {}...'.format(scaling))
                tmpf = 1/scaling
            else:
                logger.info('... proposing to expand area by {}...'.format(scaling))
                tmpf = scaling

            scaleNormal = 1.0/tmpf
            scaleTangent = tmpf**0.5
            
            #--- actually scale box ---
            oldbox = [thisbox[0], 
                        thisbox[1],
                        thisbox[2]]
            newbox = [thisbox[0]*scaleTangent,
                        thisbox[1]*scaleTangent,
                        thisbox[2]*scaleNormal]
            deltaArea = oldbox[0][0]*oldbox[1][1]*(tmpf-1.0)
            deltaL = oldbox[0][0]*(scaleTangent - 1)
            Aold = oldbox[0][0]*oldbox[1][1]
            Anew = newbox[0][0]*newbox[1][1]
            Lold = oldbox[0][0]
            Lnew = newbox[0][0]
            simulation.context.setPeriodicBoxVectors( newbox[0], newbox[1], newbox[2] )
            
            natoms = len(top.atoms) #using parmed gromacs topology object
            pos = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
            reference = np.zeros([natoms,3])
            for res in top.residues:
                atomids = [atom.idx for atom in res.atoms]
                com = np.mean( pos[atomids,:], 0 )
                reference[atomids,:] = com[None,:]

            newpos = pos + reference*np.array( [scaleTangent-1.0, scaleTangent-1.0, scaleNormal-1.0] )    
            simulation.context.setPositions(newpos)
            Enew = simulation.context.getState(getEnergy=True).getPotentialEnergy()

            #--- Monte Carlo Acceptance/Rejection ---
            w = Enew - Eold - tension*deltaArea + alpha*( (Lnew-L0)**2.0 - (Lold-L0)**2.0 )
            betaw = w/kBT
            logger.info('... MC transition energy: {}'.format(betaw))
            if betaw > 0 and np.random.random_sample() > np.exp(-betaw):
                #Reject the step
                logger.info('... Rejecting Step')
                simulation.context.setPeriodicBoxVectors( oldbox[0], oldbox[1], oldbox[2] )
                simulation.context.setPositions(pos)
            else:
                #Accept step
                logger.info('... Accepting Step')
                TotalAcceptances = TotalAcceptances + 1
            TotalMCMoves += 1

            #--- Print out final state ---
            logger.info('... box state after MC move:')
            logger.info( simulation.context.getState().getPeriodicBoxVectors() )
            logger.info('... acceptance rate: {}'.format(np.float(TotalAcceptances)/np.float(TotalMCMoves)))
            logger.info('  ')

        #finish membrane barostating


        if args.tension is not None and np.mod(iblock,100) != 0 and iblock!=nblocks-1:
            continue
        else:
            simulation.saveState(checkpointxml)
            positions = simulation.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions()
            app.PDBFile.writeFile(simulation.topology, positions, open(checkpointpdb, 'w')) 
            np.savetxt('boxdimensions.dat',boxsizes)

#END main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulation Properties')
    parser.add_argument("paramfile", default='params.in', type=str, help="param.in file")
    parser.add_argument("--deviceid", default=-1, type=int, help="GPU device id")
    parser.add_argument("--progressreport", default=True, type=bool, help="Whether or not to print progress report. Incurs small overhead")
    #parser.add_argument("simtime", type=float, help="simulation runtime (ns)")
    #parser.add_argument("Temp", type=float, help="system Temperature")
    #parser.add_argument("--NPT", action="store_true", help="NPT flag")
    #parser.add_argument("LJcut", type=float, help="LJ cutoff (Angstroms)")
    cmdln_args = parser.parse_args()


    #================================#
    #    The command line parser     #
    #================================#
    '''
    # Taken from MSMBulder - it allows for easy addition of arguments and allows "-h" for help.
    def add_argument(group, *args, **kwargs):
        if 'default' in kwargs:
            d = 'Default: {d}'.format(d=kwargs['default'])
            if 'help' in kwargs:
                kwargs['help'] += ' {d}'.format(d=d)
            else:
                kwargs['help'] = d
        group.add_argument(*args, **kwargs)

    print
    print " #===========================================#"
    print " #|    OpenMM general purpose simulation    |#"
    print " #| (Hosted @ github.com/leeping/OpenMM-MD) |#"
    print " #|  Use the -h argument for detailed help  |#"
    print " #===========================================#"
    print

    parser = argparse.ArgumentParser()
    add_argument(parser, 'pdb', nargs=1, metavar='input.pdb', help='Specify one PDB or AMBER inpcrd file \x1b[1;91m(Required)\x1b[0m', type=str)
    add_argument(parser, 'xml', nargs='+', metavar='forcefield.xml', help='Specify multiple force field XML files, one System XML file, or one AMBER prmtop file \x1b[1;91m(Required)\x1b[0m', type=str)
    add_argument(parser, '-I', '--inputfile', help='Specify an input file with options in simple two-column format.  This script will autogenerate one for you', default=None, type=str)
    cmdline = parser.parse_args()
    pdbfnm = cmdline.pdb[0]
    xmlfnm = cmdline.xml
    args = SimulationOptions(cmdline.inputfile, pdbfnm)
    '''

    # === RUN === #
    main(cmdln_args.paramfile, {}, deviceid=cmdln_args.deviceid, progressreport=cmdln_args.progressreport)

#End __name__

