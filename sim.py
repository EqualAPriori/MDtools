################################################################
# Kevin Shen, 2019                                             #
# kevinshen@ucsb.edu                                           #
#                                                              #
# General purpose openMM simulation script.                    #
# Allows for (verlet, langevin); barostats; LJPME              #
# simulation protocol:                                         #
#   1) equilibrate                                             #
#   2) production run                                          #
#                                                              #
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

# OpenMM Imports
import simtk.openmm as mm
import simtk.openmm.app as app

# ParmEd & MDTraj Imports
from parmed import gromacs
#gromacs.GROMACS_TOPDIR = "/home/kshen/SDS"
from parmed.openmm.reporters import NetCDFReporter
from parmed import unit as u
import parmed as pmd
#import mdtraj

# Custom Tools
import mdparse




def add_barostat(system,args):
    if args.pressure <= 0.0:
        logger.info("This is a constant volume (NVT) run")
    else:
        logger.info("This is a constant pressure (NPT) run at %.2f bar pressure" % args.pressure)
        logger.info("Adding Monte Carlo barostat with volume adjustment interval %i" % args.nbarostat)
        logger.info("Anisotropic box scaling is %s" % ("ON" if args.anisotropic else "OFF"))
        if args.anisotropic:
            logger.info("Only the Z-axis will be adjusted")
            barostat = mm.MonteCarloAnisotropicBarostat(Vec3(args.pressure*u.bar, args.pressure*u.bar, args.pressure*u.bar), args.temperature*u.kelvin, False, False, True, args.nbarostat)
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

    incoord         = args.incoord
    out_pdb         = args.outpdb
    out_netcdf      = args.outnetcdf
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
        pdbfreq    = int(args.pdb_report_interval/dt)
        checkfreq  = int(args.checkpoint_interval/dt)
        #simtime    = int( simtime ) #nanoseconds; make sure division is whole... no remainders...
        blocksteps = int(args.block_interval/dt)   #1e6, steps per block of simulation 
        nblocks    = args.nblocks #aiming for 1 block is 1ns
    else:
        reportfreq = args.report_interval
        netcdffreq = args.netcdf_report_interval
        pdbfreq    = args.pdb_report_interval
        checkfreq  = args.checkpoint_interval
        blocksteps = args.block_interval
        nblocks    = args.nblocks 

    if quiktest==True:
        reportfreq = 1
        blocksteps = 10
        nblocks = 2

    # === Start Making System === #
    top = gromacs.GromacsTopologyFile(top_file, defines=defines)
    gro = gromacs.GromacsGroFile.parse(box_file)
    top.box = gro.box

    constr = {None: None, "None":None,"HBonds":app.HBonds,"HAngles":app.HAngles,"AllBonds":app.AllBonds}[args.constraints]   
    system = top.createSystem(nonbondedMethod=app.PME, ewaldErrorTolerance = args.ewald_error_tolerance,
                        nonbondedCutoff=args.nonbonded_cutoff*u.nanometers,
                        rigidWater = args.rigid_water, constraints = constr)
                        
    nbm = {"NoCutoff":mm.NonbondedForce.NoCutoff, "CutoffNonPeriodic":mm.NonbondedForce.CutoffNonPeriodic,
                "Ewald":mm.NonbondedForce.Ewald, "PME":mm.NonbondedForce.PME, "LJPME":mm.NonbondedForce.LJPME}[args.nonbonded_method]

    ftmp = [f for ii, f in enumerate(system.getForces()) if isinstance(f,mm.NonbondedForce)]
    fnb = ftmp[0]
    fnb.setNonbondedMethod(nbm)
    logger.info("Nonbonded method ({},{})".format(args.nonbonded_method, fnb.getNonbondedMethod()) )
    if (not args.dispersion_correction) or (args.nonbonded_method=="LJPME"):
        logger.info("Turning off tail correction...")
        fnb.setUseDispersionCorrection(False)
        logger.info("Check dispersion flag: {}".format(fnb.getUseDispersionCorrection()) )


    # === Integrator, Barostat, Additional Constraints === #
    integrator = set_thermo(system,args)

    if not hasattr(args,'constraints') or (str(args.constraints) == "None" and args.rigidwater == False):
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
            platform.setPropertyDefaultValue("CudaDeviceIndex", device)
            #platform.setPropertyDefaultValue("OpenCLDeviceIndex", device)
        else:
            logger.info("Using the default (fastest) device")
    else:
        logger.info("Using the default (fastest) device, or not using CUDA nor OpenCL")

    if "CudaPrecision" in platform.getPropertyNames() and (platform.getName()=="OpenCL" or platform.getName()=="CUDA"):
        platform.setPropertyDefaultValue("CudaPrecision", args.cuda_precision)
    else:
        logger.info("Not setting precision")
        args.deactivate("cuda_precision",msg="Platform does not support setting cuda_precision.")

    # === Create Simulation === #
    logger.info("Creating the Simulation object")
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


    #print platform we're using
    mdparse.printcool_dictionary({i:simulation.context.getPlatform().getPropertyValue(simulation.context,i) for i in simulation.context.getPlatform().getPropertyNames()},title="Platform %s has properties:" % simulation.context.getPlatform().getName())

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
        if incoord.split(".")[1]=="pdb":
            pdb = pmd.load_file(incoord)
            simulation.context.setPositions(pdb.positions)
        elif incoord.split(".")[1]=="xml":
            simulation.loadState(incoord)
        else:
            logger.info("Error, can't handle input coordinate filetype")
            
        simulation.context.applyConstraints(args.constraint_tolerance) #applies constraints in current frame.
        logger.info("Initial potential energy is: {}".format(simulation.context.getState(getEnergy=True).getPotentialEnergy()) )

        if args.integrator != 'mtsvvvr':
            eda = mdparse.EnergyDecomposition(simulation)
            eda_kcal = OrderedDict([(i, "%10.4f" % (j/4.184)) for i, j in eda.items()])
            mdparse.printcool_dictionary(eda_kcal, title="Energy Decomposition (kcal/mol)")

        # Minimize the energy.
        if args.minimize:
            logger.info("Minimization start, the energy is:", simulation.context.getState(getEnergy=True).getPotentialEnergy())
            simulation.minimizeEnergy()
            logger.info("Minimization done, the energy is", simulation.context.getState(getEnergy=True).getPotentialEnergy())
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

    for iblock in range(0,nblocks):
        logger.info("Starting block {}".format(iblock))
        start = time.time()
        simulation.step(blocksteps)
        end = time.time()
        logger.info('Took {} seconds for block {}'.format(end-start,iblock))

        simulation.saveState(checkpointxml)
        positions = simulation.context.getState(getPositions=True).getPositions()
        app.PDBFile.writeFile(simulation.topology, positions, open(checkpointpdb, 'w'))    
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

