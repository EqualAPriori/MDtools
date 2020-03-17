#/usr/bin/env python
#Utilities to parse/validate formatting for specifying system options
#
#

import numpy as np

# ===== Functions that parse system settings =====

def parseSys( system_specs ):
    """
    Parameters
    ----------
    system_specs : Ordered Dict
        read in from a yaml file
    """
    if "SystemOptions" in system_specs:
        system_options = system_specs["SystemOptions"]
    else:
        raise ValueError("SystemOptions key not found in param dictionary")

    return system_options

def parseBox( system_options ):
    """
    Parameters
    ----------
    system_options : Ordered Dict
        read in from a yaml file, under the "SystemOptions" key

    Returns
    -------
    [Lx, Ly, Lz] : list of three floats
    """

    if "unitcell_lengths" in system_options:
        if isinstance(system_options['unitcell_lengths'], list):
            box = np.array(system_options["unitcell_lengths"])
        else:
            box = np.array([float(length) for length in system_options['unitcell_lengths'].split()] )
    else:
        raise ValueError('unitcell_lengths key not found in system options')

    if len(box) == 3 and box.dtype.name.startswith('float'):
        Lx, Ly, Lz = box
    else:
        raise ValueError('Invalid unitcell_lengths entry: {}'.format(system_options["unitcell_lengths"]))
    
    return [Lx, Ly, Lz]

# ===== Functions that parse the force field =====

def parseBond( system_specs ):
    """
    Parameters
    ----------
    system_options : Ordered Dict
        read in from a yaml file

    Returns
    -------
    bond_ffs : list
        FF specification. each entry is a tuple (bondTypeName, atom type entry ..., parameters ...)
    """
    bond_ffs = []
    for key,description in system_specs.items():
        if key == "HarmonicBonds":
            """ currently wants format to be list of strings, 'atomTypeName1 atomTypeName2 length K'
                Assumed functional form [lammps/sim]: K( r - length )^2
            """
            bond_type = 'harmonic'

            for entry in description:
                entry = entry.split()
                atype_name1 = entry[0]
                atype_name2 = entry[1]
                length = float(entry[2])
                K = np.inf if entry[3].lower() in ['inf','infty','infinity'] else float(entry[3])
                #bond_ffs.append( (bond_type, atype_name1, atype_name2, length, K) )
                bond_ffs.append( {'bond_type':bond_type, 'atype1':atype_name1, 'atype2':atype_name2, 'length':length, 'K':K} )
        else: continue

    return bond_ffs

def parseGaussian( system_specs ):
    """
    Parameters
    ----------
    system_options : Ordered Dict
        read in from a yaml file

    Returns
    -------
    ag : n x n matrix 
        Gaussian smearing lengths
    u0 : n x n matrix
        bonding strenghts
    dist00 : n x n matrix
        Gaussian offset
    rcut : global cutoff
    indv_gaussians : list
        Extra specified Gaussians

    Notes
    -----
    assumed form: exp(-(r-r0^2)/4a^2) * u0/(4pia^2)^1.5 

    TODO: allow for additional Gaussians specified one-by-one; need to implement interpolating spline to do so
    """
   
    indv_gaussians = []
    rcut = None
    ag = []
    u0 = []
    dist0 = []

    if 'GaussianBase' in system_specs:
        "allowing for shorthand definition of full matrix of interactions"
        description = system_specs['GaussianBase']

        rcut = float( description['rcut'] )
        ag   = np.array([ list(map(float,row.split())) for row in description['a'] ])
        u0   = np.array([ list(map(float,row.split())) for row in description['u0'] ])
        if description['dist0'] == 0.0:
            dist0 = np.zeros(ag.shape)
        else:
            dist0 = np.array([ list(map(float,row.split())) for row in description['dist0'] ])

        if ag.shape == u0.shape and ag.shape == dist0.shape:
            print("GaussianBase matrices the same size, good")
        else:
            raise ValueError("GaussianBase matrix sizes are not equal")


    if 'Gaussian' in system_specs:
        "one-by-one addition of specific Gaussians"
        description = system_specs['Gaussian']
        for entry in description:
            entry = entry.split()
            atype_name1 = entry[0]
            atype_name2 = entry[1]
            awidth = float( entry[2] )
            u0strength = float(entry[3] )
            cutoff = float( entry[4] )

            indv_gaussians.append( ('Gaussian', atype_name1, atype_name2, awidth, u0strength, cutoff) ) 

    return [ag, u0, dist0, rcut, indv_gaussians]


def parseUExt( system_specs ):
    """
    Parameters
    ----------
    system_options : Ordered Dict
        read in from a yaml file

    Returns
    -------
    uext_ffs: list
        list of applied external potentials

    Notes
    -----
    assumed form: U * sin( 2*pi*Nperiod*(r-r0)/L )
    currently only apply one potential per species!
    """
    uext_ffs = []
    axis_dict = {'x':0,'X':0, 'y':1,'Y':1, 'z':2,'Z':2}

    for key,description in system_specs.items():
        if key == "ExternalPotential":
            """ currently wants format to be list of strings, 'atomTypeName1 Uext NPeriod axis displacement''
            """
            for entry in description:
                entry = entry.split()
                print(entry)
                atype_name = entry[0]
                Uext = float(entry[1])
                n_period = int(entry[2])
                axis = axis_dict[entry[3]]
                offset = float(entry[4])
                uext_ffs.append( ('UExt', [atype_name], Uext, n_period, axis, offset) )

        else: continue

    return uext_ffs


def parseElec( system_specs ):
    """
    Parameters
    ----------
    system_options : Ordered Dict
        read in from a yaml file

    Returns
    -------
    lb: float
    rcut: float
    ewld_tol: float
    aborn: list of floats

    Notes
    -----
    if aborn !=0, assuming smearing
    method defaults to pme. can implement alternative later
    """
    lb = None
    rcut = None
    aborn = None
    ewld_tol = 1e-5

    if 'Electrostatics' in system_specs: 
        description = system_specs['Electrostatics']
        
        lb = description['lb']
        rcut = description['rcut']
        ewld_tol = description['ewld_tol']
        aborn = [float(entry) for entry in description['aborn'].split()]
        
        
    return lb, rcut, ewld_tol, aborn


# ===== Functions that parse simulation run =====
def parseSimulation( system_specs ):
    """
    Parameters
    ----------
    system_options : Ordered Dict
        read in from a yaml file

    Returns
    -------
    sim_options : dictionary
        stores parsed simulation options

    Notes
    -----
    """
    sim_options = {}
    run_options = {}

    if 'SimulationOptions' in system_specs:
        description = system_specs['SimulationOptions']

        if 'dt' not in description:
            raise ValueError('Must specify dt')
        else:
            sim_options['dt'] = description['dt']

        if 'T' not in description:
            sim_options['T'] = None #shorthand for NVE
        elif description['T'] is None:
            sim_options['T'] = None #shorthand for NVE
        else:
            entry = description['T'].split()
            if entry[0].lower() in ['null','none']:
                sim_options['T'] is None
            else:
                sim_options['T'] = float(entry[0])
                sim_options['thermostat'] = entry[1]
                sim_options['t_damp'] = float(entry[2])
        
        if 'P' not in description:
            sim_options['P'] = None
        elif description['P'] is None:
            sim_options['P'] = None
        else:
            entry = description['P'].split()
            if entry[0].lower() in ['null','none']:
                sim_options['P'] = None
            else:
                sim_options['P'] = float( entry[0] )
                sim_options['barostat_freq'] = int(entry[1])
                if len(entry) == 2:
                    sim_options['barostat_axis'] = 'isotropic'
                else:
                    sim_options['barostat_axis'] = entry[2]
                    if entry[2] not in ['isotropic','iso','all','xyz']:
                        if entry[2] in [0,'x','X']:
                            sim_options['barostat_axis'] = 0
                        elif entry[2] in [1,'y','Y']:
                            sim_options['barostat_axis'] = 1
                        elif entry[2] in [2,'z','Z']:
                            sim_options['barostat_axis'] = 2
                        else:
                            raise ValueError('Unrecognized barostat axis: {}'.format(entry[2]))

        if 'tension' not in description:
            sim_options['tension'] = None
        elif description['tension'] is None:
            sim_options['tension'] = None
        else:       
            entry = description['tension'].split()
            if entry[0].lower() in ['null','none']:
                sim_options['tension'] = None
            else:
                sim_options['tension'] = float( entry[0] )
                sim_options['tension_freq'] = int( entry[1] )
                sim_options['tension_axis'] = entry[2]
                if entry[2] in [0,'x','X']:        
                    sim_options['tension_axis'] = 0
                elif entry[2] in [1,'y','Y']:
                    sim_options['tension_axis'] = 1
                elif entry[2] in [2,'z','Z']:
                    sim_options['tension_axis'] = 2
                else:
                    raise ValueError('Unrecognized tension axis: {}'.format(entry[2]))
                sim_options['tension_alphascale'] = float( entry[3] )
                sim_options['tension_Amax'] = float( entry[4] )

                #TODO: bounding tension

        if 'platform' not in description:
            sim_options['platform'] = None
        elif description['platform'] is None:
            sim_options['platform'] = None
        else:
            entry = description['platform'].split()
            if entry[0].lower() in ['null','none']:
                sim_options['platform'] = None
            elif entry[0].lower() in ['cuda','opencl']:
                sim_options['platform'] = entry[0]
                sim_options['device'] = int(entry[1])
                if len(entry) > 2:
                    sim_options['precision'] = entry[2]
    else:
        print("CAUTION No 'SimulationOptions' section found")


    if "RuntimeOptions" not in system_specs:
        print("CAUTION No 'RuntimeOptions' section found")
    else:
        description = system_specs['RuntimeOptions']
        if 'initial' in description:
            run_options['initial'] = description['initial']
        
        if 'nsteps_min' in description:
            run_options['nsteps_min'] = description['nsteps_min']
        else:
            run_options['nsteps_min'] = 0

        if 'ntau_equil' in description:
            run_options['ntau_equil'] = description['ntau_equil']
        else:
            run_options['ntau_equil'] = 0

        if 'ntau_prod' in description:
            run_options['ntau_prod'] = description['ntau_prod']
        else:
            run_options['ntau_prod'] = 0

        if 'write_freq' in  description: #write_freq is also given in terms of tau
            run_options['write_freq'] = description['write_freq']
        else:
            run_options['write_freq'] = 0

        if 'protocol' in description:
            run_options['protocol'] = description['protocol']
        else:
            run_options['protocol'] = 'simple'

        if 'tension' in sim_options:
            if sim_options['tension'] is not None and 'tension' not in run_options['protocol']:
                print("detected applied tension, using tension protocol")
                run_options['protocol'] = 'tension'

    return sim_options, run_options


