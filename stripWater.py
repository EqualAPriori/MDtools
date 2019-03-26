#Monolayer -- water: OPC4, LJcut=12, LJPME, NO tail correction
import numpy as np
import sys
import os
import shutil
print("loading mdtraj")
import mdtraj as md
print("finished loading mdtraj")
import timeit
import argparse as ap


class cd:
    """Context manager for changing the current working directory. Usage: with cd("pathname")...:
    """
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def read(topname,trajname):
    """ Read and strips the water from a trajectory file. Uses MDtraj

    Parameters
    ----------
    topname : str
        name of topology file
    trajname : str
        name of trajectory file, file type should be something MDtraj can process with mdtraj.load(trajname,top)

    Returns
    -------
    partialtraj
        a mdtraj trajectory object with the waters stripped

    Notes
    -----
    Uses mdtraj as the main engine doing the parsing. Can load whatever files mdtraj can parse with md.load
    Detects waters by their residue name, "SOL" or "HOH"

    References
    ----------
    [1] http://mdtraj.org/1.9.0/
    """
    toptraj = md.load(topname)
    print("loading topology")
    start = timeit.default_timer()
    top = toptraj.top
    print("took {}sec".format(timeit.default_timer()-start))
    sel = [atom.index for residue in top.residues for atom in residue.atoms if (residue.name!="SOL") and (residue.name!="HOH")] 
    
    print("trying to load trajectory")
    start = timeit.default_timer()
    #for chunk in md.iterload(trajname, atom_indices=sel, top=top):
    #    print(chunk)
    fulltraj = md.load(trajname, atom_indices=sel, top=top)
    print("took {}sec".format(timeit.default_timer()-start))
    #partialtraj = fulltraj.atom_slice(sel)
    #return partialtraj
    return fulltraj

def writeNetcdf(traj,filename):
    traj.save_netcdf(filename)
    # also need tow rite out topology of new system, i.e. to at least load it
    # currently doesn't write out connectivity...
    tmp = filename.split('.')
    tmp = '.'.join(tmp[:-1])+'_top.pdb'
    traj[0].save_pdb(tmp)

def writeDCD(traj,filename):
    traj.save_dcd(filename)
    # also need tow rite out topology of new system, i.e. to at least load it
    # currently doesn't write out connectivity...
    tmp = filename.split('.')
    tmp = '.'.join(tmp[:-1])+'_top.pdb'
    traj[0].save_pdb(tmp)
    
    

def writeLammps(traj,filename):
    traj.save_lammpstrj(filename)

def main(topname,trajname,outname,fmt=None):
    """ Strips the water from a trajectory file. Uses MDtraj

    Parameters
    ----------
    topname : str
        name of topology file
    trajname : str
        name of trajectory file, file type should be something MDtraj can process with mdtraj.load(trajname,top)
    outname : str
        name of desired output file
    fmt : str, optional
        case insensitive, defaults to netcdf. Currently supports ["netcdf","lammps"]

    Returns
    -------
    """
    fileformats = ["nc","netcdf","lammps","lammpstrj","dcd"]

    if fmt is None:
        fmt = outname.split('.')[-1]

    fmt = fmt.lower()
    assert fmt in fileformats, "Format should be one of (case-insensitive) supported ones: "+", ".join(fileformats)
    trj = read(topname,trajname)

    start = timeit.default_timer()
    print("writing...")
    if fmt in ["nc","netcdf"]:
        writeNetcdf(trj,outname)
    if fmt in ["dcd"]:
        writeDCD(trj,outname)
    elif fmt in ["lammps","lammpstrj"]:
        writeLammps(trj,outname)
    print("took {} seconds".format(timeit.default_timer()-start))

if __name__ == "__main__":
    
    parser = ap.ArgumentParser(description="Strip Water from a trajectory")
    parser.add_argument('top', type=str, help = "topology file")
    parser.add_argument('traj', type=str, help = "trajectory file")
    parser.add_argument('output', type=str, help = "output file")
    args = parser.parse_args()

    print("Parsing...\n")
    print("topology file \t= {}".format( args.top ) )
    print("netcdf file \t = {}".format( args.traj ) )
    print("output file \t = {}".format( args.output ) )
    main( args.top, args.traj, args.output )

    ''' 
    print("Parsing...\n")
    print("topology file \t= {}".format( sys.argv[1] ) )
    print("netcdf file \t = {}".format( sys.argv[1] ) )
    print("output file \t = {}".format( sys.argv[1] ) )
    main(sys.argv[1],sys.argv[2],sys.argv[3])
    '''
