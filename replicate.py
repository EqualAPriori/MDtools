# 2019.04.04 Kevin Shen
# simple script for replicating a cell *exactly* by some integer xyz amount
# Currently assumes that all residues of same resname are *contiguous* in the xyz!
# i.e. not suitable for chains of residues!
#
# ===== Imports ===== #
import mdtraj
import numpy as np
import argparse as ap

# ===== Parsing ===== #
parser = ap.ArgumentParser(description="Replicate a rectangular unit cell")
parser.add_argument('infile', type=str, help = "coordinate file to replicate")
parser.add_argument('x', type=int, help = "times to replicate in x-direction")
parser.add_argument('y', type=int, help = "times to replicate in y-direction")
parser.add_argument('z', type=int, help = "times to replicate in z-direction")
parser.add_argument('bx', type=float, help = "box size in x-direction")
parser.add_argument('by', type=float, help = "box size in y-direction")
parser.add_argument('bz', type=float, help = "box size in z-direction")
parser.add_argument('--outfile', type=str, default="replicated.xyz", help = "output file")
args = parser.parse_args()

print("Parsing...\n")
print("input file\t= {}".format( args.infile ) )
print("native box [{},{},{}]".format( args.bx, args.by, args.bz ))

print("replicating [{},{},{}]".format( args.x, args.y, args.z ))
print("output file\t={}".format( args.outfile ))
parser = ap.ArgumentParser

infile = args.infile #'chk_00.pdb'
outfile = args.outfile

# ===== Utility Functions ===== #
def wrap(xyz,box):
    for dim in range(3):
        xyz[:,dim] = np.mod(xyz[:,dim],box[dim])
    return(xyz)

images = [args.x, args.y, args.z]
box    = [args.bx, args.by, args.bz]
trj = mdtraj.load(infile)
#trj.image_molecules(inplace=True, make_whole=True) #make sure wrapped
xyz = wrap(trj.xyz[0], box)

# ===== Determine topology =====
print(" === Determining Topology === ")
rlist = []          #list of (unique) residue names
nlist = []          #list of counter of # of residues of a certain resname
atomsInRes = []     #list of list of atoms in residue
for r in trj.topology.residues:
    if r.name not in rlist:
        rlist.append(r.name)                #append residue name
        nlist.append(0)                     #start counter of number atoms in this residue
        atomsInRes.append( list(r.atoms) )  #append list of atoms
    nlist[-1] += 1                          #counter for # residues with this resname
print("Residues\t\t{}".format(rlist))
print("Multiplicities\t{}".format(nlist))

# ===== Create new topology, with appropriate replication =====
# strategy is to go residue by residue, adding all its replicated images, before moving on
print(" === Creating New Topology === ")
newtop = mdtraj.Topology()

#our convention, only one chain, not important anyway
ch = newtop.add_chain()

for itype,restype in enumerate(rlist):  #iterate of the different resnames
    print("Adding residue {}".format(restype))
    for img in range(np.prod(images)):  #residue types are still contiguous in memory!
        for ir in range( nlist[itype] ):#iterate over number residues of this resname
            res = newtop.add_residue(restype, ch)
            for a in atomsInRes[itype]: #add the atoms that go into this residue
                newa = newtop.add_atom(a.name, a.element, res)

# ===== Replicate coordinates appropriately =====
print(" === Replicating Coordinates === ")
newxyz = np.zeros( [newtop.n_atoms,3] )
atomsAdded = 0      #counter of atoms added in new topology
atomsFromOld = 0    #counter of atoms considered from old topology
for itype, restype in enumerate(rlist): #iterate over different residue names
    print("Adding residue {}".format(restype))
    nAtomsOfTypeRes = nlist[itype] * len(atomsInRes[itype])     #total # atoms of the residues of given resname
    tempxyz = xyz[atomsFromOld:atomsFromOld + nAtomsOfTypeRes]  #assume coordinates are contiguous!
    for ix in range(images[0]):                                 #Loop over images
        for iy in range(images[1]):
            for iz in range(images[2]):
                newxyz[atomsAdded:atomsAdded+nAtomsOfTypeRes,:] = tempxyz + [ix*box[0], iy*box[1], iz*box[2]]
                atomsAdded += nAtomsOfTypeRes
    atomsFromOld += nAtomsOfTypeRes


# ===== Save new trajectory/coordinate file =====
print(" === Saving Trajectory === ")
newtraj = mdtraj.Trajectory(newxyz, newtop, unitcell_lengths = np.array(box) * np.array(images, dtype=float), unitcell_angles=[90.0,90.0,90.0])
newtraj.save(outfile)

