# 2019.12.20
# (C) kevinshen@ucsb.edu
# Lightweight and transportable way for encoding a system's topology
# Allows one to easily set up both Sim and OpenMM systems
#
# Handles the parsing of a lightweight yaml specification of a coarse-grained system, and validation
#
# Things to include:
# - Units
# - Dimension
# - 
#
#Desired interface:
# 1) AtomType
# 2) ResType
# 3) MolType
#
#As topology is adding molecules... need to keep track of
#atomlist: each atom entry knows its atomtype, restype, moltype,
#mol list: each molecule knows its atoms, bonds

import sys
import ruamel.yaml as yaml
import collections
#import numpy as np
VVerbose = False

class AtomType(object):
    def __init__(self,name,mass=1.0,charge=0.0,element=None):
        if mass < 0.0:
            raise ValueError("Mass must be float > 0")
        if not isinstance(charge,float):
            raise ValueError("Charge for atomType {} must be a float".format(name))
        
        self.name = name
        self.mass = mass
        self.charge = charge
        self.element = element

    def __repr__(self):
        return self.name

    def New(self,Parent=None):
        return Atom(self, Parent = Parent)

class ResType(list):
    def __init__(self,name,atomList=[]):
        list.__init__(self)
        self.name = name
        self.extend( atomList )
    def __repr__(self):
        return "{}:{}".format(self.name, list.__repr__(self))

class MolType(list):
    """A molecule type is a list of residues"""
    """Todo:
        1) add bonding
        2) validation that bond indices are valid...
    """
    def __init__(self,name,resList=[]):
        list.__init__(self)
        self.name = name
        self.extend( resList )
        self.bonds = []
        self.atoms = []
        self.numAtomsInMol = 0
    def bond(self, topology, bondlist=None):
        "if bondlist is None, assume simple bonding"
        if bondlist is None:
            print("...Assuming simple bonding of this molecule")
            #NEED TO KNOW EXISTING RESIDUES! EITHER NEED TO FEED IN A WORLD, OR RESLIST NEEDS TO BE ACTUAL RESIDUE OBJECTS
            numAtomsInMol = 0
            previousBackboneIndex = 0
            backboneIndex = 0
            for ir, res in enumerate(self):
                resname = res.name
                if VVerbose: print("...Working on {},{}".format(ir,resname))
                if ir > 0: #need to connect residue to previous residue, from first atom to first atom
                    if VVerbose: print("......Bonding {} to {}".format(previousBackboneIndex,backboneIndex))
                    self.bonds.append( (previousBackboneIndex, backboneIndex) )
                

                numAtomsInMol += 1
                #assume residue is linearly bonded 
                numAtomsInRes = len(topology.ResTypes[resname])
                for ia in range(backboneIndex, backboneIndex + numAtomsInRes - 1):
                    if VVerbose: print("......Bonding {} to {}".format(ia,ia+1))
                    self.bonds.append( (ia, ia+1) )
                    numAtomsInMol += 1

                previousBackboneIndex = backboneIndex
                backboneIndex = backboneIndex + numAtomsInRes #should be numAtomsInMol
                #print("{}, {}".format(backboneIndex,numAtomsInMol))
        else:
            'should be a list of pairs(tuples)'
            print("...Attempting to read in prescribed bonding")
            for bond in bondlist:
                if isinstance(bond,tuple) and len(bond)==2:
                    if bond[0]==bond[1]:
                        raise ValueError("...bond {} tries bonding an atom to itself".format(bond))
                    if isinstance(bond[0],int) and isinstance(bond[1],int):
                        self.bonds.append( bond )
                    else:
                        raise ValueError("...bond must be specified between integers representing *site type* (i.e. index of atoms in molecule, starting with zero for first atom in molecule type)")


    def __repr__(self):
        return "{}:{}".format(self.name, list.__repr__(self))
    def New(self):
        return Mol(self)


class Topology(object):
    def __init__(self,topfile=None):
        self.AtomTypes = collections.OrderedDict()
        self.ResTypes = collections.OrderedDict()
        self.MolTypes = collections.OrderedDict()
        
        self.Atoms = []
        self.Residues = []
        self.Molecules = []

        self.BondList = []
        self.AtomsInMol = []

        if topfile is not None:
            self.load(topfile)

    def save(self,outfile):
        """Save topology"""
        """
        Right now depends on having read in a topology first...
        would be nice if it could "aggregate/summarize" a topology by itself (i.e. if one creates topology manually instead of via the loader
        """
        print("Need to implement")
        with open(outfile,'w') as f:
            yaml.dump(self.myyaml, f)
            #yaml.dump(self.myyaml, f, Dumper=yaml.RoundTripDumper)

    def load(self,topfile):
        # --- Read in Data ---
        # Assumed format: (note can be expressed in many equivalent ways in yaml)
        """
        {
         Atoms:[ [anameA,massA,chargeA], [anameB,massB,chargeB], ... ],
         Residues: [ [resname1,[atomname1_1, numatom1_1],[atomname1_2, numatom1_2],...], ...]
         Molecules: [ [molnameX, [resnameX_1, numresX_1],[resnameX_2, numresX_2],...], ...]
         System: [ [molnameI, nummolI], [molnameII, nummolII], ... ]
        }
        """
        with open(topfile,'r') as stream:
            self.myyaml = yaml.YAML()
            topdata = yaml.load(stream, Loader=yaml.RoundTripLoader)
             
        print('===== Read in Data =====')
        for key,val in topdata.items():
            print('{}: {}'.format(key,val))
        print("\n")

        self.createSys(topdata)

    def createSys(self,topdata):
        print("===== Trying to create system from given topology dictionary =====")
        self.topdict = topdata #may be a ruamel.yaml object with extra commenting bonuses

        # --- Try create all the constituents, validating input as we go ---
        # ... Atoms ...
        try:
            print("Attempting to read in atom types: {}".format(self.AtomTypes))
            for atomTypeEntry in topdata["Atoms"]:
                name, mass, charge = self.parseAtom(atomTypeEntry)
                #self.AtomTypes.append( AtomType(name, mass, charge) )
                self.AtomTypes[name] = AtomType(name, mass, charge)
            print("Read in atom types: {}".format(self.AtomTypes))
        except:
            print("Unknown error while adding atoms")
        # ... Residues ...
        try:
            print("\nAttempting to read in residue types: {}".format(self.ResTypes))
            for residueTypeEntry in topdata["Residues"]:
                name, atomList = self.parseResidue(residueTypeEntry)
                self.ResTypes[name] = ResType(name, atomList)
                #self.ResTypes.append( ResType(name, atomlist) )
            print("Read in residue types: {}".format(self.ResTypes))
        except:
            print("Unknown error while adding residues")
        
        # ... Molecules ...
        try:
            print("\nAttempting to read in molecule types: {}".format(self.MolTypes))
            for moleculeTypeEntry in topdata["Molecules"]:
                name, residueList = self.parseMolecule(moleculeTypeEntry)
                self.MolTypes[name] = MolType(name, residueList)
                #self.MolTypes.append( MolType(name, residueList) )
                self.MolTypes[name].bond(self)
            print("Read in molecule types: {}".format(self.MolTypes))
        except:
            print("Unknown error while adding molecules")

        # ... System ...
        try:
            print("\nAttempting to add molecules to system")
            for moleculeType in topdata["System"]:
                name,numMol = self.parseSystem(moleculeType)
                for ii in range(numMol):
                    self.addMolecule(name)
            print("Added molecules to system")
        except:
            print("Unknown error while constructing system")

    def parseAtom(self,atomTypeEntry):
        print("...parsing {}".format(atomTypeEntry))
        name, mass, charge = atomTypeEntry[0], float(atomTypeEntry[1]), float(atomTypeEntry[2])
        return name, mass, charge

    def parseResidue(self,residueTypeEntry):
        print("...parsing {}".format(residueTypeEntry))
        name = residueTypeEntry[0]
        atoms = residueTypeEntry[1:]
        
        atomList = []
        for atom in atoms:
            print("......processing {}".format(atom))
            aname = atom[0]
            anum = int(atom[1])
            if anum < 0:
                raise ValueError("residue {}'s atom {} number must be positive integer, is right now {}".format(name,aname,anum))

            if aname in self.AtomTypes.keys():    
                atomList.extend( anum*[self.AtomTypes[aname]] )
            else:
                raise ValueError("residue {} has an unrecognized atom {}".format(name, aname))

        return name, atomList

    def parseMolecule(self,moleculeTypeEntry):
        print("...parsing {}".format(moleculeTypeEntry))
        name = moleculeTypeEntry[0]
        residues = moleculeTypeEntry[1:]
        
        residueList = []
        for residue in residues:
            print("......processing {}".format(residue))
            resname = residue[0]
            resnum = int(residue[1])
            if resnum < 0:
                raise ValueError("molecule {}'s residue {} number must be positive integer, is right now {}".format(name,resname,resnum))

            if resname in self.ResTypes.keys():
                residueList.extend( resnum*[self.ResTypes[resname]] )
            else:
                raise ValueError("molecule {} has an unrecognized residue {}".format(name,resname))
        
        return name, residueList

    def parseSystem(self,systemMoleculeEntry):
        name = systemMoleculeEntry[0]
        numMol = int(systemMoleculeEntry[1])
        return name, numMol

    def addAtom(self,aname):
        if aname in self.AtomTypes.keys():
            self.Atoms.append(aname)
        else:
            raise ValueError("Unrecognized atom type {}, can't add".format(aname))

    def addResidue(self,resname):
        if resname in self.ResTypes.keys():
            self.Residues.append(resname)
            for atom in self.ResTypes[resname]:
                self.Atoms.append(atom.name)
        else:
            raise ValueError("Unrecognized residue type {}, can't add".format(resname))

    def addMolecule(self,molname):
        if molname in self.MolTypes.keys():
            print('adding {}'.format(molname))
            oldNumAtoms = len(self.Atoms)
            self.Molecules.append(molname)
            for residue in self.MolTypes[molname]:
                self.addResidue(residue.name)
            newNumAtoms = len(self.Atoms)

            #self.AtomsInMol.append( (len(self.Atoms) - self.MolTypes[molname].numAtomsInMol, len(self.Atoms)  )) 
            #self.AtomsInMol.append( np.arange(oldNumAtoms,newNumAtoms) )
            self.AtomsInMol.append( list( range(oldNumAtoms,newNumAtoms) ) )
            self.addBonds(molname,atomsInMol = self.AtomsInMol[-1],bonds=None)
        else:
            raise ValueError("Unrecognized molecule type {}, can't add".format(molname))

    def addBonds(self, molname, atomsInMol, bonds=None):
        "adds bonds for the latest molecule"
        "bond=None is default, simple bonding behavior"
        moltype = self.MolTypes[molname]
        if VVerbose: print(atomsInMol)
        for bond in moltype.bonds:
            #print("...relative bond: {}".format(bond))
            currentBond = (atomsInMol[bond[0]], atomsInMol[bond[1]])
            self.BondList.append(currentBond)
            if VVerbose: print("adding (absolute index) bond {}".format(currentBond))






