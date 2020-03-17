# 2019.12.20
# (C) kevinshen@ucsb.edu
# Lightweight and transportable way for encoding a system's topology
# Allows one to easily set up both Sim and OpenMM systems
#
# Handles the parsing of a lightweight yaml specification of a coarse-grained system, and validation
#
# Kind of an overloading abuse, but also use Topology object, which stores the loaded yaml file, to store other force field and system options, which aren't strictly topology related 
#
# Things to include:
# - Units
# - Dimension
# - 
#
#Desired interface:
# 1) atom_types
# 2) residue_types
# 3) molecule_types
#
#As topology is adding molecules... need to keep track of
#atomlist: each atom entry knows its atomtype, restype, moltype,
#mol list: each molecule knows its atoms, bonds
#
# Challenge: generally, bonding should be defined on a site-site basis...
#   i.e. ideally, for each molecule, define bonded interactions one-by-one...
#        would need a more advanced filtering syntax to define...
#
#   temporary solution 1 -- choose by atom type...
#     bonding export will work by:
#     1) looping over bond (i,j) pairs
#     2) loop over defined bonds, and apply ones where the atom types match
#
#   temporary solution 2 -- hybrid by defining a bond as (molecule, atomType, atomType)
#       reasonable because in CG model, unlikely that (molecule, atomType, atomType) will lead to redundancies. I.e. might expect C2-C2 bond to be different b/t diff. molecules, but most likely all the same within a molecule.
#     bonding export will work by:
#     1) looping over molecules
#     2) loop over bond site pairs
#     3) loop over defined bonds, see if any match a (molecule,site1,site2) specification
#   



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
            raise ValueError("Charge for atom_type {} must be a float".format(name))
        
        self.name = name
        self.mass = mass
        self.charge = charge
        self.element = element

    def __repr__(self):
        return self.name

    def New(self,parent=None):
        return Atom(self, parent = parent)

class Atom(object):
    def __init__ (self, atom_type, parent=None):
        """
        Creates a new atom based on atom_type
        """
        self.atom_type = atom_type
        self.parent = parent
        self.ind = -1 #not initialized yet!
    
    def __repr__(self):
        return "{}:{}".format(self.ind,self.atom_type)

    def __getattr__(self, name):
        if name == 'name':
            return self.atom_type.name
        elif name == 'mass':
            return self.atom_type.mass
        elif name =='charge':
            return self.atom_type.charge
        else:
            AttributeError(name)

class ResType(list):
    """A residue type is a list of atom types"""
    def __init__(self,name,atom_type_list=[]):
        list.__init__(self)
        self.name = name
        self.extend( atom_type_list )
    def __repr__(self):
        return "{}:{}".format(self.name, list.__repr__(self))

    def New(self,parent=None):
        return Res(self, parent = parent)

class Res(list):
    def __init__ (self, res_type, parent=None):
        """
        Creates a new residue based on res_type
        """
        self.res_type = res_type
        self.parent = parent
        self.ind = -1 #not initialized yet!
        
        self.extend( [atom_type.New(parent = self) for atom_type in res_type] )

    def __repr__(self):
        return "{}:{}".format(self.ind,self.res_type)

    def __getattr__(self, name):
        if name == 'name':
            return self.res_type.name
        else:
            AttributeError(name)

class MolType(list):
    """A molecule type is a list of residues types"""
    """Todo:
        1) add bonding
        2) validation that bond indices are valid...
    """
    def __init__(self,name,res_type_list=[]):
        list.__init__(self)
        self.name = name
        self.extend( res_type_list )
        self.bond_mode = None
        self.bonds = []
        self.atoms = [a for res_type in self for a in res_type]
        self.num_atoms_in_mol = 0

    def Bond(self, topology, bonds=None):
        """
        Parameters
        ----------
        topology : chemlib topology
        bonds: None or list of bond pairs
            if `bonds` is None, assume simple bonding

        Notes
        -----
        these bond indices are intra-molecule "site indices"
        """
        if bonds is None or bonds in ['simple','Simple']:
            self.bond_mode = bonds
            print("...Assuming simple bonding of this molecule")
            #NEED TO KNOW EXISTING RESIDUES! EITHER NEED TO FEED IN A WORLD, OR RESLIST NEEDS TO BE ACTUAL RESIDUE OBJECTS
            num_atoms_in_mol = 0
            previous_backbone_index = 0
            backbone_index = 0
            for ir, res in enumerate(self):
                resname = res.name
                if VVerbose: print("...Working on res {},{}".format(ir,resname))
                if ir > 0: #need to connect residue to previous residue, from first atom to first atom
                    if VVerbose: print("......Bonding site {} to site {} in mol {}".format(previous_backbone_index,backbone_index, self.name))
                    self.bonds.append( Bond( (previous_backbone_index, backbone_index) ) )
                num_atoms_in_mol += 1
                #assume residue is linearly bonded 
                num_atoms_in_res = len(topology.residue_types[resname])
                for ia in range(backbone_index, backbone_index + num_atoms_in_res - 1):
                    if VVerbose: print("......Bonding site {} to site {} in mol {}".format(ia,ia+1, self.name))
                    self.bonds.append( Bond([ia, ia+1]) )
                    num_atoms_in_mol += 1

                previous_backbone_index = backbone_index
                backbone_index = backbone_index + num_atoms_in_res #should be num_atoms_in_mol
                #print("{}, {}".format(backbone_index,num_atoms_in_mol))
        else:
            self.bond_mode = 'list'
            'should be a list of pairs(tuples)'
            if isinstance(bonds,list):
                print("...Attempting to read in prescribed bonding")
                if len(bonds) == 0:
                    print("...empty bond list!")
                    self.bond_mode = 'unprescribed'
                for bond in bonds:
                    if isinstance(bond,tuple) and len(bond)==2:
                        if bond[0]==bond[1]:
                            raise ValueError("...bond {} tries bonding an atom to itself".format(bond))
                        if isinstance(bond[0],int) and isinstance(bond[1],int):
                            self.bonds.append( Bond(bond) )
                        else:
                            raise ValueError("...bond must be specified between integers representing *site type* (i.e. index of atoms in molecule, starting with zero for first atom in molecule type)")
            else: 
                raise ValueError("...bonding specification is neither a known keyword ('simple','Simple') nor a list")

    def __repr__(self):
        return "{}:{}".format(self.name, list.__repr__(self))
    def New(self):
        return Mol(self)

class Mol(list):
    def __init__(self,mol_type):
        """
        Creates a new molecule based on mol_type

        Parameters
        ----------
        mol_type : MolType object

        Notes
        -----
        TODO:
        1) Bonded
        2) AtomBonds
        3) BondMap
        4) WithinBonOrd

        5) internal bondlist (stype to stype)
        6) absolute bondlist indices
        """
        list.__init__(self)
        self.mol_type = mol_type
        self.ind = -1 #not initialized yet
        
        #make new residues from residue types
        self.extend( [residue_type.New(parent = self) for residue_type in mol_type] )
        self.atoms = [a for res in self for a in res]
        self.bonds = self.mol_type.bonds
        self.bond_atom_ind = [ [self.atoms[b[0]].ind, self.atoms[b[1]].ind] for b in self.bonds ]

    def __repr__(self):
        return "{}:{}".format(self.ind, self.mol_type)
    
    def __getattr__(self,name):
        if name == 'name':
            return self.mol_type.name
        else:
            raise AttributeError(name)

class Bond(tuple):
    """Stores a bond for a moltype, as a tuple of the intra-molecule site indices
    Parameters
    ----------
    bondpair : tuple of indices
    bondlength : None or float
        None is free bond, float is fixed bond
    """

    def __new__(cls, bondpair, length = None):
        return super(Bond, cls).__new__(cls, tuple(bondpair))
    def __init__(self, bondpair, length = None):
        self.length = length
    def __getattr__(self,name):
        if name == "rigid":
            return type(self.length) is float
        else:
            raise AttributeError(name)


class Topology(object):
    def __init__(self,topfile=None):
        #chemTypes: stores OBJECTS representing different species
        self.atom_types = collections.OrderedDict() 
        self.residue_types = collections.OrderedDict()
        self.molecule_types = collections.OrderedDict()
        
        #constituents: stores NAMES (strings) of the species included
        self.atoms = []
        self.residues = []
        self.molecules = []

        #other details: stores INDICES
        self.bond_list = []
        self.atoms_in_mol = []
        self.bonds_in_mol = []

        #actually load
        system_specs = None
        if topfile is not None:
            self.load(topfile)


    def __getattr__(self,name):
        if name == 'num_atom_types':
            return len(self.atom_types)
        if name == 'num_atoms':
            return len(self.atoms)
        if name == 'num_residues':
            return len(self.residues)
        if name == 'num_molecules':
            return len(self.molecules)
        if name == 'num_bonds':
            return len(self.bond_list)
        else:
            raise AttributeError(name)

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
        Parameters
        ----------
        topfile: str to topology file

        Notes
        -----
        {
         Atoms:[ [anameA,massA,chargeA], [anameB,massB,chargeB], ... ],
         Residues: [ [resname1,[atomname1_1, numatom1_1],[atomname1_2, numatom1_2],...], ...]
         Molecules: [ [molnameX, [resnameX_1, numresX_1],[resnameX_2, numresX_2],...], ...]
         System: [ [molnameI, nummolI], [molnameII, nummolII], ... ]

         HarmonicBonds
         GaussianBase
        }
       """
        with open(topfile,'r') as stream:
            self.myyaml = yaml.YAML()
            self.system_specs = yaml.load(stream, Loader=yaml.RoundTripLoader)
             
        print('===== Read in Data =====')
        for key,val in self.system_specs.items():
            print('{}: {}'.format(key,val))
        print("\n")

        self.createSys()

    def createSys(self,system_specs=None):
        print("===== Trying to create system from given topology dictionary =====")
        #self.top_dict = system_specs #may be a ruamel.yaml object with extra commenting bonuses
        if system_specs == None:
            system_specs = self.system_specs

        # --- Try create all the constituents, validating input as we go ---
        # ... Atom Types ...
        try:
            print("Attempting to read in atom types: {}".format(self.atom_types))
            for atom_type_entry in system_specs["Atoms"]:
                name, mass, charge = self.parseAtom(atom_type_entry)
                #self.atom_types.append( AtomType(name, mass, charge) )
                self.atom_types[name] = AtomType(name, mass, charge)
            print("Read in atom types: {}".format(self.atom_types))
        except:
            print("Unknown error while adding atoms")
        # ... Residue Types ...
        try:
            print("\nAttempting to read in residue types: {}".format(self.residue_types))
            for residue_type_entry in system_specs["Residues"]:
                name, atom_type_list = self.parseResidue(residue_type_entry)
                self.residue_types[name] = ResType(name, atom_type_list)
                #self.residue_types.append( ResType(name, atomlist) )
            print("Read in residue types: {}".format(self.residue_types))
        except:
            print("Unknown error while adding residues")
        
        # ... Molecule Types ...
        try:
            print("\nAttempting to read in molecule types: {}".format(self.molecule_types))
            for molecule_type_entry in system_specs["Molecules"]:
                name, residue_type_list, bonds = self.parseMolecule(molecule_type_entry)
                self.molecule_types[name] = MolType(name, residue_type_list)
                #self.molecule_types.append( MolType(name, residue_type_list) )
                self.molecule_types[name].Bond(self, bonds)
            print("Read in molecule types: {}".format(self.molecule_types))
        except:
            print("Unknown error while adding molecules")

        # ... System ...
        try:
            print("\nAttempting to add molecules to system")
            for moleculeType in system_specs["System"]:
                name,num_mol = self.parseSystem(moleculeType)
                for ii in range(num_mol):
                    self.addMolecule(name)
            print("Added molecules to system")
        except:
            print("Unknown error while constructing system")



    def parseAtom(self,atom_type_entry):
        print("...parsing {}".format(atom_type_entry))
        name, mass, charge = atom_type_entry[0], float(atom_type_entry[1]), float(atom_type_entry[2])
        return name, mass, charge

    def parseResidue(self,residue_type_entry):
        print("...parsing {}".format(residue_type_entry))
        name = residue_type_entry["name"]
        atoms = residue_type_entry["atoms"]

        atom_type_list = []
        for atom in atoms:
            print("......processing {}".format(atom))
            aname = atom[0]
            anum = int(atom[1])
            if anum < 0:
                raise ValueError("residue {}'s atom {} number must be positive integer, is right now {}".format(name,aname,anum))

            if aname in self.atom_types.keys():    
                atom_type_list.extend( anum*[self.atom_types[aname]] )
            else:
                raise ValueError("residue {} has an unrecognized atom {}".format(name, aname))

        return name, atom_type_list

    def parseMolecule(self,molecule_type_entry):
        print("...parsing {}".format(molecule_type_entry))
        name = molecule_type_entry["name"]
        residues = molecule_type_entry["residues"]
        bonding = molecule_type_entry["bonds"]

        residue_type_list = []
        for residue in residues:
            print("......processing {}".format(residue))
            resname = residue[0]
            resnum = int(residue[1])
            if resnum < 0:
                raise ValueError("molecule {}'s residue {} number must be positive integer, is right now {}".format(name,resname,resnum))

            if resname in self.residue_types.keys():
                residue_type_list.extend( resnum*[self.residue_types[resname]] )
            else:
                raise ValueError("molecule {} has an unrecognized residue {}".format(name,resname))
        
        return name, residue_type_list, bonding

    def parseSystem(self,system_molecule_entry):
        name = system_molecule_entry[0]
        num_mol = int(system_molecule_entry[1])
        return name, num_mol

    def addAtom(self,new_atom):
        """
        Parameters
        ----------
        new_atom : atom object
        """
        self.atoms.append( new_atom )
        new_atom.ind = len(self.atoms) - 1

    def addResidue(self, new_residue):
        """
        Parameters
        ----------
        new_residue : residue object
        """
        self.residues.append( new_residue )
        new_residue.ind = len(self.residues) - 1
        for atom in new_residue:
            self.addAtom( atom )

    def addMolecule(self,molname):
        """
        Parameters
        ----------
        molname : str
        """
        if molname in self.molecule_types.keys():
            if VVerbose: print('adding {}'.format(molname))
            old_num_atoms = len(self.atoms)

            this_mol_type = self.molecule_types[molname]
            new_mol = this_mol_type.New()
            self.molecules.append( new_mol )
            new_mol.ind = len(self.molecules) - 1

            for residue in new_mol:
                self.addResidue( residue )

            new_num_atoms = len(self.atoms)

            #self.atoms_in_mol.append( (len(self.atoms) - self.molecule_types[molname].num_atoms_in_mol, len(self.atoms)  )) 
            #self.atoms_in_mol.append( np.arange(old_num_atoms,new_num_atoms) )
            self.atoms_in_mol.append( list( range(old_num_atoms,new_num_atoms) ) )
            current_mol_ID = len(self.molecules) - 1
            self.addBonds(current_mol_ID)
        else:
            raise ValueError("Unrecognized molecule type {}, can't add".format(molname))

    def addBonds(self, current_mol_ID):
        "adds bonds for the latest molecule"
        "bond=None is default, simple bonding behavior"
        mol_type = self.molecules[current_mol_ID].mol_type
        #molname = self.molecules[current_mol_ID].name
        atoms_in_mol = self.atoms_in_mol[current_mol_ID]
        bonds = mol_type.bonds
        self.bonds_in_mol.append([])

        if VVerbose: print(atoms_in_mol)
        for bond in mol_type.bonds:
            #print("...relative bond: {}".format(bond))
            current_bond = (atoms_in_mol[bond[0]], atoms_in_mol[bond[1]])
            self.bond_list.append(current_bond)
            self.bonds_in_mol[-1].append(current_bond)
            if VVerbose: print("adding (absolute index) bond {}".format(current_bond))






