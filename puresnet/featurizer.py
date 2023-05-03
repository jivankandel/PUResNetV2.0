import numpy as np
from openbabel import pybel
class Featurizer():
    def __init__(self):

        # Remember namse of all features in the correct order
        self.FEATURE_NAMES = []

        
        self.ATOM_CODES = {}

        metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                    + list(range(37, 51)) + list(range(55, 84))
                    + list(range(87, 104)))

        # List of tuples (atomic_num, class_name) with atom types to encode.
        atom_classes = [
            (5, 'B'),
            (6, 'C'),
            (7, 'N'),
            (8, 'O'),
            (15, 'P'),
            (16, 'S'),
            (34, 'Se'),
            ([9, 17, 35, 53], 'halogen'),
            (metals, 'metal')
        ]

        for code, (atom, name) in enumerate(atom_classes):
            if type(atom) is list:
                for a in atom:
                    self.ATOM_CODES[a] = code
            else:
                self.ATOM_CODES[atom] = code
            self.FEATURE_NAMES.append(name)

        self.NUM_ATOM_CLASSES = len(atom_classes)

        
        # pybel.Atom properties to save
        self.NAMED_PROPS = ['hyb', 'heavydegree', 'heterodegree',
                            'partialcharge']
        self.FEATURE_NAMES += self.NAMED_PROPS


        # SMARTS definition for other properties
        self.SMARTS = [
            '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
            '[a]',
            '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
            '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
            '[r]'
        ]
        smarts_labels = ['hydrophobic', 'aromatic', 'acceptor', 'donor',
                            'ring']

        # Compile patterns
        self.compile_smarts()
        self.FEATURE_NAMES += smarts_labels

    def compile_smarts(self):
        self.__PATTERNS = []
        for smarts in self.SMARTS:
            self.__PATTERNS.append(pybel.Smarts(smarts))

    def encode_num(self, atomic_num):
        """Encode atom type with a binary vector. If atom type is not included in
        the `atom_classes`, its encoding is an all-zeros vector.

        Parameters
        ----------
        atomic_num: int
            Atomic number

        Returns
        -------
        encoding: np.ndarray
            Binary vector encoding atom type (one-hot or null).
        """

        if not isinstance(atomic_num, int):
            raise TypeError('Atomic number must be int, %s was given'
                            % type(atomic_num))

        encoding = np.zeros(self.NUM_ATOM_CLASSES)
        try:
            encoding[self.ATOM_CODES[atomic_num]] = 1.0
        except:
            pass
        return encoding
    def scale(self,array):
        array=array-np.mean(array,axis=0)
        scale=(1/np.max(np.abs(array)))*0.9999
        return array*scale

    def find_smarts(self, molecule):
        """Find atoms that match SMARTS patterns.

        Parameters
        ----------
        molecule: pybel.Molecule

        Returns
        -------
        features: np.ndarray
            NxM binary array, where N is the number of atoms in the `molecule`
            and M is the number of patterns. `features[i, j]` == 1.0 if i'th
            atom has j'th property
        """

        if not isinstance(molecule, pybel.Molecule):
            raise TypeError('molecule must be pybel.Molecule object, %s was given'
                            % type(molecule))

        features = np.zeros((len(molecule.atoms), len(self.__PATTERNS)))

        for (pattern_id, pattern) in enumerate(self.__PATTERNS):
            atoms_with_prop = np.array(list(*zip(*pattern.findall(molecule))),
                                       dtype=int) - 1
            features[atoms_with_prop, pattern_id] = 1.0
        return features

    def get_features(self, molecule):
        """Get coordinates and features for all heavy atoms in the molecule.

        Parameters
        ----------
        molecule: pybel.Molecule
        molcode: float, optional
            Molecule type. You can use it to encode whether an atom belongs to
            the ligand (1.0) or to the protein (-1.0) etc.

        Returns
        -------
        coords: np.ndarray, shape = (N, 3)
            Coordinates of all heavy atoms in the `molecule`.
        features: np.ndarray, shape = (N, F)
            Features of all heavy atoms in the `molecule`: atom type
            (one-hot encoding), pybel.Atom attributes, type of a molecule
            (e.g protein/ligand distinction), and other properties defined with
            SMARTS patterns
        """

        if not isinstance(molecule, pybel.Molecule):
            raise TypeError('molecule must be pybel.Molecule object,'
                            ' %s was given' % type(molecule))

        coords = []
        features = []
        heavy_atoms = []

        for i, atom in enumerate(molecule):
            # ignore hydrogens and dummy atoms (they have atomicnum set to 0)
            if atom.atomicnum > 1:
                heavy_atoms.append(i)
                coords.append(atom.coords)

                features.append(np.concatenate((
                    self.encode_num(atom.atomicnum),
                    [atom.__getattribute__(prop) for prop in self.NAMED_PROPS],
                )))

        coords = np.array(coords, dtype=np.float32)
        features = np.array(features, dtype=np.float32)
        features = np.hstack([features,
                              self.find_smarts(molecule)[heavy_atoms]])
        if np.isnan(features).any():
            raise RuntimeError('Got NaN when calculating features')
        return features
