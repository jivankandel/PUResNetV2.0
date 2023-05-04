import tqdm
from math import sin, cos, sqrt, pi
from itertools import combinations
from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np
import os
import requests
from zipfile import ZipFile
import pkg_resources
import MinkowskiEngine as ME

def download_data():
    buffer_size=64 * 1024
    url='https://nsclbio.jbnu.ac.kr/pv2_dataset/training_dataset.zip'
    response=requests.get(url,stream=True)
    file_size = int(response.headers.get("Content-Length", 0))
    if not 'training_dataset.zip' in os.listdir('.'): 
        with open('training_dataset.zip', "wb") as f:
            for data in tqdm.tqdm(response.iter_content(chunk_size=(1024*1024)), total=file_size // (1024*1024), unit="MB",desc='Downloading'):
                f.write(data)
    with ZipFile('training_dataset.zip', 'r') as zip_ref:
        for member in tqdm.tqdm(zip_ref.infolist(),desc='Extracting'):
           
            if member.is_dir():
                if not os.path.exists(member.filename):
                    os.makedirs(member.filename)
            else:
                with zip_ref.open(member, 'r') as source, open(member.filename, 'wb') as target:
                    while True:
                        chunk = source.read(buffer_size)
                        if not chunk:
                            break
                        target.write(chunk)
class SparseDataset(Dataset):
    def __init__(self,path,pdbs=None,set='Train'):
        self.set=set
        self.path=path
        if pdbs is None:
            self.paths=os.listdir(self.path)
        else:
            self.paths=pdbs
        # Create matrices for all possible 90* rotations of a box
        self.ROTATIONS = [self.rotation_matrix([1, 1, 1], 0)]

        # about X, Y and Z - 9 rotations
        for a1 in range(3):
            for t in range(1, 4):
                axis = np.zeros(3)
                axis[a1] = 1
                theta = t * pi / 2.0
                self.ROTATIONS.append(self.rotation_matrix(axis, theta))

        # about each face diagonal - 6 rotations
        for (a1, a2) in combinations(range(3), 2):
            axis = np.zeros(3)
            axis[[a1, a2]] = 1.0
            theta = pi
            self.ROTATIONS.append(self.rotation_matrix(axis, theta))
            axis[a2] = -1.0
            self.ROTATIONS.append(self.rotation_matrix(axis, theta))

        # about each space diagonal - 8 rotations
        for t in [1, 2]:
            theta = t * 2 * pi / 3
            axis = np.ones(3)
            self.ROTATIONS.append(self.rotation_matrix(axis, theta))
            for a1 in range(3):
                axis = np.ones(3)
                axis[a1] = -1
                self.ROTATIONS.append(self.rotation_matrix(axis, theta))
    
    def rotation_matrix(self,axis, theta):
        """Counterclockwise rotation about a given axis by theta radians"""

        if not isinstance(axis, (np.ndarray, list, tuple)):
            raise TypeError('axis must be an array of floats of shape (3,)')
        try:
            axis = np.asarray(axis, dtype=np.float)
        except ValueError:
            raise ValueError('axis must be an array of floats of shape (3,)')

        if axis.shape != (3,):
            raise ValueError('axis must be an array of floats of shape (3,)')

        if not isinstance(theta, (float, int)):
            raise TypeError('theta must be a float')

        axis = axis / sqrt(np.dot(axis, axis))
        a = cos(theta / 2.0)
        b, c, d = -axis * sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    def rotate(self,coords, rotation):
        """Rotate coordinates by a given rotation

        Parameters
        ----------
        coords: array-like, shape (N, 3)
            Arrays with coordinates and features for each atoms.
        rotation: int or array-like, shape (3, 3)
            Rotation to perform. You can either select predefined rotation by
            giving its index or specify rotation matrix.

        Returns
        -------
        coords: np.ndarray, shape = (N, 3)
            Rotated coordinates.
        """


        if not isinstance(coords, (np.ndarray, list, tuple)):
            raise TypeError('coords must be an array of floats of shape (N, 3)')
        try:
            coords = np.asarray(coords, dtype=np.float)
        except ValueError:
            raise ValueError('coords must be an array of floats of shape (N, 3)')
        shape = coords.shape
        if len(shape) != 2 or shape[1] != 3:
            raise ValueError('coords must be an array of floats of shape (N, 3)')

        if isinstance(rotation, int):
            if rotation >= 0 and rotation < len(self.ROTATIONS):
                return np.dot(coords, self.ROTATIONS[rotation])
            else:
                raise ValueError('Invalid rotation number %s!' % rotation)
        elif isinstance(rotation, np.ndarray) and rotation.shape == (3, 3):
            return np.dot(coords, rotation)

        else:
            raise ValueError('Invalid rotation %s!' % rotation)



    def __getitem__(self, idx):
        rot = int(np.random.choice(range(24)))
        x=self.paths[idx]
        coords_path=os.path.join(self.path,x,'coords.pt')
        feature_path=os.path.join(self.path,x,'feat.pt')
        label_path=os.path.join(self.path,x,'label.pt')
        coords, feats = torch.load(coords_path).numpy(),torch.load(feature_path).numpy()
        labels = torch.load(label_path).numpy()
        if self.set=='Train':
            coords=self.rotate(coords,rot)
            coords=coords.round().astype(int)
            _,indices=np.unique(coords,axis=0,return_index=True)
            feats=feats[indices,::]
            coords=coords[indices,::]
            labels=labels[indices]
        return  coords,feats,labels

    def __len__(self):
        return len(self.paths)
def custom_collation_fn(data_labels):
    coords, feats, labels = list(zip(*data_labels))
    bcoords = ME.utils.batched_coordinates(coords)
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels_batch = torch.from_numpy(np.concatenate(labels, 0)).int()

    return bcoords, feats_batch, labels_batch
def get_trainVal(path='sparse'):
    train_data = np.load(pkg_resources.resource_filename('puresnet', 'data/training_data.npy'))
    val_data= np.load(pkg_resources.resource_filename('puresnet', 'data/validation_data.npy'))
    train_dataset=SparseDataset(path=path,pdbs=train_data,set='Train')
    val_dataset=SparseDataset(path=path,pdbs=val_data,set='Val')
    return train_dataset,val_dataset

def get_trainVal_loder(path='sparse',batch_size=80,num_workers=4):
    train_dataset,val_dataset=get_trainVal(path=path)
    train_loder=DataLoader(
    train_dataset, batch_size=batch_size,shuffle=True,pin_memory=True,collate_fn=custom_collation_fn,num_workers=num_workers)
    val_loder=DataLoader(
    val_dataset, batch_size=batch_size,shuffle=True,pin_memory=True,collate_fn=custom_collation_fn,num_workers=num_workers)
    return train_loder,val_loder