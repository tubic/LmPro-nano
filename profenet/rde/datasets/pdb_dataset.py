import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionException

from rde.utils.protein.parsers import parse_biopython_structure


class ProteinDataset(Dataset):
    """Generic Protein Dataset for structure-based learning."""
    
    def __init__(self, data_dir, transform=None, load_geometric=False, load_topological=False):
        """
        Initialize protein dataset.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing protein data
        transform : callable, optional
            Optional transform to be applied on a sample
        load_geometric : bool
            Whether to load geometric features if available
        load_topological : bool
            Whether to load topological features if available
        """
        self.data_dir = data_dir
        self.transform = transform
        self.load_geometric = load_geometric
        self.load_topological = load_topological
        
        # Load all protein files
        self.protein_files = []
        if os.path.isdir(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith('.pdb') or filename.endswith('.cif') or filename.endswith('.pkl'):
                    self.protein_files.append(os.path.join(data_dir, filename))
        elif os.path.isfile(data_dir) and data_dir.endswith('.txt'):
            # If data_dir is a file list
            with open(data_dir, 'r') as f:
                for line in f:
                    filepath = line.strip()
                    if os.path.exists(filepath):
                        self.protein_files.append(filepath)
    
    def __len__(self):
        return len(self.protein_files)
    
    def __getitem__(self, idx):
        protein_file = self.protein_files[idx]
        
        if protein_file.endswith('.pkl'):
            # Load preprocessed data
            with open(protein_file, 'rb') as f:
                data = pickle.load(f)
        else:
            # Load and parse structure
            data = self._load_structure(protein_file)
            
        # Apply transform if provided
        if self.transform:
            data = self.transform(data)
            
        return data
    
    def _load_structure(self, filepath):
        """Load protein structure from file."""
        if filepath.endswith('.pdb'):
            parser = PDBParser(QUIET=True)
        elif filepath.endswith('.cif'):
            parser = MMCIFParser(QUIET=True)
        else:
            raise ValueError(f'Unsupported file format: {filepath}')
        
        try:
            structure = parser.get_structure('protein', filepath)
            data, _ = parse_biopython_structure(structure[0])
            
            # Add identifier
            data['protein_id'] = os.path.basename(filepath).split('.')[0]
            data['filepath'] = filepath
            
            # Add dummy target values for compatibility with training scripts
            # In a real implementation, these would come from external data
            data['targets'] = torch.zeros((len(data['chain_id']), 2), dtype=torch.float32)
            
            return data
        except Exception as e:
            print(f"Error loading structure {filepath}: {e}")
            # Return minimal valid data structure
            return {
                'protein_id': os.path.basename(filepath).split('.')[0],
                'filepath': filepath,
                'coordinates': torch.zeros((10, 3), dtype=torch.float32),
                'features': torch.zeros((10, 128), dtype=torch.float32),
                'targets': torch.zeros((10, 2), dtype=torch.float32),
                'chain_id': ['A'] * 10,
                'residue_number': list(range(1, 11)),
                'aa': torch.zeros(10, dtype=torch.long)
            }


# Adapter class to make existing SKEMPI dataset compatible with enhanced RDE framework
class SKEMPIDatasetAdapter(Dataset):
    """Adapter for SKEMPI dataset to be compatible with enhanced RDE framework."""
    
    def __init__(self, skempi_dataset, load_geometric=False, load_topological=False):
        """
        Initialize SKEMPI dataset adapter.
        
        Parameters:
        -----------
        skempi_dataset : SkempiDataset
            Original SKEMPI dataset
        load_geometric : bool
            Whether to load geometric features
        load_topological : bool
            Whether to load topological features
        """
        self.skempi_dataset = skempi_dataset
        self.load_geometric = load_geometric
        self.load_topological = load_topological
    
    def __len__(self):
        return len(self.skempi_dataset)
    
    def __getitem__(self, idx):
        data = self.skempi_dataset[idx]
        
        # Adapt data format if needed
        if 'targets' not in data:
            data['targets'] = torch.tensor([[data['ddG'], 0.0]], dtype=torch.float32)
        
        # Ensure coordinates and features are in expected format
        if 'coordinates' not in data and 'pos' in data:
            data['coordinates'] = data['pos']
        
        return data


# Adapter class for PDBRedoChain dataset
class PDBRedoChainAdapter(Dataset):
    """Adapter for PDBRedoChain dataset to be compatible with enhanced RDE framework."""
    
    def __init__(self, pdbredo_dataset, load_geometric=False, load_topological=False):
        """
        Initialize PDBRedoChain dataset adapter.
        
        Parameters:
        -----------
        pdbredo_dataset : PDBRedoChainDataset
            Original PDBRedoChain dataset
        load_geometric : bool
            Whether to load geometric features
        load_topological : bool
            Whether to load topological features
        """
        self.pdbredo_dataset = pdbredo_dataset
        self.load_geometric = load_geometric
        self.load_topological = load_topological
    
    def __len__(self):
        return len(self.pdbredo_dataset)
    
    def __getitem__(self, idx):
        data = self.pdbredo_dataset[idx]
        
        # Adapt data format if needed
        if 'targets' not in data:
            # Add dummy targets for training compatibility
            data['targets'] = torch.zeros((len(data['chain_id']), 2), dtype=torch.float32)
        
        return data