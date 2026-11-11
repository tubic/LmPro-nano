from fairseq.data import Dictionary

# Decoder-side residue order. MUST match AbbertDictionary's residue order (indices 4..23)
# so that the AbBERT->GNN handoff (`antibody_seq - 3` in the datasets, the `x[:,4:24]` LM-head
# slice in ab_roberta_base.py, and soft_forward) maps every amino acid to the SAME physicochemical
# row. Keep ALPHABET_FOR_INIT, AbgenDictionary, and AbbertDictionary[4:24] in lock-step.
ALPHABET_FOR_INIT = '#GSVATLYRQDENKPFICWMH'

class AbgenDictionary(Dictionary):
    def __init__(self):
        self.symbols = [
            '#', 'G', 'S', 'V', 'A', 'T', 'L', 'Y', 'R', 'Q', 'D', 'E', 'N', 'K', 'P', 'F', 'I', 'C', 'W', 'M', 'H'
        ]
        self.count = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.indices = {
            '#': 0,  'G': 1, 'S': 2, 'V': 3, 'A': 4, 'T': 5, 'L': 6, 'Y': 7, 'R': 8, 'Q': 9, 'D': 10,
            'E': 11, 'N': 12, 'K': 13, 'P': 14, 'F': 15, 'I': 16, 'C': 17, 'W': 18, 'M': 19, 'H': 20
        }
        self.nspecial = len(self.symbols)


class AbbertDictionary(Dictionary):
    def __init__(self):
        bos="<s>"
        pad="<pad>"
        eos="</s>"
        unk="<unk>"
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = [
            '<s>', '<pad>', '</s>', '<unk>', 'G', 'S', 'V', 'A', 'T', 'L', 'Y', 'R', 'Q', 'D', 'E', 'N', 'K', 'P', 'F', 'I', 'C', 'W', 'M', 'H'
        ]
        self.count = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.indices = {
            '<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, 'G': 4, 'S': 5, 'V': 6, 'A': 7, 'T': 8, 'L': 9, 'Y': 10, 'R': 11, 'Q': 12, 
            'D': 13, 'E': 14, 'N': 15, 'K': 16, 'P': 17, 'F': 18, 'I': 19, 'C': 20, 'W': 21, 'M': 22, 'H': 23}
        self.bos_index = self.indices[bos]
        self.pad_index = self.indices[pad]
        self.eos_index = self.indices[eos]
        self.unk_index = self.indices[unk]

        self.nspecial = len(self.symbols) 

class TagDictionary(Dictionary):
    def __init__(self):
        bos="<s>"
        pad="<pad>"
        eos="</s>"
        unk="<unk>"
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = [
            '<s>', '<pad>', '</s>', '<unk>', '0', '3', '1', '2'
        ]
        self.count = [1, 1, 1, 1, 0, 0, 0, 0]
        self.indices = {
            '<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, '0': 4, '3': 5, '1': 6, '2': 7
        }
        self.bos_index = self.indices[bos]
        self.pad_index = self.indices[pad]
        self.eos_index = self.indices[eos]
        self.unk_index = self.indices[unk]

        self.nspecial = len(self.symbols) 


ALPHABET = AbgenDictionary()
ALPHABET_FULL = AbbertDictionary()
TAG_FULL = TagDictionary()

RESTYPE_1to3 = {
     "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN","E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

# ALPHABET = ['#', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
ATOM_TYPES = [
    '', 'N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]

# atom14 heavy-atom names per residue, keyed by 1-letter code (order-independent).
# RES_ATOM14 below is materialized in ALPHABET order, so it always tracks the vocab
# and can never silently drift from AbgenDictionary again.
_RES_ATOM14_BY_RES = {
    '#': ['', '', '', '', '', '', '', '', '', '', '', '', '', ''],
    'G': ['N', 'CA', 'C', 'O', '', '', '', '', '', '', '', '', '', ''],
    'A': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', ''],
    'S': ['N', 'CA', 'C', 'O', 'CB', 'OG', '', '', '', '', '', '', '', ''],
    'C': ['N', 'CA', 'C', 'O', 'CB', 'SG', '', '', '', '', '', '', '', ''],
    'T': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '', '', '', '', '', '', ''],
    'V': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '', '', '', '', '', '', ''],
    'P': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', '', '', '', '', '', '', ''],
    'D': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', '', '', '', '', '', ''],
    'N': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', '', '', '', '', '', ''],
    'L': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', '', '', '', '', '', ''],
    'I': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '', '', '', '', '', ''],
    'M': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', '', '', '', '', '', ''],
    'E': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', '', '', '', '', ''],
    'Q': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', '', '', '', '', ''],
    'K': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', '', '', '', '', ''],
    'H': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', '', '', '', ''],
    'F': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', '', '', ''],
    'R': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', '', '', ''],
    'Y': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', '', ''],
    'W': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
}
RES_ATOM14 = [_RES_ATOM14_BY_RES[s] for s in ALPHABET.symbols]

UNK_LIST = [
    '6u8k',
    '6db8',
    '5e08',
    '5gkr',
    '4xwo',
    '4kzd',
    '6x5n',
    '6db9',
    '1keg',
    '2ok0',
    '2fr4',
    '6b3k',
    '6b14',
    '2r8s',
    '1xf2',
    '1i8m',
    '6mwn',
    '6x5m',
    '1cbv',
    '4kze',
    '1ehl',
    '3ivk',
    '6u8d'
]
