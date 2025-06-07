import torch
from torch.utils.data import Dataset
import numpy as np

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25


def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1200):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


class CustomDataSet(Dataset):
    def __init__(self, pairs, mol_embeds, prot_embeds, pair_embeds):
        self.pairs = pairs
        self.mol_embeds = mol_embeds
        self.prot_embeds = prot_embeds
        self.pair_embeds = pair_embeds

    def __getitem__(self, index):
        pair = self.pairs.loc[index]
        _, id, mol_id, pro_name, pro_id, compoundstr, proteinstr, label = pair
        mol_embed = self.mol_embeds[mol_id]
        prot_embed = self.prot_embeds[pro_name]
        pair_embed = self.pair_embeds[id]

        return id, compoundstr, proteinstr, mol_embed, prot_embed, pair_embed, label

    def __len__(self):
        return len(self.pairs)




def collate_fn(batch_data, max_d=100, max_p=2000, mol_len = 94, mol_dim = 384, pro_len = 2551, pro_dim = 320):
    N = len(batch_data)
    id_new = torch.zeros(N, dtype=torch.int)
    compound_new = torch.zeros((N, max_d), dtype=torch.long)
    protein_new = torch.zeros((N, max_p), dtype=torch.long)

    # ChemBERTa:davis 94 metz 114 
    mol_embed_new = torch.zeros((N, mol_len, mol_dim), dtype=torch.float)

    # esm2:davis 2551 metz 2529
    prot_embed_new = torch.zeros((N, pro_len, pro_dim), dtype=torch.float)
    pair_embed_new = torch.zeros((N, 768), dtype=torch.float)
    labels_new = torch.zeros(N, dtype=torch.float)

    for i, pair in enumerate(batch_data):
        id, compoundstr, proteinstr, mol_embed, prot_embed, pair_embed, label = pair
        id_new[i] = int(id)
        compoundint = torch.from_numpy(label_smiles(compoundstr, CHARISOSMISET, max_d))
        compound_new[i] = compoundint
        proteinint = torch.from_numpy(label_sequence(proteinstr, CHARPROTSET, max_p))
        protein_new[i] = proteinint
        mol_embed_new[i] = mol_embed
        prot_embed_new[i] = prot_embed
        pair_embed_new[i] = pair_embed
        labels_new[i] = np.float64(label)

    return (id_new, compound_new, protein_new, mol_embed_new, prot_embed_new, pair_embed_new, labels_new)


