import torch
import torch.nn as nn
import torch.nn.functional as F


class attention(nn.Module):
    def __init__(self, head=8, conv=128):
        super(attention, self).__init__()
        self.conv = conv
        self.head = head
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.d_a = nn.Linear(self.conv * 3, self.conv * 3 * head)
        self.p_a = nn.Linear(self.conv * 3, self.conv * 3 * head)
        self.scale = torch.sqrt(torch.FloatTensor([self.conv * 3])).cuda()

    def forward(self, drug, protein):
        bsz, d_ef, d_il = drug.shape
        bsz, p_ef, p_il = protein.shape
        
        drug_att = self.relu(self.d_a(drug.permute(0, 2, 1))).view(bsz, self.head, d_il, d_ef)
        protein_att = self.relu(self.p_a(protein.permute(0, 2, 1))).view(bsz, self.head, p_il, p_ef)
        
        interaction_map = torch.mean(
            self.tanh(torch.einsum('bhid,bhjd->bhij', drug_att, protein_att) / self.scale), dim=1
        )
        
        Compound_atte = self.tanh(torch.einsum('bij->bi', interaction_map)).unsqueeze(1)
        Protein_atte = self.tanh(torch.einsum('bij->bj', interaction_map)).unsqueeze(1)
        
        drug = drug * Compound_atte
        protein = protein * Protein_atte
        return drug, protein


class LMDTA(nn.Module):
    def __init__(self, 
                 conv=32, char_dim=128, head_num=8, dropout_rate=0.1,
                 mol_len=0, mol_dim=0, pro_len=0, pro_dim=0):
        super(LMDTA, self).__init__()
        
        self.dim = char_dim
        self.conv = conv
        self.head_num = head_num
        self.dropout_rate = dropout_rate

        # Activation & Dropout
        self.leaky_relu = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Fully connected layers
        self.fc1 = nn.Linear(384, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        nn.init.constant_(self.out.bias, 5)

        # Linear projection to unify embedding dims
        self.mol_fc = nn.Linear(mol_dim, 384)
        self.pro_fc = nn.Linear(pro_dim, 384)

        # Self-attention layers
        self.mol_self_attention = nn.MultiheadAttention(embed_dim=384, num_heads=8, batch_first=True)
        self.pro_self_attention = nn.MultiheadAttention(embed_dim=384, num_heads=8, batch_first=True)

        # Cross-attention modules
        self.attention1 = attention(head=head_num, conv=128)
        self.attention2 = attention(head=head_num, conv=64)

        # Pooling layers
        self.mol_max_pool = nn.MaxPool1d(mol_len)
        self.pro_max_pool = nn.MaxPool1d(pro_len)
        self.pair_max_pool = nn.MaxPool1d(4)
        self.pairtensor_max_pool = nn.MaxPool1d(4)

    def forward(self, id, drug, protein, mol_emb, pro_emb, pair_emb):
        # Step 1: Project raw embeddings to unified dimension
        mol_emb = self.mol_fc(mol_emb)  # [B, mol_len, 384]
        pro_emb = self.pro_fc(pro_emb)  # [B, pro_len, 384]

        # Step 2: Apply self-attention
        mol_emb = self.mol_self_attention(mol_emb, mol_emb, mol_emb)[0]
        pro_emb = self.pro_self_attention(pro_emb, pro_emb, pro_emb)[0]

        # Step 3: Cross attention
        mol_emb, pro_emb = self.attention1(
            mol_emb.permute(0, 2, 1), 
            pro_emb.permute(0, 2, 1)
        )

        # Step 4: Pooling
        mol_emb = self.mol_max_pool(mol_emb).squeeze(-1)
        pro_emb = self.pro_max_pool(pro_emb).squeeze(-1)
        pair = torch.cat([mol_emb, pro_emb], dim=1)  # [B, 384]

        # Step 5: Prepare pair and pair_emb for attention
        bsz = pair.size(0)
        pair = pair.view(bsz, 192, 4)
        pair_emb = pair_emb.view(bsz, 192, 4)

        # Step 6: Cross attention
        pair, pair_emb = self.attention2(pair, pair_emb)
        pair = self.pair_max_pool(pair).squeeze(2)
        pair_emb = self.pairtensor_max_pool(pair_emb).squeeze(2)

        # Step 7: Concatenate and MLP
        x = torch.cat([pair, pair_emb], dim=1)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.leaky_relu(self.fc3(x))
        out = self.out(x)

        return out