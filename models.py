import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class JointMLP(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_joints = 7
        input_dim = 4 + 6 + 8
        trunk = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            trunk.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        self.trunk = nn.Sequential(*trunk)

        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(self.num_joints)
        ])

    def forward(self, data):
        x = data.x.float()
        global_feat = data.global_feat.float()
        ptr = data.ptr
        B = global_feat.size(0)
        device = x.device

        real_indices = []
        for i in range(B):
            start = ptr[i]
            real_indices.append(torch.arange(start, start+7, device=device))
        real_indices = torch.cat(real_indices)

        x_real = x[real_indices]

        target_expanded = global_feat.repeat_interleave(7, dim=0)

        pos = torch.arange(7, device=device).repeat(B)
        pos_onehot = F.one_hot(pos, num_classes=8).float() 

        # Формируем вход
        inp = torch.cat([x_real, target_expanded, pos_onehot], dim=-1)

        h = self.trunk(inp)

        out = torch.zeros(B*7, device=device)
        for j in range(7):
            mask = (pos == j)
            out[mask] = self.heads[j](h[mask]).squeeze(-1)

        return out
    

class TConvGNN(nn.Module):
    def __init__(self, hidden_dim=128, heads=4, num_layers=3, max_nodes=10, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.max_nodes = max_nodes

        self.node_encoder = nn.Linear(4, hidden_dim)
        self.edge_encoder = nn.Linear(5, hidden_dim)

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    hidden_dim, hidden_dim // heads, heads=heads,
                    edge_dim=hidden_dim, dropout=dropout, concat=True
                )
            )

        self.output_norm = nn.LayerNorm(hidden_dim)
        # Вход MLP: признаки узла + целевая поза
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float()
        ptr = data.ptr

        global_feat = data.global_feat.float()

        x_encoded = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)

        # 2. GNN
        for conv, norm in zip(self.convs, self.norms):
            res = x_encoded
            x_encoded = norm(x_encoded)
            x_encoded = conv(x_encoded, edge_index, e)
            x_encoded = F.gelu(x_encoded)
            x_encoded = x_encoded + res

        real_mask = torch.ones(x_encoded.size(0), dtype=torch.bool, device=x.device)
        virtual_indices = ptr[1:] - 1
        real_mask[virtual_indices] = False

        x_real = x_encoded[real_mask]

        num_nodes_per_graph = ptr[1:] - ptr[:-1]
        num_real_per_graph = num_nodes_per_graph - 1
        target_real = torch.repeat_interleave(global_feat, num_real_per_graph, dim=0)

        x_real = torch.cat([x_real, target_real], dim=-1)
        out = self.output_mlp(x_real)
        return out.squeeze(-1)
    

class IK_CVAE(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=256, num_layers=3, dropout=0.1, dec_heads=4, dec_lay=3):
        super().__init__()
        self.latent_dim = latent_dim
        enc_in = 7 + 7 + 6
        enc_layers = []
        for i in range(num_layers):
            in_dim = enc_in if i == 0 else hidden_dim
            enc_layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        self.encoder = nn.Sequential(*enc_layers)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = GraphDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=dec_lay,
            dropout=dropout,
            heads=dec_heads,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data, q_goal=None):
        T_target = data.global_feat.float()
        ptr = data.ptr
        B = data.num_graphs
        q_start_list = []
        for i in range(B):
            start = ptr[i]
            idx = torch.arange(start, start+7, device=T_target.device)
            q_start_list.append(data.x[idx, 0])
        q_start_batch = torch.stack(q_start_list)

        if q_goal is not None:
            x = torch.cat([q_start_batch, q_goal, T_target], dim=-1)
            h = self.encoder(x)
            mu, logvar = self.mu(h), self.logvar(h)
            z = self.reparameterize(mu, logvar)
            q_recon = self.decoder(z, T_target, data)
            return q_recon, mu, logvar
        else:
            num_samples = 1
            best_q = None
            min_dist = None
            for _ in range(num_samples):
                z = torch.randn(B, self.latent_dim).to(T_target.device)
                q_candidate = self.decoder(z, T_target, data)
                dist = torch.norm(q_candidate - q_start_batch, dim=1)
                if best_q is None:
                    best_q = q_candidate
                    min_dist = dist
                else:
                    mask = dist < min_dist
                    best_q[mask] = q_candidate[mask]
                    min_dist = torch.min(min_dist, dist)
            return best_q
        

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv  # вместо TransformerConv

class GraphDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers, heads, dropout):
        super().__init__()
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        self.node_encoder = nn.Linear(4, hidden_dim)
        self.edge_encoder = nn.Linear(5, hidden_dim)   # убираем

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(hidden_dim, hidden_dim // heads, heads=heads,
                                edge_dim=hidden_dim, dropout=dropout, concat=True)
                # GATv2Conv(
                #     in_channels=hidden_dim,
                #     out_channels=hidden_dim // heads,  # после конкатенации heads получим hidden_dim
                #     heads=heads,
                #     concat=True,
                #     dropout=dropout,
                #     add_self_loops=False  # мы сами строим рёбра, петли не нужны
                # )
            )

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 7, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, T_target, data):
        ptr = data.ptr
        B = data.num_graphs
        device = z.device
        num_real = 7

        real_indices = []
        for i in range(B):
            start = ptr[i]
            real_indices.append(torch.arange(start, start + num_real, device=device))
        real_indices = torch.cat(real_indices)
        x_real = data.x[real_indices]

        z_proj = self.z_proj(z)
        z_expanded = z_proj.repeat_interleave(num_real, dim=0)

        h = self.node_encoder(x_real)
        h = h + z_expanded

        # построение рёбер кинематической цепи (остаётся без изменений)
        row, col = [], []
        for b in range(B):
            base = b * num_real
            for j in range(num_real - 1):
                row.append(base + j)
                col.append(base + j + 1)
                row.append(base + j + 1)
                col.append(base + j)
        edge_index_real = torch.tensor([row, col], dtype=torch.long, device=device)

        # edge_attr больше не нужен
        num_chain_edges = (num_real - 1) * 2
        edge_attr_real = data.edge_attr.view(B, -1, 5)[:, :num_chain_edges, :].reshape(-1, 5)
        e = self.edge_encoder(edge_attr_real.float())

        for conv, norm in zip(self.convs, self.norms):
            res = h
            h = norm(h)
            h = conv(h, edge_index_real, e)   # старое
            # h = conv(h, edge_index_real)         # новое – только узлы и рёбра
            h = F.gelu(h)
            h = h + res

        # Финальная часть без изменений
        T_target_expanded = T_target.repeat_interleave(num_real, dim=0)
        q_start_per_joint = x_real[:, 0:1]
        h_with_all = torch.cat([h, T_target_expanded, q_start_per_joint], dim=-1)
        q_recon_flat = self.output_mlp(h_with_all).squeeze(-1)
        q_recon = q_recon_flat.view(B, num_real)
        return q_recon


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GATv2Conv

class IK_CVAE_GraphEnc_MLPDec(nn.Module):
    def __init__(
        self,
        latent_dim=32,
        hidden_dim=256,
        num_layers=3,
        dropout=0.1,
        heads=4,
        in_node_feats=4,
        in_edge_feats=5,
        use_transformer=True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_real = 7
        self.in_node_feats = in_node_feats
        self.in_edge_feats = in_edge_feats
        self.edges_per_graph = 2 * (self.num_real - 1)  # 12

        # Шаблон рёбер кинематической цепи (индексы 0..6)
        row, col = [], []
        for j in range(self.num_real - 1):
            row.extend([j, j + 1])
            col.extend([j + 1, j])
        self.register_buffer(
            'edge_index_template',
            torch.tensor([row, col], dtype=torch.long)
        )

        # ---------- Энкодер ----------
        # Вход: in_node_feats (x_real) + 6 (T_target) + 7 (q_goal)
        self.node_encoder = nn.Linear(in_node_feats + 6 + 7, hidden_dim)
        self.edge_encoder = nn.Linear(in_edge_feats, hidden_dim)

        self.encoder_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.encoder_convs = nn.ModuleList()
        ConvClass = TransformerConv if use_transformer else GATv2Conv
        for _ in range(num_layers):
            if use_transformer:
                self.encoder_convs.append(
                    TransformerConv(
                        hidden_dim,
                        hidden_dim // heads,
                        heads=heads,
                        edge_dim=hidden_dim,
                        dropout=dropout,
                        concat=True,
                    )
                )
            else:
                self.encoder_convs.append(
                    GATv2Conv(
                        hidden_dim,
                        hidden_dim // heads,
                        heads=heads,
                        concat=True,
                        dropout=dropout,
                        add_self_loops=False,
                    )
                )

        # Pooling по узлам (mean + max)
        self.pool_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # ---------- Декодер (MLP) ----------
        # Вход: latent_dim + 6 (T_target) + 7 (q_start)
        decoder_input_dim = latent_dim + 6 + 7
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_real),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _get_q_start(self, data):
        """Векторизованное извлечение q_start (первый признак 7 реальных узлов)."""
        if hasattr(data, 'q_start'):
            return data.q_start
        device = data.x.device
        ptr = data.ptr
        # Индексы реальных узлов: 0..6, 8..14, ... (пропускаем виртуальный на позиции 7)
        idx = ptr[:-1].unsqueeze(1) + torch.arange(self.num_real, device=device)
        q_start = data.x[idx, 0]   # (B, 7)
        return q_start

    def _build_batch_edge_index(self, data):
        """Строит edge_index для всей батчи, используя смещения, кратные 7."""
        if hasattr(data, 'edge_index_batch'):
            return data.edge_index_batch

        device = data.x.device
        ptr = data.ptr
        B = data.num_graphs
        # Смещения: 0, 7, 14, ... (без учёта виртуального узла)
        offsets = torch.arange(B, device=device) * self.num_real  # (B,)
        edge_index = self.edge_index_template.unsqueeze(0) + offsets.unsqueeze(1).unsqueeze(2)
        return edge_index.view(2, -1)

    def encode(self, data, q_goal, T_target):
        B = data.num_graphs
        device = data.x.device
        ptr = data.ptr

        # Извлекаем признаки реальных узлов (7 на граф)
        idx = ptr[:-1].unsqueeze(1) + torch.arange(self.num_real, device=device)
        x_real = data.x[idx].view(B * self.num_real, -1)  # (B*7, in_node_feats)

        # Расширяем глобальные признаки
        T_exp = T_target.unsqueeze(1).repeat(1, self.num_real, 1).view(-1, 6)
        q_goal_exp = q_goal.unsqueeze(1).repeat(1, self.num_real, 1).view(-1, 7)

        node_feats = torch.cat([x_real, T_exp, q_goal_exp], dim=-1)  # (B*7, in_node_feats+13)
        h = self.node_encoder(node_feats)  # (B*7, hidden_dim)

        # Рёбра (только цепные)
        edge_index = self._build_batch_edge_index(data)  # (2, B*12)

        # Признаки рёбер – берём первые B*12 из data.edge_attr
        total_chain_edges = B * self.edges_per_graph
        edge_attr_real = data.edge_attr[:total_chain_edges].float()  # (B*12, in_edge_feats)
        e = self.edge_encoder(edge_attr_real)  # (B*12, hidden_dim)

        # Графовые свёртки
        for conv, norm in zip(self.encoder_convs, self.encoder_norms):
            res = h
            h = norm(h)
            if isinstance(conv, TransformerConv):
                h = conv(h, edge_index, e)
            else:
                h = conv(h, edge_index)
            h = F.gelu(h)
            h = h + res

        # Pooling
        h = h.view(B, self.num_real, self.hidden_dim)
        h_pool = torch.cat([h.mean(dim=1), h.max(dim=1)[0]], dim=1)  # (B, hidden_dim*2)
        h_pool = self.pool_proj(h_pool)  # (B, hidden_dim)

        mu = self.fc_mu(h_pool)
        logvar = self.fc_logvar(h_pool)
        return mu, logvar

    def decode(self, z, T_target, q_start):
        """MLP-декодер."""
        decoder_input = torch.cat([z, T_target, q_start], dim=-1)
        return self.decoder(decoder_input)

    def forward(self, data, q_goal=None, num_samples=1):
        device = data.x.device
        T_target = data.global_feat.float()  # (B, 6)
        q_start = self._get_q_start(data)    # (B, 7)
        B = data.num_graphs

        if q_goal is not None:
            # Обучение
            mu, logvar = self.encode(data, q_goal, T_target)
            z = self.reparameterize(mu, logvar)
            q_recon = self.decode(z, T_target, q_start)
            return q_recon, mu, logvar
        else:
            # Инференс
            if num_samples == 1:
                z = torch.randn(B, self.latent_dim, device=device)
                return self.decode(z, T_target, q_start)
            else:
                # Если несколько семплов, выбираем ближайший к q_start
                best_q = None
                min_dist = None
                for _ in range(num_samples):
                    z = torch.randn(B, self.latent_dim, device=device)
                    q_candidate = self.decode(z, T_target, q_start)
                    dist = torch.norm(q_candidate - q_start, dim=1)
                    if best_q is None:
                        best_q = q_candidate
                        min_dist = dist
                    else:
                        mask = dist < min_dist
                        best_q[mask] = q_candidate[mask]
                        min_dist = torch.min(min_dist, dist)
                return best_q