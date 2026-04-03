"""
RetinaVQA Model - Causal Graph-Guided Vision-Language Model for OCT Screening
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EvidentialGATLayer(nn.Module):
    """Evidential Graph Attention Layer"""

    def __init__(self, in_dim, out_dim, num_classes=3, dropout=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(2 * out_dim, 1))
        nn.init.xavier_uniform_(self.a)
        self.evidence_head = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, num_classes)
        )
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, return_attention=False):
        num_nodes = x.size(0)
        Wh = self.W(x)
        Wh = self.dropout(Wh)
        edge_src, edge_tgt = edge_index[0], edge_index[1]
        Wh_src, Wh_tgt = Wh[edge_src], Wh[edge_tgt]
        concat = torch.cat([Wh_src, Wh_tgt], dim=1)
        alpha = self.leaky_relu(torch.matmul(concat, self.a).squeeze())
        alpha = F.softmax(alpha, dim=0)
        x_new = torch.zeros(num_nodes, self.out_dim).to(x.device)
        for i in range(len(edge_src)):
            x_new[edge_tgt[i]] += alpha[i] * Wh_src[i]
        evidence = F.softplus(self.evidence_head(x_new)) + 1.0
        if return_attention:
            return x_new, evidence, alpha
        return x_new, evidence


class HierarchicalEvidentialGAT(nn.Module):
    """Hierarchical Evidential GAT (HE-GAT)"""

    def __init__(self, in_dim=768, hidden_dim=256, out_dim=128, 
                 num_classes=3, num_heads=4, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            EvidentialGATLayer(in_dim, hidden_dim, num_classes, dropout) 
            for _ in range(num_heads)
        ])
        self.head_aggregation = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gat2 = EvidentialGATLayer(hidden_dim, out_dim, num_classes, dropout)
        self.norm2 = nn.LayerNorm(out_dim)
        self.residual = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, return_attention=False):
        head_outputs, head_evidences = [], []
        for head in self.heads:
            out, ev, att = head(x, edge_index, return_attention=True)
            head_outputs.append(out)
            head_evidences.append(ev)
        x_concat = torch.cat(head_outputs, dim=1)
        x = self.head_aggregation(x_concat)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)
        evidence1 = torch.stack(head_evidences, dim=0).mean(dim=0)
        x2, evidence2, att2 = self.gat2(x, edge_index, return_attention=True)
        x2 = self.norm2(x2)
        x2 = F.elu(x2)
        x_out = x2 + self.residual(x)
        final_evidence = (evidence1 + evidence2) / 2
        if return_attention:
            return x_out, final_evidence, (att, att2)
        return x_out, final_evidence


class RetinaVQA(nn.Module):
    """
    RetinaVQA: Causal Graph-Guided Vision-Language Model for OCT Screening

    Args:
        disease_node_indices: Indices of disease nodes in the graph (default: [8,9,10,11])
        num_nodes: Total number of nodes in the causal graph (default: 16)
    """

    def __init__(self, disease_node_indices=None, num_nodes=16):
        super().__init__()
        self.num_nodes = num_nodes
        self.disease_node_indices = disease_node_indices or [8, 9, 10, 11]

        # Vision encoder (ResNet18)
        self.vision_encoder = models.resnet18(pretrained=False)
        self.vision_encoder.fc = nn.Identity()

        # Projection layer
        self.vision_projection = nn.Linear(512, 768)

        # Hierarchical Evidential GAT
        self.he_gat = HierarchicalEvidentialGAT(
            in_dim=768, hidden_dim=256, out_dim=128,
            num_classes=3, num_heads=4, dropout=0.2
        )

        # Severity head (outputs 0-1 severity score)
        self.severity_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Uncertainty head (from evidence)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, images, edge_index, edge_weights=None):
        """
        Forward pass

        Args:
            images: Input images [batch, 3, 224, 224]
            edge_index: Graph edge indices [2, num_edges]
            edge_weights: Optional edge weights [num_edges]

        Returns:
            severity: Severity score [batch]
            uncertainty: Uncertainty estimate [batch]
            evidence: Evidence tensor [batch, num_nodes, 3]
        """
        batch_size = images.size(0)

        # Vision encoding
        visual_features = self.vision_encoder(images)
        visual_features = visual_features.view(batch_size, -1)
        visual_features = self.vision_projection(visual_features)

        # Expand to nodes
        node_features = visual_features.unsqueeze(1).expand(-1, self.num_nodes, -1)

        # Graph reasoning
        enhanced_nodes, all_evidence = [], []
        for b in range(batch_size):
            enhanced_b, evidence_b = self.he_gat(node_features[b], edge_index)
            enhanced_nodes.append(enhanced_b)
            all_evidence.append(evidence_b)

        enhanced = torch.stack(enhanced_nodes, dim=0)
        evidence = torch.stack(all_evidence, dim=0)

        # Pool disease node features
        disease_features = enhanced[:, self.disease_node_indices, :].mean(dim=1)

        # Severity and uncertainty
        severity = self.severity_head(disease_features).squeeze()
        uncertainty = self.uncertainty_head(
            evidence[:, self.disease_node_indices, :].mean(dim=1)
        ).squeeze()

        return severity, uncertainty, evidence


def load_retinavqa(model_path, graph_path, device='cuda'):
    """
    Load a trained RetinaVQA model

    Args:
        model_path: Path to model weights (.pt file)
        graph_path: Path to causal graph (.pt file)
        device: Device to load model on

    Returns:
        model: Loaded RetinaVQA model
        edge_index: Graph edge indices
        edge_weights: Graph edge weights
    """
    # Load graph
    graph_data = torch.load(graph_path, map_location=device)
    edge_index = graph_data['edge_index']
    edge_weights = graph_data.get('edge_weights', None)

    # Load model
    model = RetinaVQA().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    return model, edge_index, edge_weights
