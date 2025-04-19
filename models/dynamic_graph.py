# models/dynamic_graph.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicGraphEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_neighbors=5):
        super(DynamicGraphEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # 添加 hidden_dim 属性
        self.num_neighbors = num_neighbors

        # 示例：一个简单的图嵌入网络
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征，形状 (B, num_patches, input_dim)
        
        Returns:
            graph_features: 图嵌入后的特征，形状 (B, num_patches, hidden_dim)
        """
        B, N, D = x.size()
        x = x.view(B * N, D)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        graph_features = x.view(B, N, self.hidden_dim)
        return graph_features
