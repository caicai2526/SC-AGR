import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicPolicyInstanceSelector(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DynamicPolicyInstanceSelector, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, features):
        """
        参数:
            features: 全局特征，形状为 (B, input_dim)
        返回:s
            action_probs: 每个patch被选择的概率，形状为 (B, input_dim)
        """
        x = F.relu(self.fc1(features))
        action_probs = F.softmax(self.fc2(x), dim=-1)
        return action_probs
