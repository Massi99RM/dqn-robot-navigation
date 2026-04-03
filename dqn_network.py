import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """
    Neural network for Deep Q-Learning.
    Input: robot state (15 values)
    Output: Q-values for the 4 possible actions
    """
    
    def __init__(self, state_size=15, action_size=4, hidden_size=128):
        """
        Initialize the neural network.
        
        Args:
            state_size: Dimension of the state vector
            action_size: Number of possible actions (4: up, down, left, right)
            hidden_size: Number of neurons in hidden layers
        """
        super(DQNNetwork, self).__init__()
        
        # Network architecture: 4 fully connected layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor (state)
            
        Returns:
            Q-values for each action
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x)
