import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import os
from experience import ReplayBuffer
from dqn_network import DQNNetwork


class DQNAgent:
    """
    Agent implementing the Deep Q-Network algorithm.
    Uses a ReplayBuffer to store experiences and a target network to stabilize training.
    """
    
    def __init__(self, state_size, action_size, learning_rate, 
                 gamma, epsilon_start, epsilon_end, epsilon_decay,
                 buffer_size, batch_size, target_update):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Dimension of the state vector
            action_size: Number of possible actions
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial epsilon value for exploration
            epsilon_end: Final epsilon value
            epsilon_decay: Epsilon decay factor
            buffer_size: Replay buffer size
            batch_size: Batch size for training
            target_update: Frequency of target network updates
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Training counters
        self.steps = 0
        self.training_step = 0
        
        # Device for GPU/CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks: main and target
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        
        # Initialize target network with main network weights
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer for storing experiences
        self.memory = ReplayBuffer(buffer_size)
        
        print(f"DQN Agent initialized on {self.device}")
        print(f"Network: {state_size} -> {action_size}")
    
    def predict(self, state, training=True):
        """
        Predict the action to execute given a state.
        Uses epsilon-greedy with improved exploration.
        
        Args:
            state: Current robot state
            training: If True, applies epsilon-greedy; otherwise always chooses best action
            
        Returns:
            Action to execute (0-3)
        """
        # Epsilon-greedy: exploration vs exploitation
        if training and random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        
        # Exploitation: best action according to the network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        return action
    
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store an experience in the replay buffer.
        
        Args:
            state: Current state
            action: Action executed
            reward: Reward obtained
            next_state: Next state
            done: Whether the episode ended
        """
        self.memory.add(state, action, reward, next_state, done)
    
    def train(self):
        """
        Train the neural network using a batch of experiences from the replay buffer.
        Implements the DQN algorithm with target network to stabilize training.
        
        Returns:
            float: Training loss (None if not enough memory)
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample a batch of experiences
        experiences = self.memory.sample(self.batch_size)
        
        # Separate batch components
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        # Current Q-values for current states
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Q-values for next states (from target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            # If episode ended, future value is 0
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss (Mean Squared Error)
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update epsilon (reduce exploration over time)
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        # Periodically update target network
        self.training_step += 1
        if self.training_step % self.target_update == 0:
            self.update_target_network()
        
        return loss.item()
    
    def update_target_network(self):
        """
        Copy weights from main network to target network.
        This stabilizes training by preventing the target from changing too rapidly.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath):
        """
        Save the model and agent parameters to file.
        
        Args:
            filepath: Path where to save the model
        """
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'training_step': self.training_step,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'target_update': self.target_update
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load the model and agent parameters from file.
        
        Args:
            filepath: Path of the file to load
        """
        if not os.path.exists(filepath):
            print(f"File {filepath} not found.")
            return False
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Load network weights
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load agent parameters
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']
            self.training_step = checkpoint['training_step']
            
            print(f"Model loaded from {filepath}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_stats(self):
        """
        Return agent statistics for monitoring.
        
        Returns:
            dict: Dictionary with statistics
        """
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'training_steps': self.training_step,
            'total_steps': self.steps
        }

    def reset_epsilon_for_new_phase(self, new_epsilon=0.3):
        """
        Reset epsilon for a new training phase.
        Useful when loading a model and wanting more exploration.
        """
        self.epsilon = new_epsilon
        print(f"Epsilon set to {self.epsilon} for new phase")
