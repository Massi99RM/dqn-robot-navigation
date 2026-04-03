import random
import pickle
import os
from collections import deque, namedtuple

# Structure for storing a single experience
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Circular buffer for storing robot experiences.
    When the buffer is full, new experiences overwrite the oldest ones.
    """
    
    def __init__(self, capacity=10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.
        
        Args:
            state: Current state
            action: Action executed (0,1,2,3)
            reward: Reward obtained
            next_state: Next state
            done: Boolean indicating whether the episode ended
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of randomly sampled experiences
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """
        Return the number of experiences currently stored in the buffer.
        
        Returns:
            Integer number of experiences in the buffer
        """
        return len(self.buffer)
    
    def save_to_file(self, filename):
        """
        Save the buffer to a pickle file.
        
        Args:
            filename: Name of the file where to save the buffer
        """
        try:
            with open(filename, 'wb') as f:
                pickle.dump(list(self.buffer), f)
            print(f"Buffer saved to {filename} ({len(self.buffer)} experiences)")
        except Exception as e:
            print(f"Error saving buffer: {e}")
    
    def load_from_file(self, filename):
        """
        Load the buffer from a pickle file.
        
        Args:
            filename: Name of the file to load the buffer from
        """
        if not os.path.exists(filename):
            print(f"File {filename} not found. Buffer remains empty.")
            return
        
        try:
            with open(filename, 'rb') as f:
                loaded_experiences = pickle.load(f)
            
            # Clear current buffer and load new experiences
            self.buffer.clear()
            for experience in loaded_experiences:
                self.buffer.append(experience)
            
            print(f"Buffer loaded from {filename} ({len(self.buffer)} experiences)")
        except Exception as e:
            print(f"Error loading buffer: {e}")
    
    def is_ready(self, min_size=1000):
        """
        Check if the buffer contains enough experiences to start training.
        
        Args:
            min_size: Minimum number of required experiences
            
        Returns:
            True if the buffer is ready for training, False otherwise
        """
        return len(self.buffer) >= min_size
