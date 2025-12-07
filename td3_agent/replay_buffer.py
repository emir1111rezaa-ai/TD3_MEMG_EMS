import numpy as np
import torch


class ReplayBuffer:
    """
    Experience Replay Buffer for TD3 Agent
    Stores transitions: (state, action, reward, next_state, done)
    Supports prioritized and uniform sampling
    """
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cpu'):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device
        
        # Pre-allocate memory
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add transition to buffer
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        """
        Uniform random sampling from buffer
        Returns: (states, actions, rewards, next_states, dones) as torch tensors
        """
        if batch_size > self.size:
            batch_size = self.size
        
        ind = np.random.randint(0, self.size, size=batch_size)
        
        states = torch.FloatTensor(self.states[ind]).to(self.device)
        actions = torch.FloatTensor(self.actions[ind]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[ind]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[ind]).to(self.device)
        dones = torch.FloatTensor(self.dones[ind]).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return self.size
