import torch
import numpy as np
from agent_network import AgentNetwork
from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer

class MarioAgent:
    def __init__(self, 
                 input_shape, 
                 num_actions, 
                 learning_rate=0.00025, 
                 gamma=0.9, 
                 exploration_rate=1.0, 
                 exploration_decay=0.99999975, 
                 exploration_min=0.1, 
                 replay_buffer_size=50000, 
                 batch_size=32, 
                 sync_steps=10000):

        # Hyperparams
        self.num_actions = num_actions
        self.steps = 0
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.batch_size = batch_size
        self.sync_network_rate = sync_steps

        self.online_network = AgentNetwork(input_shape, num_actions)
        # Freezing learning for target network, will sync with with online network every 10k steps
        self.target_network = AgentNetwork(input_shape, num_actions, freeze=True)
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        self.loss = torch.nn.MSELoss()
        storage = LazyMemmapStorage(replay_buffer_size)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

    def choose_action(self, observation):
        # Exploration
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.num_actions)
        
        # Unsqueezing adds dimension for batch size
        observation = torch.tensor(np.array(observation), dtype=torch.float32) \
                        .unsqueeze(0) \
                        .to(self.online_network.device)
        # Exploitation
        return self.online_network(observation).argmax().item()
    
    # decay exploration rate every step
    def decay_exploration_rate(self):
        self.exploration_rate = max(self.exploration_rate * self.exploration_decay, self.exploration_min)

    # add recent experience to reaply buffer
    def store_in_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(TensorDict({
                                            "state": torch.tensor(np.array(state), dtype=torch.float32), 
                                            "action": torch.tensor(action),
                                            "reward": torch.tensor(reward), 
                                            "next_state": torch.tensor(np.array(next_state), dtype=torch.float32), 
                                            "done": torch.tensor(done)
                                          }, batch_size=[]))
    # sync online and target networks
    def sync_networks(self):
        if self.steps % self.sync_network_rate == 0 and self.steps > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    # save model
    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)

    # load existing model
    def load_model(self, path):
        self.online_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))

    # calculate Q values
    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        self.sync_networks()
        
        self.optimizer.zero_grad()

        samples = self.replay_buffer.sample(self.batch_size).to(self.online_network.device)

        keys = ("state", "action", "reward", "next_state", "done")

        states, actions, rewards, next_states, dones = [samples[key] for key in keys]

        predicted_q_values = self.online_network(states)
        predicted_q_values = predicted_q_values[np.arange(self.batch_size), actions.squeeze()]

        target_q_values = self.target_network(next_states).max(dim=1)[0]
        # ignore future rewards if done
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        loss = self.loss(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        self.decay_exploration_rate()