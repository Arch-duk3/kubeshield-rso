import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import csv
import time
from collections import deque
from env import K8sEnv

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = K8sEnv()
    
    import requests
    max_retries = 30
    for i in range(max_retries):
        try:
            requests.get("http://localhost:8080/reset", timeout=2)
            print(f"Connected to Simulator for seed {seed}!")
            break
        except:
            print("Waiting for simulator...")
            time.sleep(1)
    else:
        print("Simulator not reachable. Ensure it's running on port 8080.")
        return

    log_file = open(f"experiment_seed_{seed}.csv", "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["time", "S", "R", "KRSI", "action", "n_rep", "n_nodes", "latency", "energy"])
    global_time = 0

    state_dim = env.state_dim
    action_dim = env.action_space
    
    model = DQN(state_dim, action_dim)
    target_model = DQN(state_dim, action_dim)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    replay_buffer = deque(maxlen=10000)
    
    epochs = 100
    steps_per_epoch = 100
    epsilon = 1.0
    epsilon_decay = 0.99
    epsilon_min = 0.1
    batch_size = 32
    gamma = 0.99
    
    R_critical = env.R_critical
    
    for epoch in range(epochs):
        state = env.reset()
        epoch_reward = 0
        krsi_sum = 0
        
        for step in range(steps_per_epoch):
            R = state[4]
            if R < R_critical:
                action = env.heuristic_action(state, R)
            else:
                if random.random() < epsilon:
                    action = random.randint(0, action_dim - 1)
                else:
                    with torch.no_grad():
                        state_t = torch.tensor(state, dtype=torch.float32)
                        q_values = model(state_t)
                        action = torch.argmax(q_values).item()
                        
            next_state, reward, done, info = env.step(action)
            
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            epoch_reward += reward
            krsi_sum += info['KRSI']
            global_time += 1
            
            csv_writer.writerow([
                global_time, info['S'], info['R'], info['KRSI'], 
                info['action'], info['n_rep'], info['n_nodes'], 
                info['latency'], info['energy']
            ])
            
            if len(replay_buffer) > batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states_t = torch.tensor(np.array(states), dtype=torch.float32)
                actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
                rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
                
                # Task 6: RL Stability Improvement (batch reward normalization)
                rewards_mean = rewards_t.mean()
                rewards_std = rewards_t.std()
                rewards_t = (rewards_t - rewards_mean) / (rewards_std + 1e-5)
                
                next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
                dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
                
                q_vals = model(states_t).gather(1, actions_t)
                with torch.no_grad():
                    # Task 3: Double DQN Target
                    a_star = model(next_states_t).argmax(1).unsqueeze(1)
                    next_q_vals = target_model(next_states_t).gather(1, a_star)
                    target = rewards_t + gamma * next_q_vals * (1 - dones_t)
                    
                loss = nn.MSELoss()(q_vals, target)
                optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            if done:
                break
                
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if epoch % 5 == 0:
            target_model.load_state_dict(model.state_dict())
            avg_krsi = krsi_sum / steps_per_epoch
            print(f"Seed {seed} | Epoch {epoch:3d} | Epsilon {epsilon:.2f} | Reward {epoch_reward:8.2f} | Avg KRSI {avg_krsi:.3f}")
            log_file.flush()

    log_file.close()

def train():
    seeds = [42, 43, 44, 45, 46]
    for seed in seeds:
        print(f"\n--- Starting experiment for seed {seed} ---")
        train_seed(seed)

if __name__ == "__main__":
    train()
