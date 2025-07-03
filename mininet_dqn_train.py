from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Controller, OVSKernelSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.cli import CLI

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import re
from collections import deque

# --- Custom Topology: 6 UEs and 3 Edge Servers ---
class CustomTopo(Topo):
    def build(self):
        switch = self.addSwitch('s1')

        # Add 6 UEs
        for i in range(6):
            host = self.addHost(f'ue{i+1}')
            self.addLink(host, switch, cls=TCLink, bw=10)

        # Add 3 Edge Servers
        for i in range(3):
            edge = self.addHost(f'edge{i+1}')
            self.addLink(edge, switch, cls=TCLink, bw=100)


# --- DQN Model ---
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# --- Experience Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states)

    def __len__(self):
        return len(self.buffer)


# --- Parse RTT from Mininet ping output ---
def get_rtt(net, src_name, dst_name):
    src = net.get(src_name)
    dst = net.get(dst_name)
    # Run ping command with 1 packet, suppress output
    ping_result = src.cmd(f'ping -c 1 {dst.IP()}')
    # Extract time=xxx ms using regex
    match = re.search(r'time=(\d+\.\d+) ms', ping_result)
    if match:
        rtt = float(match.group(1))
    else:
        # If ping fails or no RTT found, assign high RTT penalty
        rtt = 100.0
    return rtt


# --- Generate synthetic task requirements dynamically ---
def generate_task(ue_name):
    # Randomly generate task size in MB and CPU demand in percentage
    task_size = random.uniform(1, 10)  # 1 to 10 MB
    cpu_demand = random.uniform(5, 50)  # 5% to 50%
    return task_size, cpu_demand


# --- Get state: RTTs + task size + cpu demand ---
def get_state(net, ue_name, task_idx):
    rtts = []
    for j in range(3):
        edge_name = f'edge{j+1}'
        rtt = get_rtt(net, ue_name, edge_name)
        rtts.append(rtt)
    # Generate task dynamically per UE
    task_size, cpu_demand = generate_task(ue_name)
    # Normalize inputs for NN stability (example normalization)
    max_rtt = 100.0
    max_task = 10.0
    max_cpu = 50.0
    state = np.array(
        [rtt / max_rtt for rtt in rtts] +
        [task_size / max_task, cpu_demand / max_cpu],
        dtype=np.float32
    )
    return state, task_size, cpu_demand


# --- Choose Action ---
def choose_action(state, model, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 3)  # 0 = local, 1–3 = edge1-edge3
    with torch.no_grad():
        q_values = model(torch.tensor(state))
        return torch.argmax(q_values).item()


# --- Train the model with Experience Replay and Target Network ---
def train(model, target_model, optimizer, replay_buffer, batch_size, gamma, criterion):
    if len(replay_buffer) < batch_size:
        return  # Not enough samples yet

    states, actions, rewards, next_states = replay_buffer.sample(batch_size)

    states_tensor = torch.tensor(states, dtype=torch.float32)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.long)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

    model.train()
    q_values = model(states_tensor)
    state_action_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_model(next_states_tensor)
        max_next_q_values = next_q_values.max(1)[0]
        expected_state_action_values = rewards_tensor + gamma * max_next_q_values

    loss = criterion(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# --- Main Loop ---
def main():
    topo = CustomTopo()
    net = Mininet(topo=topo, controller=Controller, link=TCLink, switch=OVSKernelSwitch)
    net.start()

    input_size = 5  # 3 RTTs + task_size + cpu_demand
    output_size = 4  # local + 3 edge servers

    model = DQN(input_size, output_size)
    target_model = DQN(input_size, output_size)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    replay_buffer = ReplayBuffer()
    gamma = 0.9
    epsilon = 0.3
    batch_size = 32
    target_update_freq = 10  # episodes

    episodes = 50  # training episodes
    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")
        for i in range(6):  # loop through all UEs
            ue_name = f'ue{i + 1}'

            # Get current state and task
            state, task_size, cpu_demand = get_state(net, ue_name, episode)

            # Choose action
            action = choose_action(state, model, epsilon)

            # Calculate latency and reward
            if action == 0:
                # Local processing: CPU time + fixed overhead
                latency = 15.0 + cpu_demand * 0.5
                reward = -latency
                print(f"{ue_name} processed locally. Task: {task_size:.2f}MB, CPU: {cpu_demand:.2f}%, Latency: {latency:.2f} ms")
            else:
                # Offload to edge server
                edge_name = f'edge{action}'
                rtt = get_rtt(net, ue_name, edge_name)
                # Transmission time proportional to task size + edge processing time (assumed fixed)
                transmission_latency = rtt + task_size * 0.3
                processing_latency = cpu_demand * 0.2  # Edge CPU demand weight
                latency = transmission_latency + processing_latency
                reward = -latency
                print(f"{ue_name} offloaded to {edge_name}. Task: {task_size:.2f}MB, CPU: {cpu_demand:.2f}%, Latency: {latency:.2f} ms")

            # Get next state for training
            next_state, _, _ = get_state(net, ue_name, episode + 1)

            # Store in replay buffer
            replay_buffer.push(state, action, reward, next_state)

            # Train the model
            train(model, target_model, optimizer, replay_buffer, batch_size, gamma, criterion)

        # Update target network periodically
        if (episode + 1) % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())
            print("Target network updated.")

    print("Training complete. Starting testing...")

    # --- Testing without epsilon exploration ---
    test_episodes = 10
    epsilon_test = 0.0
    for episode in range(test_episodes):
        print(f"Test Episode {episode + 1}/{test_episodes}")
        for i in range(6):
            ue_name = f'ue{i + 1}'
            state, task_size, cpu_demand = get_state(net, ue_name, episode)
            action = choose_action(state, model, epsilon_test)

            if action == 0:
                latency = 15.0 + cpu_demand * 0.5
                print(f"{ue_name} processed locally. Task: {task_size:.2f}MB, CPU: {cpu_demand:.2f}%, Latency: {latency:.2f} ms")
            else:
                edge_name = f'edge{action}'
                rtt = get_rtt(net, ue_name, edge_name)
                transmission_latency = rtt + task_size * 0.3
                processing_latency = cpu_demand * 0.2
                latency = transmission_latency + processing_latency
                print(f"{ue_name} offloaded to {edge_name}. Task: {task_size:.2f}MB, CPU: {cpu_demand:.2f}%, Latency: {latency:.2f} ms")

    CLI(net)
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    main()

