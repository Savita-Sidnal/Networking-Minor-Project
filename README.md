# 📡 Intelligent Task Offloading in Edge Computing using Deep Q-Learning

This project implements an intelligent task offloading mechanism using Deep Q-Learning (DQN) in a simulated edge computing environment. The network is built using Mininet, and PyTorch is used to train a reinforcement learning agent that dynamically decides where to process user tasks — locally or on one of several edge servers.

---

## 🗺️ Network Topology

The custom topology consists of:

- 6 User Equipments (UEs): ue1, ue2, ue3, ue4, ue5, ue6
- 3 Edge Servers: edge1, edge2, edge3
- 1 Switch: s1
- 1 Controller: c0

💡 Bandwidth:
- UEs → Switch: 10 Mbps
- Edge Servers → Switch: 100 Mbps

📷 Topology Diagram:

![Topology](https://raw.githubusercontent.com/Savita-Sidnal/Networking-Minor-Project/main/images/topology.png)

---

## 🚀 Features

- ✅ Mininet Custom Topology
- ✅ Deep Q-Network (DQN) using PyTorch
- ✅ Real-time RTT measurement using ping
- ✅ Dynamic task generation per UE
- ✅ Reward computation based on latency
- ✅ Experience Replay Buffer
- ✅ Target Network Synchronization
- ✅ Command-line visualization of offloading decisions

---

## 🛠️ Prerequisites

- OS: Linux (tested on Ubuntu)
- Python 3.8+
- Mininet
- PyTorch
- NumPy

Install dependencies using:

```bash```
pip install torch numpy

## 🧠 DQN Overview

- Input (State):
  - RTT to edge1, edge2, edge3
  - Task Size (1MB–10MB)
  - CPU Demand (5%–50%)

- Output (Action Space):
  - 0: Process locally
  - 1: Offload to edge1
  - 2: Offload to edge2
  - 3: Offload to edge3

- Reward:
  - Negative of total latency (smaller latency → higher reward)

- Architecture:
  - Input layer: 5 neurons
  - Hidden layer: 128 ReLU
  - Output layer: 4 Q-values

- Training:
  - 50 episodes with 6 UEs per episode
  - Experience Replay Buffer
  - Target Network update every 10 episodes

---
## 📊 Result Analysis

- 📉 Latency reduces over episodes as the agent learns optimal offloading.
- 📈 Edge servers are preferred when RTT + CPU cost < local processing.
- 🧠 DQN learns adaptively using feedback from network measurements.
- ⚙️ CPU % represents the workload for each task and influences the total latency during both local and edge execution.

--- 
