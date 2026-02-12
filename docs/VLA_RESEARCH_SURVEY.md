# VLA Research Survey Guide

A comprehensive guide for researching Vision-Language-Action models for robotics.

---

## 1. Foundational Papers

### 1.1 Core VLA Papers

| Paper | Year | Lab | Key Contribution | Link |
|-------|------|-----|------------------|------|
| **RT-1: Robotics Transformer** | 2022 | Google | First large-scale VLA for real robots, Token Learner | [arXiv](https://arxiv.org/abs/2212.06817) |
| **RT-2: Vision-Language-Action Models** | 2023 | Google | Web-scale pretraining + robot fine-tuning, emergent reasoning | [arXiv](https://arxiv.org/abs/2307.15818) |
| **OpenVLA** | 2024 | Stanford/UC Berkeley | Open-source VLA, efficient fine-tuning | [arXiv](https://arxiv.org/abs/2406.09246) |
| **Octo** | 2024 | UC Berkeley | Transformer-based diffusion policy, strong generalization | [arXiv](https://arxiv.org/abs/2405.12213) |

### 1.2 Architecture Papers

| Paper | Year | Lab | Key Contribution | Link |
|-------|------|-----|------------------|------|
| **Flamingo** | 2022 | DeepMind | Gated cross-attention, Perceiver Resampler | [arXiv](https://arxiv.org/abs/2204.14198) |
| **PaLM-E** | 2023 | Google | Embodied multimodal, continuous inputs | [arXiv](https://arxiv.org/abs/2303.03378) |
| **RT-X** | 2024 | Google | Cross-embodiment training, data scaling laws | [arXiv](https://arxiv.org/abs/2310.08899) |
| **GROOT** | 2024 | NVIDIA | Vision-language foundation for robots | [arXiv](https://arxiv.org/abs/2404.16807) |

### 1.3 Action Prediction Papers

| Paper | Year | Lab | Key Contribution | Link |
|-------|------|-----|------------------|------|
| **ACT (Action Chunking)** | 2023 | Stanford | Action chunking with transformers, CVAE | [arXiv](https://arxiv.org/abs/2304.13705) |
| **Diffusion Policy** | 2023 | Columbia | Denoising diffusion for actions, multimodal handling | [arXiv](https://arxiv.org/abs/2303.04137) |
| **Transformer Policy** | 2024 | Various | Various transformer-based policy architectures | - |

---

## 2. Paper Reading Order

### Beginner Track (Start Here)

1. **RT-1** - Understand the basic VLA concept
2. **ACT** - Learn action chunking fundamentals
3. **OpenVLA** - Study the open-source implementation
4. **Diffusion Policy** - Understand alternative action representation

### Advanced Track

1. **RT-2** - Web-scale training approach
2. **Flamingo** - Vision-language fusion techniques
3. **Octo** - Diffusion + transformer combination
4. **RT-X** - Cross-embodiment learning

### Key Questions to Answer While Reading

- [ ] How is the vision encoder designed? (ViT, CLIP, SigLIP?)
- [ ] How are vision tokens fused with language tokens?
- [ ] How is the action head designed? (MLP, diffusion, CVAE?)
- [ ] What is the action representation? (Delta positions, absolute, velocity?)
- [ ] How is training data formatted? (Episode-based, frame-based?)
- [ ] What loss functions are used?
- [ ] How is evaluation performed? (Sim, real robot, both?)

---

## 3. Open-Source Projects to Study

### 3.1 Primary References

| Project | URL | Stars | Key Learnings |
|---------|-----|-------|---------------|
| **OpenVLA** | [github.com/openvla/openvla](https://github.com/openvla/openvla) | 2K+ | Full VLA stack, training scripts, HuggingFace integration |
| **Octo** | [github.com/octo-models/octo](https://github.com/octo-models/octo) | 1K+ | Diffusion policy, JAX implementation, dataset handling |
| **LeRobot** | [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot) | 5K+ | HF ecosystem, dataset formats, evaluation, pretrained models |

### 3.2 Secondary References

| Project | URL | Key Learnings |
|---------|-----|---------------|
| **ACT** | [github.com/tonyzhaozh/act](https://github.com/tonyzhaozh/act) | Action chunking, simulation setup, real robot deployment |
| **Diffusion Policy** | [github.com/real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | Diffusion for robot actions, U-Net architecture |
| **RT-1** | [github.com/google-research/robotics_transformer](https://github.com/google-research/robotics_transformer) | Original Token Learner, film-efficient attention |
| **GROOT** | [github.com/NVIDIA/IsaacLab](https://github.com/NVIDIA/IsaacLab) | NVIDIA's VLA with Isaac simulation |

### 3.3 Code Study Checklist

For each project, study:
- [ ] Model architecture (how vision + language + action connect)
- [ ] Data loading pipeline (dataset format, preprocessing)
- [ ] Training loop (losses, optimization, logging)
- [ ] Evaluation scripts (metrics, visualization)
- [ ] Configuration system (how to configure experiments)

---

## 4. Datasets

### 4.1 Primary Datasets

| Dataset | Size | Tasks | Embodiments | Format | Link |
|---------|------|-------|-------------|--------|------|
| **Bridge v2** | 60K demos | 13 tasks | Single arm | HDF5/RLDS | [HuggingFace](https://huggingface.co/datasets/rail-berkeley/bridge_v2) |
| **DROID** | 76K demos | 56 tasks | Single arm | HDF5/RLDS | [Website](https://droid-dataset.github.io/) |
| **RT-X (Open X-Embodiment)** | 1M+ demos | 100+ tasks | Multiple | RLDS | [HuggingFace](https://huggingface.co/datasets/openx-embodiment) |
| **LIBERO** | 130K demos | 40 tasks | Sim only | HDF5 | [Website](https://libero-project.github.io/) |

### 4.2 Dataset Format Comparison

| Format | Description | Used By |
|--------|-------------|---------|
| **RLDS** | RL Dataset Standard (TFRecord-based) | RT-X, DROID |
| **HDF5** | Hierarchical Data Format | Bridge, custom datasets |
| **LeRobot** | HuggingFace datasets format | LeRobot ecosystem |
| **Zarr** | Chunked array storage | Some research codebases |

### 4.3 Data Format Details

**Bridge v2 Episode Structure:**
```
episode_0/
├── observations/
│   ├── images/          # RGB camera images
│   │   └── 0.png
│   └── state/           # Robot state (joint pos, etc.)
│       └── 0.npy
├── actions/             # Actions (delta positions)
│   └── 0.npy
└── metadata.json        # Episode info
```

**DROID Format:**
```python
{
    "observation": {
        "exterior_image_1_left": np.ndarray,  # [H, W, 3]
        "wrist_image_left": np.ndarray,
        "joint_state": np.ndarray,  # [7]
        ...
    },
    "action": np.ndarray,  # [7] - xyz + rotation + gripper
    "language_instruction": str,
}
```

### 4.4 Dataset Selection Guide

| Your Situation | Recommended Dataset |
|----------------|---------------------|
| Initial development | Bridge v2 (smaller, clean) |
| Diverse tasks | DROID (56 tasks) |
| Scale experiments | RT-X (1M+ demos) |
| Simulation only | LIBERO |

---

## 5. Simulation Environments

### 5.1 Environment Comparison

| Environment | Physics | GPU Accel | Parallel Env | Setup Complexity | Best For |
|-------------|---------|-----------|--------------|------------------|----------|
| **MuJoCo** | Accurate | No | CPU only | Easy | Initial development |
| **Isaac Gym** | Fast | Yes | 1000s envs | Medium | Large-scale training |
| **Isaac Orbit** | Accurate | Yes | 100s envs | High | Realistic simulation |
| **RLBench** | Moderate | No | Limited | Medium | Diverse tasks |
| **Maniskill3** | Fast | Yes | 100s envs | Medium | GPU-parallel training |

### 5.2 Installation Guides

**MuJoCo (Recommended for Starting):**
```bash
pip install mujoco
pip install gymnasium[mujoco]

# Test
python -c "import mujoco; print(mujoco.__version__)"
```

**Isaac Gym (NVIDIA GPU Required):**
```bash
# Download from NVIDIA Developer
pip install isaacgym

# Test
python -c "from isaacgym import gymapi; print('Isaac Gym installed')"
```

**Isaac Orbit (Newer, More Realistic):**
```bash
# Requires Isaac Sim installation first
pip install isaacsim
pip install isaacsim-rl

# See: https://isaac-sim.github.io/IsaacLab/
```

### 5.3 Recommended Setup

**Phase 1:** MuJoCo
- Fast iteration for debugging
- CPU-based, easy to run
- Good for testing model pipeline

**Phase 2:** Isaac Orbit
- GPU-parallel data generation
- More realistic physics
- Good for scaling experiments

---

## 6. Key Technical Concepts

### 6.1 Vision Encoders

| Encoder | Params | Resolution | Notes |
|---------|--------|------------|-------|
| **CLIP-ViT-B/16** | 86M | 224 | Original, widely used |
| **SigLIP-SO400M** | 400M | 384 | Better for robotics, used in OpenVLA |
| **DINOv2-Small** | 22M | 224 | Very efficient, good features |
| **MAE-ViT-B** | 86M | 224 | Self-supervised, good for fine-tuning |

**Recommendation:** SigLIP-SO400M (used in OpenVLA, proven for robotics)

### 6.2 Vision-Language Fusion

| Method | Memory | Complexity | Used In |
|--------|--------|------------|---------|
| **Concatenation** | High | Low | Simple VLA |
| **Cross-Attention** | Medium | Medium | Flamingo, OpenVLA |
| **Perceiver Resampler** | Low | Medium | Flamingo (variable patches) |
| **Prefix Tuning** | Low | Low | Efficient fine-tuning |

### 6.3 Action Representations

| Type | Description | Used In |
|------|-------------|---------|
| **Delta Position** | xyz delta from current | RT-1, ACT |
| **Absolute Position** | Target xyz position | Some works |
| **Velocity** | End-effector velocity | Continuous control |
| **Joint Positions** | Direct joint angles | Low-level control |

**Common Action Dim:** 7 (3 position + 3 rotation + 1 gripper)

### 6.4 Action Chunking

Instead of predicting single action, predict sequence:

```
Single Action:    a_t
Action Chunking:  [a_t, a_t+1, a_t+2, ..., a_t+H]

Where H = chunk_size (e.g., 10)
```

**Benefits:**
- Temporal consistency
- Better handling of multimodal actions
- Smoother trajectories

---

## 7. Training Considerations

### 7.1 Loss Functions

| Loss | Use Case | Formula |
|------|----------|---------|
| **MSE** | Deterministic actions | L2 loss |
| **L1** | Robust to outliers | L1 loss |
| **Gaussian NLL** | Uncertainty estimation | -log p(y|x) |
| **Diffusion Loss** | Diffusion policy | Denoising score matching |

### 7.2 Data Augmentation

**Image Augmentation:**
- Random crop
- Color jitter
- Random rotation (small)

**Action Augmentation:**
- Gaussian noise injection
- Temporal jittering

### 7.3 Training Recipes

**From Scratch:**
- Large data needed (1M+ demos)
- Long training time
- Not recommended for limited resources

**Fine-tuning (Recommended):**
- Start with pretrained VLM
- Only train projector + action head
- Works with 10K-100K demos

---

## 8. Evaluation Metrics

### 8.1 Simulation Metrics

| Metric | Description |
|--------|-------------|
| **Success Rate** | Task completion percentage |
| **Average Return** | Cumulative reward |
| **Trajectory Error** | L2 distance from expert |
| **Action Error** | MSE between predicted and expert actions |

### 8.2 Real Robot Metrics

| Metric | Description |
|--------|-------------|
| **Success Rate** | Real-world task completion |
| **Human Rating** | Quality score (1-5) |
| **Completion Time** | Time to complete task |
| **Safety Violations** | Number of unsafe actions |

---

## 9. Research Directions (2024-2025)

### Hot Topics

1. **World Models for Robotics** - Learn physics, predict futures
2. **Hierarchical VLA** - High-level planning + low-level control
3. **Multi-Embodiment** - Single model for different robots
4. **Sim-to-Real Transfer** - Domain randomization, adaptation
5. **Language-Conditioned** - Natural language instructions
6. **Few-Shot Learning** - Learn new tasks from few demos

### Open Problems

- Long-horizon task planning
- Real-time inference on edge devices
- Robustness to distribution shift
- Handling partial observability

---

## 10. Quick Start Checklist

### Week 1-2: Foundation
- [ ] Read RT-1 paper
- [ ] Read OpenVLA paper
- [ ] Clone and explore OpenVLA codebase
- [ ] Install MuJoCo, test basic environment
- [ ] Download Bridge v2 dataset sample

### Week 3-4: Understanding
- [ ] Read ACT paper
- [ ] Read Flamingo paper (fusion techniques)
- [ ] Run OpenVLA inference demo
- [ ] Study LeRobot dataset format
- [ ] Test data loading with Bridge v2

### Week 5-6: Deep Dive
- [ ] Read RT-2 paper
- [ ] Read Diffusion Policy paper
- [ ] Study Octo architecture
- [ ] Compare different VLA approaches
- [ ] Plan your architecture

---

## 11. Resources

### Courses
- [Stanford CS231n](http://cs231n.stanford.edu/) - Computer Vision
- [Stanford CS224n](http://web.stanford.edu/class/cs224n/) - NLP
- [Berkeley CS285](http://rail.eecs.berkeley.edu/deeprlcourse/) - Deep RL

### Blogs
- [The AI Epiphany](https://www.youtube.com/c/TheAIEpiphany) - Paper explanations
- [Lilian Weng's Blog](https://lilianweng.github.io/) - RL and transformers

### Communities
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [HuggingFace Discord](https://hf.co/join/discord)
- [Robotics Discord servers](https://discord.gg/robotics)

---

## 12. Citation List (BibTeX)

```bibtex
@article{rt1,
  title={RT-1: Robotics Transformer for Real-World Control at Scale},
  author={Brohan, Anthony and others},
  journal={arXiv preprint arXiv:2212.06817},
  year={2022}
}

@article{rt2,
  title={RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control},
  author={Brohan, Anthony and others},
  journal={arXiv preprint arXiv:2307.15818},
  year={2023}
}

@article{openvla,
  title={OpenVLA: An Open-Source Vision-Language-Action Model},
  author={Kim, Moo Jin and others},
  journal={arXiv preprint arXiv:2406.09246},
  year={2024}
}

@article{octo,
  title={Octo: An Open-Source Generalist Robot Policy},
  author={Team, OctoModel and others},
  journal={arXiv preprint arXiv:2405.12213},
  year={2024}
}

@article{flamingo,
  title={Flamingo: a Visual Language Model for Few-Shot Learning},
  author={Alayrac, Jean-Baptiste and others},
  journal={NeurIPS},
  year={2022}
}

@article{act,
  title={Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author={Zhao, Tony Z and others},
  journal={RSS},
  year={2023}
}

@article{diffusion_policy,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Chi, Cheng and others},
  journal={RSS},
  year={2023}
}
```
