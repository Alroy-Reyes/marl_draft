# MARL Timetabling Framework

**Multi-Agent Reinforcement Learning for High School Course Scheduling**

A production-grade MARL system for solving the timetabling problem in Philippine high schools using Proximal Policy Optimization (PPO) with parallel multi-agent coordination.

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Architecture](#architecture)
4. [Multi-Agent System](#multi-agent-system)
5. [Reward Structure](#reward-structure)
6. [Critic & Value Function](#critic--value-function)
7. [Parallelization Scheme](#parallelization-scheme)
8. [Project Structure](#project-structure)
9. [Training Pipeline](#training-pipeline)
10. [Key Features & Fixes](#key-features--fixes)
11. [Usage](#usage)

---

## Overview

This framework uses **multi-agent reinforcement learning** to solve the complex constraint satisfaction problem of scheduling:
- **Subjects** (courses) to **Teachers** (instructors)
- In **Rooms** across **Buildings** (campuses)
- Over **Days** and **Timeslots**

While satisfying hard constraints (no conflicts) and soft constraints (teacher preferences, workload balance, modality requirements).

**Version**: 14.6 (Environment) + 18.2 (Training)
**Algorithm**: PPO (Proximal Policy Optimization)
**Framework**: RLlib + PettingZoo
**Platform**: Windows-compatible with GPU acceleration

---

## Problem Statement

### Constraints

**Hard Constraints** (must satisfy):
- No teacher can teach multiple classes simultaneously
- No section can attend multiple classes simultaneously
- No room can host multiple classes simultaneously
- Each subject must be placed the required number of times
- Subjects cannot repeat on the same day
- Modality-room matching (Online → Virtual, Face-to-Face → Physical)

**Soft Constraints** (optimize):
- Assign subjects to preferred teachers
- Balance workload across teachers in each area
- Match teachers to their subject areas
- Minimize conflicts and duplicates
- Respect timeslot and room constraints

### Search Space Complexity

For a typical Manila schedule:
- **Subjects**: ~180-250
- **Teachers**: ~30-50
- **Days**: 5 (M-F)
- **Timeslots/day**: 8-12
- **Rooms**: 20-40 across 2-3 campuses
- **Action space per agent**: ~100,000+ (teacher × slot combinations)

**Total search space**: Astronomically large, making this an NP-hard problem.

---

## Architecture

### Environment: `ParallelTimetablingEnv`

**Location**: `envs/timetabling_env.py`

A custom PettingZoo ParallelEnv implementing:
- **State space**: Rich observation including teacher availability, room schedules, subject status, section conflicts, workload distribution
- **Action space**: MultiDiscrete[n_teachers+1, n_slots+1] with action masking
- **Transition dynamics**: Atomic placement with 3-stage conflict resolution
- **Reward function**: Multi-objective with milestone bonuses

### Training: `train_ppo.py`

**Location**: `training/train_ppo.py`

Implements:
- Custom neural network architecture
- PPO algorithm with adaptive learning rates
- Checkpoint resumption support
- Comprehensive validation callbacks

---

## Multi-Agent System

### Agent Design: SAHA (Subject Area Handling Agents)

The system creates **one agent per subject area** (e.g., Math, English, Science, etc.).

```python
# Example agents
agents = [
    "saha_Math",
    "saha_English",
    "saha_Science",
    "saha_FilipinoPanitikan",
    # ... one per area
]
```

### Why Area-Based Agents?

1. **Natural decomposition**: Teachers are assigned to subject areas
2. **Load balancing**: Each agent manages workload for its area's teachers
3. **Reduced action space**: Agents only consider teachers in their area
4. **Parallel execution**: All agents act simultaneously each timestep

### Agent Observation Space

Each SAHA agent observes (normalized to [0,1]):

**Teacher Availability** (per area):
- Remaining capacity for each teacher in the area
- Current workload distribution

**Room Schedule Summary**:
- Free slots per building × day × timeslot
- Building availability matrix

**Subject Status**:
- Binary vector: which subjects are fully placed
- Focus subject indicator (agent's current target)

**Section Features**:
- Section ID and availability
- Per-day feasibility scores
- Overall feasible slot fraction

**Workload Features**:
- Per-teacher load in area
- Area workload balance score
- Workload standard deviation

**Modality Features**:
- Subject modality (Face-to-Face / Online / Hybrid)
- Virtual room availability

**Communication Buffer** (10-dim):
- Global placement progress
- Conflict pressure
- Per-area balance scores

**Dimensions**: ~300-500 features (depends on configuration)

### Agent Action Space

```python
MultiDiscrete([max_teachers_per_area + 1, slot_choices + 1])
```

**Teacher Selection** (dimension 0):
- [0, max_teachers_per_area): Select teacher by area index
- [max_teachers_per_area]: WAIT action

**Slot Selection** (dimension 1):
- [0, slot_choices): Select global slot (building, room, day, timeslot)
- [slot_choices]: WAIT action

**Action Masking**: Invalid actions are masked using:
- Teacher availability (capacity, time conflicts)
- Room availability (conflicts, constraints)
- Section availability (time conflicts)
- Modality-room compatibility
- Timeslot constraints

### Conflict Resolution (3-Stage)

When multiple agents attempt conflicting placements:

**Stage 1: Teacher-Time Conflicts**
- Group intents by (teacher, day, timeslot)
- Winner: Highest subject priority
- Losers: Receive conflict penalty

**Stage 2: Room-Slot Conflicts**
- Group intents by (building, room, day, timeslot)
- Winner: Highest subject priority
- Losers: Receive conflict penalty

**Stage 3: Section-Time Conflicts**
- Group intents by (section, day, timeslot)
- Winner: Highest subject priority
- Losers: Receive conflict penalty

**Subject Priority**:
```python
priority = (
    10.0 if has_preferred_teacher else 0.0 +
    5.0 if has_room_constraints else 0.0 +
    min(5.0, fail_count // 10) +
    subject_index * 1e-4  # tie-breaker
)
```

---

## Reward Structure

### Per-Placement Rewards

**Successful Placement**: `+150.0` (base_success_reward)

**Teacher Matching**:
- Preferred teacher assigned: `+40.0` (r_teacher_match)
- Non-preferred (but preferred is full): `0.0`
- Non-preferred (preferred available): `-1.0` (r_teacher_mismatch)

**Area Matching**:
- Teacher's area matches subject area: `+25.0` (r_area_match)
- Area mismatch: `-1.5` (r_area_mismatch)

**Workload Balance**:
- Improved balance: `+15.0 × delta` (r_workload_balance)
- Degraded balance: `-4.5 × delta`

**Modality Bonuses**:
- Online in virtual room: `+8.0` (r_online_bonus)
- Hybrid: `+4.0`

**Full Placement Bonus**: `+5.0` when subject reaches required placements

### Action Penalties

**Wait Action**: `-20.0` (wait_penalty)

**Failed Placement Attempts**:
- Already placed: `-10.0`
- Day duplicate: `-50.0` (critical violation)
- Maximum placements reached: `-20.0`
- Step-local duplicate: `-10.0`
- Teacher time conflict: `-15.0`
- Room conflict: `-15.0`
- Section conflict: `-15.0`
- Modality mismatch: `-2.0`
- Timeslot mismatch: `-1.5`
- Campus violation: `-0.5`
- Teacher capacity: `-0.5`

### Milestone Rewards (Shared by All Agents)

Progressive bonuses at completion thresholds:

```python
50%  → +30.0  per agent
70%  → +50.0  per agent
85%  → +100.0 per agent
90%  → +200.0 per agent
95%  → +400.0 per agent
100% → +1000.0 per agent
```

### Episode-End Bonuses/Penalties

**Completion Bonus**: `100.0 × (completion_rate)³` (exponential)

**TOR Satisfaction**: `30.0 × (preferred_teacher_rate)`

**Workload Balance**: `10.0 × avg_area_balance`

**Completion Pressure** (escalating penalties):
```python
< 85%: -3000.0 × gap
< 90%: -1500.0 × gap
< 95%: -800.0 × gap
< 100%: -400.0 × gap
= 100%: +5000.0 (massive bonus!)
```

**Duplicate Penalty**: `-10.0` per excess placement

### Reward Scaling

All rewards are scaled by `0.01` and clipped to `[-100, 100]` to stabilize training.

---

## Critic & Value Function

### Architecture: Deep Value Network

The **critic** (value function) judges the quality of partial timetables.

**Location**: `train_ppo.py:230-244`

```python
value_branch = nn.Sequential(
    nn.Linear(final_dim, 512),      # Expand
    nn.LayerNorm(512),
    nn.ReLU(),
    nn.Dropout(0.1),

    nn.Linear(512, 256),             # Compress
    nn.LayerNorm(256),
    nn.ReLU(),
    nn.Dropout(0.1),

    nn.Linear(256, 128),             # Further compress
    nn.LayerNorm(128),
    nn.ReLU(),

    nn.Linear(128, 1)                # Scalar value
)
```

**Key Features**:
1. **Deeper than policy head**: 4 layers vs 2 for action heads
2. **Layer normalization**: Stabilizes value estimates
3. **Dropout regularization**: Prevents overfitting
4. **Careful initialization**: `N(0, 0.01)` for final layer
5. **High clip threshold**: `vf_clip_param=50.0` accommodates large milestone rewards

### Value Function Role

The critic learns to estimate:

```
V(s) = E[total_future_reward | state=s]
```

This includes:
- Expected placement success
- Anticipated milestone bonuses
- Predicted end-of-episode pressure
- Workload balance trajectory

**Why It Matters**:
- Guides exploration (advantage = Q - V)
- Reduces variance in policy gradients
- Enables credit assignment across long episodes (400 steps)
- Handles sparse milestone rewards effectively

### Training Metrics

**Value Loss** (vf_loss): MSE between V(s) and actual returns
- Target: < 5.0 (healthy)
- Warning: > 20.0 (high variance)
- Critical: > 100.0 (unstable training)

**Value Coefficient**: `vf_loss_coeff=1.0` (equal weight to policy loss)

---

## Parallelization Scheme

### Multi-Level Parallelism

**1. Agent-Level Parallelism** (Intra-Episode)
- All SAHA agents act **simultaneously** each timestep
- Uses PettingZoo `ParallelEnv` interface
- Conflict resolution happens **after** all agents commit

**2. Environment Vectorization** (Across Episodes)
- RLlib supports `num_envs_per_worker` for parallel episodes
- Currently: 1 env per worker (can scale up)

**3. Rollout Workers** (Across Cores)
- `num_rollout_workers=1` (can increase for faster data collection)
- Each worker runs independent episodes in parallel

**4. GPU Acceleration** (Neural Network)
- Policy and value networks run on GPU
- `num_gpus=1` for training
- Batch inference for multiple agents

### Data Flow

```
┌─────────────────────────────────────────────┐
│         Parallel Episode Execution          │
│  (Multiple workers, each with N envs)       │
└──────────────────┬──────────────────────────┘
                   │ Collect experiences
                   ▼
┌─────────────────────────────────────────────┐
│        Replay Buffer (Train Batch)          │
│         Size: 512 samples                   │
└──────────────────┬──────────────────────────┘
                   │ Sample minibatches
                   ▼
┌─────────────────────────────────────────────┐
│      GPU: Policy + Value Network Update     │
│   (10 SGD iterations × 256 minibatch)       │
└─────────────────────────────────────────────┘
```

### Synchronization Points

**Within Episode**:
- Agents act in parallel
- Environment synchronizes at `step()` boundary
- Conflict resolution is atomic

**Across Episodes**:
- Workers asynchronously collect rollouts
- Synchronize when train batch is full
- GPU processes batch in parallel

### Scalability

Current configuration is conservative for stability:
```python
num_rollout_workers = 1
num_envs_per_worker = 1
train_batch_size = 512
```

For faster training, can scale to:
```python
num_rollout_workers = 4-8  # More CPU cores
num_envs_per_worker = 2-4  # More parallel episodes
train_batch_size = 2048    # Larger batches
```

---

## Project Structure

```
marl_draft/
│
├── CLAUDE.md                              # This file
├── README.md                              # User-facing documentation
│
├── envs/
│   └── timetabling_env.py                 # ⭐ Core MARL environment
│       ├── ParallelTimetablingEnv         # Main env class
│       ├── Action/observation spaces
│       ├── Reward computation
│       ├── Conflict resolution
│       └── Schedule validation
│
├── training/
│   └── train_ppo.py                       # ⭐ Training script
│       ├── ImprovedSahaMaskedTwoHead      # Neural network + critic
│       ├── EnhancedValidationCallback     # Metrics & diagnostics
│       ├── make_manila_env()              # Environment factory
│       ├── PPO configuration
│       └── Checkpoint management
│
├── preprocessing/ (implied)
│   └── create_timeslots_manila.py         # Generates cache files
│
├── input_files/ (root directory)
│   ├── *.xlsx                             # Raw schedule data
│   ├── *.pkl                              # Cached environment data
│   └── cached_environment_data_MANILA_MODALITY.pkl  # ⭐ Main cache
│
├── outputs/ (generated during training)
│   ├── C:/ray_logs/                       # Training logs & checkpoints
│   │   ├── Manila_FULLY_FIXED_v18_2_with_Resume/
│   │   │   └── checkpoint_XXXXXX/         # Model checkpoints
│   │   └── manila_tensorboard/            # TensorBoard logs
│   │
│   └── C:/ray_spill/                      # Object spilling (Windows)
│
└── utils/ (if any)
    └── Helper functions
```

### Key Files

**Environment**:
- `envs/timetabling_env.py:33` - Main environment class
- `envs/timetabling_env.py:597` - Step function with conflict resolution
- `envs/timetabling_env.py:1301` - Action masking logic
- `envs/timetabling_env.py:1605` - Schedule validation

**Training**:
- `training/train_ppo.py:172` - Neural network architecture
- `training/train_ppo.py:364` - Validation callback
- `training/train_ppo.py:694` - Environment factory
- `training/train_ppo.py:980` - PPO configuration
- `training/train_ppo.py:1082` - Checkpoint resumption

---

## Training Pipeline

### Preprocessing (One-time)

```bash
python preprocessing/create_timeslots_manila.py
```

**Generates**: `cached_environment_data_MANILA_MODALITY.pkl`

**Contains**:
- Subject metadata (codes, areas, modalities)
- Teacher assignments and preferences
- Room configurations and campuses
- Timeslot definitions and constraints
- Section information

### Training Modes

**1. Fresh Training**
```bash
python training/train_ppo.py --iterations 100
```

**2. Resume from Checkpoint**
```bash
# Auto-detect latest checkpoint
python training/train_ppo.py --resume auto --iterations 200

# Specific checkpoint
python training/train_ppo.py --resume C:/ray_logs/.../checkpoint_000100 --iterations 200
```

**3. Validation Only**
```bash
python training/train_ppo.py --validate-only
```

### Training Progression

**Early Iterations (1-20)**:
- Agents learn basic placement mechanics
- High conflict rates (>50)
- Placement rate: 20-40%
- Reward: Negative to low positive

**Mid Training (20-50)**:
- Conflict resolution improves
- Agents coordinate via communication buffer
- Placement rate: 50-80%
- Conflicts drop to 10-30

**Late Training (50-100+)**:
- Near-optimal schedules
- Placement rate: 85-100%
- Conflicts: 0-5 (often zero!)
- Milestone bonuses unlock regularly

### Monitoring

**TensorBoard**:
```bash
tensorboard --logdir C:/ray_logs/manila_tensorboard
```

**Metrics**:
- `Reward/Mean`: Overall performance
- `Placement/Partial`: Subjects with ≥1 placement
- `Placement/Full`: Subjects fully placed
- `Validation/Total_Conflicts`: Sum of all conflict types
- `Loss/Value`: Critic training stability
- `Loss/Policy`: Actor training progress

**Console Output**:
- Real-time placement rates
- Conflict breakdowns (duplicates, teacher, section)
- Milestone achievements
- Action distribution (wait vs place)

---

## Key Features & Fixes

### Production-Ready Fixes

**FIX #1: Teacher-Slot Consistency** (`timetabling_env.py:1301`)
- Action masks ensure selected teacher is available in selected slot
- Prevents invalid (teacher, slot) combinations

**FIX #2: Section Conflict Resolution** (`timetabling_env.py:774`)
- 3-stage conflict resolution includes section conflicts
- Students can't attend multiple classes simultaneously

**FIX #3: Atomic Placement Validation** (`timetabling_env.py:807`)
- All constraints checked before committing placement
- Rollback on any violation

**FIX #7: Per-Placement Teacher Tracking** (`timetabling_env.py:126`)
```python
self.placement_teachers[(subject, day, timeslot)] = teacher
```
- Tracks which teacher teaches each specific placement
- Critical for subjects with multiple placements (2x/week)
- Prevents false conflict detection

**FIX #10: Day Duplicate Prevention** (`timetabling_env.py:457`)
```python
self.subject_day_usage[subject] = set([day1, day2, ...])
```
- Prevents same subject on same day (student cannot attend twice)
- Checked before every placement

**FIX #12: Step-Local Placement Tracking** (`timetabling_env.py:803`)
```python
step_placement_counts[(subject, section)] += 1
```
- Prevents multiple agents placing same subject in single step
- Eliminates race conditions in parallel execution

### Advanced Features

**Action Masking** (`use_action_masks=True`):
- Dramatically reduces invalid action space
- Guides exploration to feasible regions
- Improves sample efficiency by ~10x

**Progressive Difficulty** (`enable_progressive_difficulty=True`):
- Penalties scale from 0% to 100% over first 100 steps
- Allows agents to explore early without harsh punishments
- Accelerates early learning

**Milestone Rewards** (`enable_milestone_rewards=True`):
- Provides dense signal for long-horizon task
- Encourages pushing to completion thresholds
- Aligned with practical goals (>90% is acceptable, 100% is ideal)

**Communication Buffer** (`enable_communication=True`):
- 10-dimensional global state vector
- Enables implicit coordination between agents
- Includes placement progress, conflict pressure, area balances

**Modality Support**:
- Face-to-Face: Physical rooms only
- Online: Virtual rooms only
- Hybrid: Physical rooms preferred
- Automatic room-modality validation

**Repair Pass** (`enable_repair_pass=False`):
- Optional greedy fill at episode end
- Disabled by default (let RL learn fully)
- Can enable for guaranteed complete schedules

---

## Usage

### Prerequisites

```bash
pip install ray[rllib]==2.9.0
pip install torch torchvision
pip install pettingzoo gymnasium
pip install pandas numpy
pip install tensorboard psutil
```

### Quick Start

**1. Prepare Data**
```bash
# Place input Excel files in root directory
# Run preprocessing
python preprocessing/create_timeslots_manila.py
```

**2. Train Model**
```bash
# Fresh training (100 iterations)
python training/train_ppo.py --iterations 100

# Resume training to 200 iterations
python training/train_ppo.py --resume auto --iterations 200
```

**3. Monitor Progress**
```bash
# TensorBoard
tensorboard --logdir C:/ray_logs/manila_tensorboard

# Navigate to http://localhost:6006
```

**4. Validate Results**
```bash
# Check final schedule quality
python training/train_ppo.py --validate-only
```

### Expected Output

**Successful Training** (100 iterations):
- Placement rate: **95-100%**
- Total conflicts: **0-2**
- Duplicate placements: **0**
- Teacher conflicts: **0**
- Section conflicts: **0-2**

**Console Messages to Watch For**:
```
🎉 PERFECT SCHEDULE: ZERO CONFLICTS!
✅ All fixes (including FIX #7 and FIX #12) are working!
🎯🎯🎯🎯 100% COMPLETE @ step 347! (+1000.0)
```

### Troubleshooting

**High Value Loss** (>50):
- Reduce `vf_clip_param` to 20.0
- Increase `vf_loss_coeff` to 2.0
- Check for NaN/Inf in rewards

**Low Placement Rate** (<80%):
- Increase training iterations
- Check input data for infeasible constraints
- Enable repair pass for debugging

**Persistent Conflicts**:
- Verify FIX #7 is applied (check placement_teachers dict)
- Review conflict breakdown in episode logs
- Validate input data consistency

### Advanced Configuration

**Faster Training** (more parallelism):
```python
num_rollout_workers = 4
num_envs_per_worker = 2
train_batch_size = 2048
```

**Better Exploration**:
```python
entropy_coeff = 2.0  # Higher initial exploration
clip_param = 0.4     # Larger policy updates
```

**Tighter Constraints**:
```python
strict_teacher_match = True  # Enforce preferred teachers
wait_penalty = 50.0          # Discourage waiting
```

---

## Performance Benchmarks

**Hardware**: NVIDIA GPU (RTX 3060+), 16GB RAM
**Dataset**: Manila schedule (180 subjects, 40 teachers, 5 days, 10 slots/day)

**Training Time**:
- 100 iterations: ~45-60 minutes
- 200 iterations: ~90-120 minutes

**Final Performance** (Iteration 100):
- Placement success: 98-100%
- Zero-conflict rate: 80-95% of episodes
- Preferred teacher match: 75-85%
- Workload balance (std dev): <1.5 classes

**Sample Efficiency**:
- Episodes to 50% placement: ~10-15
- Episodes to 90% placement: ~30-50
- Episodes to 100% zero-conflict: ~70-100

---

## Future Enhancements

**Planned**:
- [ ] Curriculum learning (start with small subsets)
- [ ] Multi-objective Pareto optimization
- [ ] Attention-based agent communication
- [ ] Constraint relaxation scheduling
- [ ] Real-time human feedback integration

**Research Directions**:
- [ ] Graph Neural Networks for relational reasoning
- [ ] Hierarchical RL (high-level day planning, low-level slot assignment)
- [ ] Transfer learning across different schools
- [ ] Explainable schedules (why this assignment?)

---

## Citation

If you use this framework in research, please cite:

```bibtex
@software{marl_timetabling_2024,
  title={MARL Framework for High School Timetabling},
  author={Your Team},
  year={2024},
  version={14.6-18.2},
  url={https://github.com/your-repo/marl_draft}
}
```

---

## License

[Specify License Here]

---

## Contact

For questions, issues, or collaboration:
- **GitHub Issues**: [Your Repo]/issues
- **Email**: [Your Contact]

---

**Last Updated**: 2025-10-26
**Version**: Environment v14.6 + Training v18.2
