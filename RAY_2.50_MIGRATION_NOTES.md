# Ray 2.50.1 Migration Notes for MARL Timetabling

## Summary

This document outlines the changes made to migrate the MARL timetabling training script from Ray 2.9.0 (originally designed for Windows) to Ray 2.50.1 on macOS.

## Installation Requirements

### Missing Dependencies

The following packages were missing and had to be installed:

```bash
pip install dm-tree      # Required by RLlib (was conflicting with wrong 'tree' package)
pip install scipy lz4    # Required by RLlib
pip install torch torchvision torchaudio  # PyTorch
pip install tensorboard  # TensorBoard logging
pip install pettingzoo supersuit openpyxl  # Environment dependencies
```

### Dependency Issue: `tree` vs `dm-tree`

**Problem:** Two different packages named "tree":
1. `tree` (0.2.4) - File tree visualization tool (WRONG)
2. `dm-tree` - DeepMind's tree library (CORRECT)

**Solution:**
```bash
pip uninstall -y tree
pip install --force-reinstall dm-tree
```

## Code Changes

### 1. Cross-Platform Path Configuration

**File:** `training/train_ppo.py:136-152`

**Before:**
```python
# Windows-friendly Ray setup
SPILL_DIR = "C:/ray_spill"
TEMP_DIR = "C:/ray_temp"
```

**After:**
```python
# Cross-platform Ray setup
import platform
if platform.system() == "Windows":
    SPILL_DIR = "C:/ray_spill"
    TEMP_DIR = "C:/ray_temp"
    RAY_LOGS_DIR = "C:/ray_logs"
else:
    # macOS/Linux: use home directory
    SPILL_DIR = os.path.expanduser("~/ray_spill")
    TEMP_DIR = os.path.expanduser("~/ray_temp")
    RAY_LOGS_DIR = os.path.expanduser("~/ray_logs")
```

**All hardcoded `C:/ray_logs` paths were replaced with `RAY_LOGS_DIR` variable.**

### 2. RLlib API Changes (Ray 2.9 → 2.50)

#### 2.1 `.rollouts()` → `.env_runners()`

**Before:**
```python
.rollouts(
    num_rollout_workers=2,
    num_envs_per_worker=1,
)
```

**After:**
```python
.env_runners(
    num_env_runners=2,
    num_envs_per_env_runner=1,
)
```

#### 2.2 `sgd_minibatch_size` → `minibatch_size`

**Before:**
```python
.training(
    sgd_minibatch_size=2048,
    num_sgd_iter=1,
)
```

**After:**
```python
.training(
    minibatch_size=2048,
    num_epochs=1,  # renamed from num_sgd_iter
)
```

#### 2.3 `local_dir` → `storage_path` in RunConfig

**Before:**
```python
run_cfg = RunConfig(
    local_dir=RAY_LOGS_DIR,
)
```

**After:**
```python
run_cfg = RunConfig(
    storage_path=RAY_LOGS_DIR,
)
```

#### 2.4 Learning Rate and Entropy Schedules Deprecated

**Before:**
```python
.training(
    lr_schedule=[
        [0, 1e-3],
        [10000, 5e-4],
        [50000, 2e-4],
        [100000, 1e-4],
    ],
    entropy_coeff_schedule=[
        [0, 1.0],
        [20000, 0.5],
        [50000, 0.2],
        [100000, 0.05],
    ],
)
```

**After:**
```python
.training(
    # Note: lr_schedule is deprecated in Ray 2.50+
    lr=5e-4,  # Using middle value from old schedule
    # Note: entropy_coeff_schedule is deprecated in Ray 2.50+
    entropy_coeff=0.5,  # Using middle value from old schedule
)
```

**TODO:** Implement schedules using callbacks if needed in the future.

#### 2.5 `.experimental()` Parameters Removed

**Before:**
```python
.experimental(_enable_new_api_stack=False, _disable_preprocessor_api=True)
```

**After:**
```python
# Removed - these parameters no longer exist
```

### 3. New API Stack Compatibility

**CRITICAL FIX:** Ray 2.50+ defaults to the "new API stack" which doesn't support the custom model architecture used in this project.

**File:** `training/train_ppo.py:1115-1118`

**Added:**
```python
ppo_cfg = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    # ... rest of config
)
```

**Reason:** The custom `ImprovedSahaMaskedTwoHead` model uses the old API stack. The new API stack requires a completely different model definition using RLModule and Catalog classes.

## Testing

### Quick Test
```bash
python3 training/train_ppo.py --iterations 2
```

### Full Training
```bash
python3 training/train_ppo.py --iterations 100
```

## Platform-Specific Notes

### macOS
- Uses `~/ray_spill`, `~/ray_temp`, `~/ray_logs` directories
- GPU support limited to MPS (Metal Performance Shaders) - set `num_gpus=0` if not available
- All dependencies install via `pip3` or `python3 -m pip`

### Windows
- Uses `C:/ray_spill`, `C:/ray_temp`, `C:/ray_logs` directories
- Full CUDA GPU support available
- Worker count limited to 2 (3+ causes deadlocks on Windows)

### Linux
- Uses `~/ray_spill`, `~/ray_temp`, `~/ray_logs` directories
- Full CUDA GPU support available
- Can use more workers (4-8) for better parallelization

## Known Issues and Warnings

### Deprecation Warnings (Non-Breaking)
1. **Object Spilling Config:** Uses environment variable instead of `object_spilling_directory` parameter
2. **RunConfig Import:** Should import from `ray.tune` instead of `ray.air`
3. **Observation Space Warnings:** PettingZoo environment uses `observation_spaces` dict instead of `observation_space()` method

### Action Items for Future
1. Implement learning rate scheduling using callbacks
2. Implement entropy coefficient scheduling using callbacks
3. Consider migrating to new API stack (requires complete model rewrite)
4. Update RunConfig import from `ray.tune`

## Version Compatibility

| Component | Original Version | Current Version | Status |
|-----------|------------------|-----------------|--------|
| Ray       | 2.9.0           | 2.50.1          | ✅ Working |
| RLlib     | 2.9.0           | 2.50.1          | ✅ Working (old API stack) |
| PyTorch   | N/A             | 2.9.0           | ✅ Working |
| Python    | 3.12            | 3.12.8          | ✅ Working |
| PettingZoo | N/A            | 1.25.0          | ✅ Working |

## Performance Notes

The performance optimizations from the original training script (v18.8) remain intact:
- 2 rollout workers with 1 environment each
- Train batch size: 2048
- Minibatch size: 2048
- Truncate episodes mode
- GPU acceleration (if available)

Expected training time for 100 iterations: 1.5-3 hours (depending on hardware)

## References

- [Ray 2.50 Migration Guide](https://docs.ray.io/en/latest/rllib/new-api-stack-migration-guide.html)
- [RLlib Old vs New API Stack](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html)
- [PPO Configuration](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo)

---

**Last Updated:** 2025-10-26
**Migration By:** Claude Code
**Original Code:** Windows-optimized Ray 2.9.0
**Target Platform:** macOS with Ray 2.50.1
