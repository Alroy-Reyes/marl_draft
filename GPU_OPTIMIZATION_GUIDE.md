# GPU Memory Optimization Guide

**Current Status**: 500 MiB / 6144 MiB (8% utilization) - SEVERELY UNDERUTILIZED!

## Quick Fix: Increase Minibatch Size

### Current Configuration (Underutilized)
```python
sgd_minibatch_size=1024  # Only using 500MB
num_sgd_iter=2
```

### Recommended Configurations

#### **Option 1: Aggressive (Recommended for 6GB GPU)**
```python
sgd_minibatch_size=4096  # 4x larger - should use ~2-3GB
num_sgd_iter=1           # 1 iteration (since 4096 > 2048 train_batch)
train_batch_size=4096    # Match minibatch size
```

**Expected**:
- GPU memory: 2-3GB / 6GB (~50% utilization)
- Speedup: 2-3x faster GPU training
- Iteration time: 25-50s → **10-20s**

---

#### **Option 2: Moderate (Safe)**
```python
sgd_minibatch_size=2048  # 2x larger - should use ~1-1.5GB
num_sgd_iter=1           # 1 iteration (since 2048 = train_batch)
```

**Expected**:
- GPU memory: 1-1.5GB / 6GB (~25% utilization)
- Speedup: 1.5-2x faster GPU training
- Iteration time: 25-50s → **15-30s**

---

#### **Option 3: Maximum (Experimental)**
```python
sgd_minibatch_size=8192   # 8x larger - should use ~4-5GB
num_sgd_iter=1
train_batch_size=8192     # Increase train_batch too
rollout_fragment_length=256  # Increase fragment to match
```

**Expected**:
- GPU memory: 4-5GB / 6GB (~80% utilization)
- Speedup: 3-4x faster GPU training
- Iteration time: 25-50s → **8-15s**

---

## How to Apply

### Step 1: Check Current Performance

Before making changes, note:
- Current iteration time: _____s
- Current GPU usage during training: 500 MiB

### Step 2: Edit train_ppo.py

Find this section (around line 1059-1061):
```python
train_batch_size=2048,
sgd_minibatch_size=1024,
num_sgd_iter=2,
```

### Step 3: Try Option 2 First (Safest)

Replace with:
```python
train_batch_size=2048,              # Keep same
sgd_minibatch_size=2048,            # CHANGE: 1024 → 2048 (2x larger)
num_sgd_iter=1,                     # CHANGE: 2 → 1 (since minibatch = train_batch)
```

### Step 4: Monitor

Watch `nvidia-smi` during training:
```bash
# In another terminal:
watch -n 1 nvidia-smi

# Or on Windows:
nvidia-smi -l 1
```

**Look for**:
- GPU Memory usage should increase to 1-2GB
- GPU Utilization should spike to 90-100% during SGD
- Iteration time should drop

### Step 5: If Successful, Try Option 1

If Option 2 works and GPU memory is still low, try Option 1:

```python
train_batch_size=4096,              # CHANGE: 2048 → 4096
sgd_minibatch_size=4096,            # CHANGE: 2048 → 4096
num_sgd_iter=1,                     # Keep at 1
```

---

## Understanding the Settings

### Key Principle
**GPU memory usage scales with minibatch size**

- **1024 minibatch** → ~500 MB (your current usage)
- **2048 minibatch** → ~1 GB
- **4096 minibatch** → ~2 GB
- **8192 minibatch** → ~4 GB

### Why This Works

**Mixed Precision Training (FP16)** halves memory usage:
- Without FP16: 1024 minibatch uses ~1GB
- With FP16: 1024 minibatch uses ~500MB ✅ (what you're seeing)

This means you can **double the minibatch size** compared to FP32!

### Relationship Between Settings

```
train_batch_size = sgd_minibatch_size × num_sgd_iter

Examples:
- 2048 = 1024 × 2  ✅ (current)
- 2048 = 2048 × 1  ✅ (recommended)
- 4096 = 4096 × 1  ✅ (aggressive)
```

---

## Expected Performance Gains

### Current Performance (with optimizations)
- Iteration time: 25-50s
- GPU memory: 500 MB / 6 GB (8%)
- GPU utilization: Low during most of iteration

### With Option 2 (2048 minibatch)
- Iteration time: **15-30s** (1.5-2x faster)
- GPU memory: 1-1.5 GB / 6 GB (25%)
- GPU utilization: High during SGD

### With Option 1 (4096 minibatch)
- Iteration time: **10-20s** (2-3x faster)
- GPU memory: 2-3 GB / 6 GB (50%)
- GPU utilization: Very high during SGD

### Combined Speedup (from original 16 mins)
- Before all optimizations: 960s
- With parallelization + Phase 1: 25-50s (20-38x)
- With Option 2: **15-30s** (32-64x total!)
- With Option 1: **10-20s** (48-96x total!)

---

## Troubleshooting

### If You Get "Out of Memory" Error

**Symptoms**: CUDA OOM error, training crashes

**Fix**: Reduce minibatch size by half
```python
sgd_minibatch_size=1024  # Back to original
num_sgd_iter=2
```

### If Training Becomes Unstable

**Symptoms**: Loss spikes, NaN values

**Fix**: Reduce learning rate
```python
lr=2.5e-4,  # Half of current 5e-4
```

Or increase gradient clipping:
```python
grad_clip=0.5,  # Tighter than current 1.0
```

### If You Want Even More Speed

1. **Increase workers** (if you have CPU headroom):
   ```python
   num_rollout_workers=10  # From 8
   ```

2. **Increase envs per worker** (if you have RAM):
   ```python
   num_envs_per_worker=3  # From 2
   train_batch_size=6144  # 10 × 128 × 3 × 1.6 (adjustment factor)
   ```

---

## Quick Reference Table

| Config | Minibatch | Iter | GPU Mem | Speed | Risk |
|--------|-----------|------|---------|-------|------|
| Current | 1024 | 2 | 500MB | Baseline | None |
| Safe | 2048 | 1 | 1-1.5GB | 1.5-2x | Low |
| **Recommended** | **4096** | **1** | **2-3GB** | **2-3x** | **Low** |
| Aggressive | 8192 | 1 | 4-5GB | 3-4x | Medium |

---

## Hardware-Specific Notes

### 6GB GPU (Your Machine)
- Likely: GTX 1060 6GB, RTX 2060 6GB, or RTX 3050 6GB variant
- Safe maximum: 4096 minibatch (~3GB with FP16)
- Aggressive maximum: 8192 minibatch (~5GB with FP16)

### Why You Have Headroom
1. **Mixed Precision (FP16)**: Halves memory usage
2. **Model is relatively small**: ~2-3M parameters
3. **Observation space is modest**: ~300-500 features

---

## Recommended Action Plan

### Phase A: Quick Win (5 minutes)
1. Change to Option 2 (2048 minibatch)
2. Run 5 iterations
3. Verify speed improvement

### Phase B: Optimal (10 minutes)
1. Change to Option 1 (4096 minibatch)
2. Run 10 iterations
3. Monitor GPU memory stays under 4GB
4. Measure actual iteration time

### Phase C: Full Training (30-45 minutes)
1. Keep best configuration from Phase B
2. Train full 100 iterations
3. Should complete in 20-30 minutes!

---

## Expected Final Performance

**Best case scenario (with 4096 minibatch)**:

- **Iteration time**: 10-20 seconds
- **100 iterations**: 15-30 minutes
- **Total speedup**: 48-96x faster than original!
- **GPU utilization**: 50-80% memory, 90-100% compute

**From 26.7 hours to 20 minutes!** 🚀🚀🚀

---

## Copy-Paste Configurations

### For Immediate Use in train_ppo.py

**Option 2 (Safe - Recommended First Try)**:
```python
.rollouts(
    num_rollout_workers=8,
    rollout_fragment_length=128,
    batch_mode="truncate_episodes",
    num_envs_per_worker=2,
    observation_filter="MeanStdFilter",
)
.training(
    # ... other params ...
    train_batch_size=2048,
    sgd_minibatch_size=2048,      # ← CHANGED from 1024
    num_sgd_iter=1,                # ← CHANGED from 2
    # ... rest ...
)
```

**Option 1 (Aggressive - Try if Option 2 works well)**:
```python
.rollouts(
    num_rollout_workers=8,
    rollout_fragment_length=128,
    batch_mode="truncate_episodes",
    num_envs_per_worker=2,
    observation_filter="MeanStdFilter",
)
.training(
    # ... other params ...
    train_batch_size=4096,         # ← CHANGED from 2048
    sgd_minibatch_size=4096,       # ← CHANGED from 1024
    num_sgd_iter=1,                # ← CHANGED from 2
    # ... rest ...
)
```

---

## Questions to Answer

To help optimize further, please share:

1. **What GPU model is this?** (check with `nvidia-smi`)
2. **What is your actual iteration time?** (from console output)
3. **What's your CPU usage?** (Task Manager or `top`)
4. **What's your RAM usage?** (out of how much total?)

This will help me fine-tune the configuration even more precisely!
