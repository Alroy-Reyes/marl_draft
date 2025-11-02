# EMERGENCY ROLLBACK GUIDE

**Problem**: Training is SLOWER after optimizations (16+ mins, no iteration completed)

**Root Cause**: Likely Windows + Ray + 8 workers causing deadlock/overhead

---

## 🚨 IMMEDIATE ACTION

### Step 1: Kill Training

```bash
# Press Ctrl+C
# If frozen, close terminal window
```

### Step 2: Kill All Ray Processes

```bash
# Windows Command Prompt (run as admin):
taskkill /F /IM python.exe
taskkill /F /IM ray.exe

# Or Task Manager → End all python.exe processes
```

---

## 🔧 ROLLBACK CONFIGURATION

### Option A: Conservative Rollback (Start Here)

**File**: `training/train_ppo.py` (line ~1070)

**CHANGE FROM** (current - broken):
```python
.rollouts(
    num_rollout_workers=8,
    rollout_fragment_length=128,
    batch_mode="truncate_episodes",
    num_envs_per_worker=2,
    observation_filter="MeanStdFilter",  # ← REMOVE THIS
)
```

**CHANGE TO** (safe):
```python
.rollouts(
    num_rollout_workers=2,              # ← REDUCE from 8 to 2 (Windows issue)
    rollout_fragment_length=128,
    batch_mode="truncate_episodes",
    num_envs_per_worker=1,              # ← REDUCE from 2 to 1
    # observation_filter removed          ← COMMENT OUT
)
```

**AND ADJUST** (line ~1086):
```python
train_batch_size=512,               # ← REDUCE from 2048 (2 workers × 128 × 2)
sgd_minibatch_size=512,             # ← Keep at 512
num_sgd_iter=1,                     # ← Keep at 1
```

**Expected**:
- 2 workers × 1 env = 2 parallel environments
- Should complete iteration in 3-5 minutes
- Stable, predictable

---

### Option B: Minimal Rollback (If Option A still slow)

**Complete rollback to KNOWN WORKING config**:

```python
.rollouts(
    num_rollout_workers=1,              # ← Back to single worker
    rollout_fragment_length=64,         # ← Original
    batch_mode="complete_episodes",     # ← Original
    num_envs_per_worker=1,
    # No observation_filter
)
.training(
    # ... other params ...
    train_batch_size=512,
    sgd_minibatch_size=256,
    num_sgd_iter=10,
    normalize_advantage=False,          # ← DISABLE
    # ... rest ...
    _enable_amp=False,                  # ← DISABLE mixed precision
)
```

This is the **baseline** - should work but be slow (8-16 mins/iteration).

---

## 🔍 Diagnostic Steps

### After Rollback, Test Incrementally

**1. Test Baseline First**
```bash
python training/train_ppo.py --iterations 2
```
- Should see "Iter 1: Time=XXXs" within 8-16 minutes
- If this works, you have a baseline

**2. Add ONE Optimization at a Time**

**Test A: Add 1 more worker**
```python
num_rollout_workers=2  # Instead of 1
train_batch_size=1024  # Adjust for 2 workers
```

Run 2 iterations. If faster, continue. If slower, stay at 1 worker.

**Test B: Add mixed precision** (if Test A works)
```python
_enable_amp=True
```

Run 2 iterations. Check GPU usage in nvidia-smi.

**Test C: Add advantage normalization** (if Test B works)
```python
normalize_advantage=True
```

Run 2 iterations. Check if value loss is stable.

---

## 🐛 Common Windows + Ray Issues

### Issue 1: Too Many Workers

**Symptom**: Training hangs, no iteration completes
**Fix**: Reduce to 1-2 workers max on Windows

### Issue 2: Observation Filter Initialization

**Symptom**: First iteration takes 10x longer
**Fix**: Remove `observation_filter="MeanStdFilter"`

### Issue 3: Ray Object Store Overflow

**Symptom**: High memory usage, slow iteration
**Fix**: Reduce `num_envs_per_worker` to 1

### Issue 4: Mixed Precision Compatibility

**Symptom**: GPU idle, training slow
**Fix**: Disable `_enable_amp=False`

---

## 📊 Expected Performance (Realistic for Windows)

### With Single Worker (Baseline)
- num_rollout_workers=1
- Iteration time: 8-16 minutes
- Stable, reliable

### With 2 Workers (Optimized for Windows)
- num_rollout_workers=2
- Iteration time: 4-8 minutes (2x speedup)
- Good balance for Windows

### With 4 Workers (Aggressive for Windows)
- num_rollout_workers=4
- Iteration time: 2-4 minutes (4x speedup)
- May be unstable on some Windows systems

**Linux**: Can handle 8-10 workers easily
**Windows**: Typically maxes out at 2-4 workers due to Ray limitations

---

## 🎯 Recommended Windows Configuration

```python
# SAFE CONFIGURATION FOR WINDOWS
.rollouts(
    num_rollout_workers=2,              # Sweet spot for Windows
    rollout_fragment_length=128,
    batch_mode="truncate_episodes",
    num_envs_per_worker=1,
    # No observation_filter (can add later if stable)
)
.training(
    gamma=0.95,
    lr=5e-4,
    train_batch_size=512,               # 2 workers × 128 × 2
    sgd_minibatch_size=1024,            # GPU optimization
    num_sgd_iter=1,
    normalize_advantage=True,           # Usually safe
    _enable_amp=True,                   # GPU optimization
    # ... rest of config ...
)
```

**Expected**:
- Iteration time: 3-5 minutes
- Stable on Windows
- 3-5x speedup from baseline
- 100 iterations = 5-8 hours (vs 26 hours baseline)

---

## ✅ Checklist After Rollback

- [ ] Kill all Python/Ray processes
- [ ] Reduce workers to 2
- [ ] Reduce envs_per_worker to 1
- [ ] Remove observation_filter
- [ ] Adjust train_batch_size to match workers
- [ ] Test with --iterations 2
- [ ] Verify iteration completes within 5 minutes
- [ ] Check nvidia-smi shows GPU activity
- [ ] Monitor RAM stays under 14GB

---

## 📞 Next Steps

1. **Apply Option A** (2 workers, no filter)
2. **Test with 2 iterations**
3. **Report back**:
   - Did iteration complete?
   - How long did it take?
   - What's GPU memory usage?
   - What's CPU usage?

Then we can incrementally add optimizations that work on Windows.

---

## 🔬 Ray on Windows Known Issues

Ray has documented issues on Windows:
- Worker communication overhead is higher
- Object store can be slower
- Recommended max workers: 2-4 (not 8-10 like Linux)

**Alternative**: Consider using WSL2 (Windows Subsystem for Linux) for better Ray performance.
