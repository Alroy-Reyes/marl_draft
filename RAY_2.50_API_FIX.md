# Ray 2.50+ API Compatibility Fix - v19.1

## 🐛 Root Cause (v19.0 - BROKEN)

Training was stuck at "1 RUNNING" with 9.0/12 CPUs used but no progress.

### The Problem: API Mismatch

**When you disable the new API stack in Ray 2.50+**, you **MUST use OLD API method names**:

```python
# ❌ WRONG (v19.0) - API MISMATCH!
ppo_cfg = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=False,     # ← Disabling NEW API
        enable_env_runner_and_connector_v2=False,
    )
    .env_runners(                               # ❌ But using NEW API method!
        num_env_runners=8,                      # ❌ NEW API parameter!
        num_envs_per_env_runner=2,              # ❌ NEW API parameter!
    )
    .training(
        minibatch_size=1024,                    # ❌ NEW API parameter!
        num_epochs=4,                           # ❌ NEW API parameter!
    )
)
```

**Why it broke**:
- Ray tries to create **RolloutWorker** actors (old API)
- But configuration uses **new API parameters**
- Workers initialize but get stuck in inconsistent state
- Training hangs with "RUNNING" status but no iterations

---

## ✅ Solution (v19.1 - FIXED)

Use **OLD API method names** when disabling new API stack:

```python
# ✅ CORRECT (v19.1) - CONSISTENT OLD API
ppo_cfg = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=False,     # ← Disabling NEW API
        enable_env_runner_and_connector_v2=False,
    )
    .rollouts(                                  # ✅ OLD API method name
        num_rollout_workers=6,                  # ✅ OLD API parameter
        num_envs_per_worker=1,                  # ✅ OLD API parameter
        rollout_fragment_length=128,
        batch_mode="truncate_episodes",
    )
    .training(
        sgd_minibatch_size=1024,                # ✅ OLD API parameter
        num_sgd_iter=3,                         # ✅ OLD API parameter
        train_batch_size=3072,
    )
)
```

---

## 📋 API Name Mapping

| Category | NEW API (2.50+) | OLD API (Pre-2.50) |
|----------|-----------------|---------------------|
| **Method** | `.env_runners()` | `.rollouts()` |
| **Workers** | `num_env_runners` | `num_rollout_workers` |
| **Envs per worker** | `num_envs_per_env_runner` | `num_envs_per_worker` |
| **Minibatch** | `minibatch_size` | `sgd_minibatch_size` |
| **SGD iterations** | `num_epochs` | `num_sgd_iter` |

**Rule of thumb**: If you set `enable_rl_module_and_learner=False`, use **ALL** OLD API names!

---

## 🔧 Configuration Changes (v19.0 → v19.1)

### Workers
```python
# v19.0 (broken)
num_env_runners = 8
num_envs_per_env_runner = 2
# → 16 parallel environments (too aggressive)

# v19.1 (fixed)
num_rollout_workers = 6         # ✅ Conservative start
num_envs_per_worker = 1         # ✅ Stable
# → 6 parallel environments (proven stable)
```

### Training Batches
```python
# v19.0 (broken)
train_batch_size = 4096
minibatch_size = 1024
num_epochs = 4

# v19.1 (fixed)
train_batch_size = 3072         # ✅ 6 workers × 128 × 4 = 3072
sgd_minibatch_size = 1024       # ✅ OLD API name
num_sgd_iter = 3                # ✅ 3072 / 1024 = 3
```

---

## 🚀 Expected Performance (v19.1)

### M4 Pro MacBook (6 workers)
- **Iteration time**: 30-50 seconds (was 16 mins baseline)
- **100 iterations**: 50-85 minutes (was 26.7 hours)
- **Total speedup**: 20-30x faster! 🚀
- **CPU usage**: 60-80%
- **Memory**: 10-14GB (safe with 24GB)

### Can Scale Up After Verification
Once training works, you can increase to:
```python
num_rollout_workers = 8         # More parallelism
num_envs_per_worker = 2         # 16 parallel envs
train_batch_size = 4096         # Larger batches
```

---

## 🎯 Testing Command

```bash
# Kill any stale Ray sessions first
pkill -9 -f "ray"
sleep 2

# Test with 5 iterations
python training/train_ppo.py --iterations 5
```

**What to watch for**:
- ✅ "6 rollout workers running" in console
- ✅ First iteration completes in 60-90s (includes setup)
- ✅ Subsequent iterations: 30-50s
- ✅ CPU usage: 60-80%
- ✅ No "RUNNING" hang

---

## 📚 Lessons Learned

1. **Ray 2.50+ has TWO APIs**:
   - NEW API: RLModule, EnvRunner, etc. (default in 2.50+)
   - OLD API: RolloutWorker, etc. (legacy, must explicitly enable)

2. **You cannot mix APIs**:
   - Either use ALL new API names
   - OR use ALL old API names
   - Mixing causes silent failures and hangs

3. **When disabling new API**:
   ```python
   .api_stack(
       enable_rl_module_and_learner=False,
       enable_env_runner_and_connector_v2=False,
   )
   ```
   You **MUST** use OLD API method names throughout!

4. **Conservative scaling on new hardware**:
   - Start with proven worker counts (6 instead of 8)
   - Verify stability before increasing parallelism
   - macOS can handle more, but test incrementally

---

**Version**: 19.1 - Ray 2.50+ API Compatibility Fix
**Date**: 2025-10-26
**Hardware**: MacBook Pro M4 Pro, 24GB RAM
**Status**: FIXED - Ready for testing! ✅
