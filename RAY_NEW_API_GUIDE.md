# Ray 2.50+ NEW API Guide - v19.2

## ✅ Using the NEW API (Recommended)

We're now using Ray 2.50+ **NEW env_runner API** with **OLD learner API** (hybrid approach).

### Configuration (v19.2)

```python
ppo_cfg = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=False,         # ❌ Keep old learner
        enable_env_runner_and_connector_v2=True,    # ✅ Enable NEW env_runner
    )
    .env_runners(                           # ✅ NEW API method
        num_env_runners=6,                  # ✅ NEW API parameter
        num_envs_per_env_runner=1,          # ✅ NEW API parameter
        rollout_fragment_length=128,
    )
    .training(
        train_batch_size=3072,              # ✅ OLD API (still works with new env_runner)
        sgd_minibatch_size=1024,            # ✅ OLD API (still works with new env_runner)
        num_sgd_iter=3,                     # ✅ OLD API (still works with new env_runner)
    )
)
```

---

## 🎯 Hybrid API Approach (Best of Both Worlds)

### ✅ NEW env_runner API (Enabled)

**Why**: Better performance, modern architecture, future-proof

**Changes**:
- Method: `.env_runners()` instead of `.rollouts()`
- Workers: `num_env_runners` instead of `num_rollout_workers`
- Envs: `num_envs_per_env_runner` instead of `num_envs_per_worker`
- No need for `batch_mode="truncate_episodes"` (handled automatically)

**Benefits**:
- Cleaner API design
- Better error messages
- Improved performance
- Forward compatible with future Ray versions

### ❌ OLD learner API (Kept)

**Why**: Avoid rewriting our custom model (ImprovedSahaMaskedTwoHead)

**What we keep**:
- Custom `TorchModelV2` model class
- Training params: `sgd_minibatch_size`, `num_sgd_iter`, `train_batch_size`
- Action masking logic
- Multi-head architecture (teacher + slot)

**Future**: Can migrate to RLModule later if needed

---

## 📊 Comparison: OLD vs NEW vs HYBRID API

| Feature | OLD API (v19.1) | HYBRID (v19.2) | FULL NEW API |
|---------|-----------------|----------------|--------------|
| **Method** | `.rollouts()` | `.env_runners()` | `.env_runners()` |
| **Workers** | `num_rollout_workers` | `num_env_runners` | `num_env_runners` |
| **Learner** | Old (TorchModelV2) | Old (TorchModelV2) | New (RLModule) |
| **Training params** | Old names | Old names | New names |
| **Model rewrite** | ❌ Not needed | ❌ Not needed | ✅ Required |
| **Performance** | Good | Good | Potentially better |
| **Future-proof** | ⚠️ Deprecated | ✅ Supported | ✅ Recommended |
| **Effort** | Low | Low | High |

**Our choice**: HYBRID (v19.2) - Modern env_runner with existing model

---

## 🔧 Parameter Mapping

### Env Runner Configuration

| OLD API (v19.1) | NEW API (v19.2) |
|-----------------|-----------------|
| `.rollouts()` | `.env_runners()` |
| `num_rollout_workers` | `num_env_runners` |
| `num_envs_per_worker` | `num_envs_per_env_runner` |
| `batch_mode="truncate_episodes"` | (automatic) |

### Training Configuration (unchanged with hybrid approach)

| Parameter | Name (Same in both) |
|-----------|---------------------|
| Batch size | `train_batch_size` |
| Minibatch | `sgd_minibatch_size` |
| SGD iterations | `num_sgd_iter` |
| Learning rate | `lr` |
| Gamma | `gamma` |

---

## 🚀 Benefits of NEW API

1. **Better Performance**: Modern architecture, optimized for Ray 2.50+
2. **Cleaner Code**: More intuitive method names
3. **Future-Proof**: Won't be deprecated in future versions
4. **Better Errors**: Improved error messages and debugging
5. **Easy Migration**: Can still use old training params (hybrid approach)

---

## 🧪 Testing the NEW API

```bash
# Kill any stale sessions
pkill -9 -f "ray"
sleep 2

# Test with 5 iterations
python training/train_ppo.py --iterations 5
```

**Expected behavior (same as v19.1)**:
- ✅ First iteration: 60-90s (environment setup)
- ✅ Subsequent: 30-50s per iteration
- ✅ Console: "6 env_runners running" (note: "env_runners" not "rollout workers")
- ✅ CPU usage: 60-80%
- ✅ Memory: 10-14GB

---

## 📈 Scaling Recommendations

Once stable with 6 env_runners, you can increase:

### Conservative (8 runners)
```python
num_env_runners = 8
num_envs_per_env_runner = 1
train_batch_size = 4096
```
**Expected**: 25-40s/iter

### Aggressive (10 runners)
```python
num_env_runners = 10
num_envs_per_env_runner = 1
train_batch_size = 5120
```
**Expected**: 20-35s/iter

### Maximum (10 runners, 2 envs each)
```python
num_env_runners = 10
num_envs_per_env_runner = 2
train_batch_size = 10240
```
**Expected**: 15-25s/iter (60-100x faster than baseline!)
**Risk**: High memory usage (~18-20GB), thermal throttling possible

---

## ⚠️ Common Issues

### Issue 1: "Unknown parameter: num_rollout_workers"
**Cause**: Using OLD API parameter names with NEW API
**Fix**: Use `num_env_runners` instead

### Issue 2: "Unknown method: rollouts"
**Cause**: Using OLD API method with NEW env_runner enabled
**Fix**: Use `.env_runners()` instead of `.rollouts()`

### Issue 3: Training hangs
**Cause**: Stale Ray sessions or API mismatch
**Fix**: Kill all Ray processes and restart
```bash
pkill -9 -f "ray"
sleep 2
```

---

## 🎓 Migration Path (Future)

If we want to use FULL NEW API later (RLModule):

1. **Rewrite model** as RLModule class (instead of TorchModelV2)
2. **Enable new learner**: `enable_rl_module_and_learner=True`
3. **Update training params**:
   - `train_batch_size` → `train_batch_size_per_learner`
   - `sgd_minibatch_size` → `mini_batch_size_per_learner`
   - `num_sgd_iter` → `num_epochs`

**Effort**: 4-8 hours of work
**Benefit**: Potentially 10-20% better performance, full Ray 3.0 compatibility

**For now**: Hybrid approach (v19.2) is optimal!

---

**Version**: 19.2 - Ray 2.50+ NEW env_runner API (Hybrid)
**Date**: 2025-10-26
**Status**: Ready for testing! ✅
**Recommendation**: Use this configuration, it's modern and future-proof!
