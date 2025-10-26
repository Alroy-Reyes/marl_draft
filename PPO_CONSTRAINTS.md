# PPO Configuration Constraints - CRITICAL REFERENCE

## ❌ Common Mistakes to NEVER Make Again

### MISTAKE #1: sgd_minibatch_size > train_batch_size ❌

**Error Message**:
```
sgd_minibatch_size 4096 must be less than or equal to train_batch_size 1536
```

**Root Cause**:
PPO algorithm works by:
1. **Collecting** `train_batch_size` samples from rollout workers
2. **Splitting** that batch into chunks of `sgd_minibatch_size`
3. **Running** `num_sgd_iter` gradient updates on each chunk

**YOU CANNOT SPLIT A BATCH INTO CHUNKS LARGER THAN THE BATCH ITSELF!**

---

## ✅ Correct PPO Configuration Rules

### Rule #1: Minibatch Size Constraint
```python
sgd_minibatch_size <= train_batch_size  # ALWAYS!
```

**Example - WRONG**:
```python
train_batch_size=1536
sgd_minibatch_size=4096  # ❌ 4096 > 1536 - WILL CRASH!
```

**Example - CORRECT**:
```python
train_batch_size=4096
sgd_minibatch_size=4096  # ✅ 4096 <= 4096 - OK!
```

---

### Rule #2: SGD Iterations Calculation
```python
num_sgd_iter = train_batch_size / sgd_minibatch_size
```

**Examples**:

| train_batch_size | sgd_minibatch_size | num_sgd_iter | Result |
|------------------|---------------------|--------------|--------|
| 2048 | 512 | 4 | ✅ 4 iterations of 512 samples |
| 2048 | 1024 | 2 | ✅ 2 iterations of 1024 samples |
| 2048 | 2048 | 1 | ✅ 1 iteration of 2048 samples |
| 1536 | 4096 | ??? | ❌ IMPOSSIBLE - WILL CRASH |

---

### Rule #3: Train Batch Size Sources
```python
train_batch_size = num_rollout_workers × rollout_fragment_length × episodes_collected
```

**In practice**:
- With `batch_mode="truncate_episodes"`, workers collect fragments asynchronously
- `train_batch_size` should be set to desired total samples
- RLlib will collect until it reaches this amount

**Common configurations**:

| Workers | Fragment | Approx Batch Size |
|---------|----------|-------------------|
| 1 | 128 | 512-1024 |
| 2 | 128 | 1024-2048 |
| 3 | 128 | 1536-3072 |
| 4 | 128 | 2048-4096 |

---

## 🎯 Optimizing for GPU Utilization

### Goal: Maximize GPU usage with large minibatches

**Step 1: Determine maximum minibatch your GPU can handle**
- Start with 512 (baseline)
- Check GPU memory with `nvidia-smi`
- Double minibatch if GPU memory < 50%
- Continue until GPU memory reaches 60-80%

**Step 2: Ensure train_batch_size >= sgd_minibatch_size**
```python
# WRONG APPROACH:
sgd_minibatch_size = 4096  # Set this first
train_batch_size = 1536    # Then realize it's too small - ERROR!

# RIGHT APPROACH:
train_batch_size = 4096      # Set this first (large enough)
sgd_minibatch_size = 4096    # Then set equal or smaller
num_sgd_iter = 1             # Calculate: 4096 / 4096 = 1
```

**Step 3: Adjust workers if needed**
```python
# If train_batch_size is too small for desired minibatch:
# Option A: Increase workers
num_rollout_workers = 4  # More workers = more samples

# Option B: Increase fragment length
rollout_fragment_length = 256  # Longer fragments = more samples

# Option C: Just increase train_batch_size directly
train_batch_size = 8192  # RLlib will collect until it has 8192 samples
```

---

## 📊 Recommended Configurations

### Conservative (Stable)
```python
num_rollout_workers = 2
rollout_fragment_length = 128
train_batch_size = 1024
sgd_minibatch_size = 512
num_sgd_iter = 2  # 1024 / 512 = 2
```

### Balanced (Good Performance)
```python
num_rollout_workers = 3
rollout_fragment_length = 128
train_batch_size = 2048
sgd_minibatch_size = 2048
num_sgd_iter = 1  # 2048 / 2048 = 1
```

### Aggressive (Maximum GPU Utilization)
```python
num_rollout_workers = 3
rollout_fragment_length = 128
train_batch_size = 4096
sgd_minibatch_size = 4096
num_sgd_iter = 1  # 4096 / 4096 = 1
```

### Extreme (If GPU Memory Allows)
```python
num_rollout_workers = 4
rollout_fragment_length = 256
train_batch_size = 8192
sgd_minibatch_size = 8192
num_sgd_iter = 1  # 8192 / 8192 = 1
```

---

## 🔍 Validation Checklist

Before running training, verify:

- [ ] `sgd_minibatch_size <= train_batch_size` ✅
- [ ] `num_sgd_iter = train_batch_size / sgd_minibatch_size` (approximately)
- [ ] `train_batch_size` is large enough for your desired minibatch
- [ ] GPU memory can handle the minibatch size (check with smaller test first)
- [ ] RAM can handle the train_batch_size (rough estimate: 100 MB per 1000 samples)

---

## 🐛 Debugging Common Issues

### Issue: "sgd_minibatch_size must be <= train_batch_size"
**Solution**: Increase `train_batch_size` to at least match `sgd_minibatch_size`

### Issue: "Collected batch size less than train_batch_size"
**Solution**:
- Increase `num_rollout_workers`
- Increase `rollout_fragment_length`
- Check if episodes are terminating too early

### Issue: GPU Out of Memory
**Solution**: Reduce `sgd_minibatch_size` by half

### Issue: Very slow iterations
**Solution**: Check if `train_batch_size` is too large - workers spend too much time collecting samples

---

## 📚 Additional Resources

- [RLlib PPO Documentation](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- Our project: `GPU_OPTIMIZATION_GUIDE.md` for GPU-specific tuning

---

## 🎓 Key Takeaway

**ALWAYS SET train_batch_size >= sgd_minibatch_size**

Think of it this way:
- `train_batch_size` = Size of the pizza
- `sgd_minibatch_size` = Size of each slice

**You can't cut a small pizza into giant slices!** 🍕

---

**Version**: 18.8
**Last Updated**: 2025-10-26
**Never Violate These Constraints Again!** ✅
