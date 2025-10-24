# Qwen 2.5 Migration Guide

## Summary of Changes

This codebase has been updated to support modern language models (specifically Qwen 2.5) while maintaining backward compatibility with the original GPT-2 models.

## Modified Files

### 1. `model.py`
**Changes:**
- Replaced `GPT2LMHeadModel` and `GPT2Tokenizer` with `AutoModelForCausalLM` and `AutoTokenizer`
- Added `trust_remote_code=True` parameter for Qwen models
- Added automatic padding token setup for models that don't define one
- Updated OOM (Out of Memory) handling to detect model sizes generically
- Made model size detection work with both GPT-2 naming (gpt2-xl) and Qwen naming (7B, 14B, etc.)

**Why:** The original code was hardcoded for GPT-2. Using Auto classes makes it compatible with any causal language model from Hugging Face.

### 2. `main.py`
**Changes:**
- Updated `--model` argument help text to show Qwen examples
- No breaking changes - all existing functionality preserved

### 3. New Files Created

#### `requirements.txt`
Python dependencies needed to run the experiments:
- torch>=2.0.0
- transformers>=4.40.0
- numpy, scipy, scikit-learn
- pyyaml, easydict, tqdm

#### `script/run_agnews_qwen.sh`
Example script to run the full experimental pipeline with Qwen 2.5 models. Based on the original `run_agnews.sh` but:
- Uses Qwen/Qwen2.5-0.5B and Qwen/Qwen2.5-1.5B (can be extended to larger models)
- Handles model name cleaning (replaces / with _ for filenames)
- Reduced seeds (1-3) for faster initial testing
- Outputs to separate directory: `experiment/agnews_qwen/`

#### `test_qwen.py`
Verification script to test that Qwen models load and work correctly:
- Tests model loading
- Tests tokenization
- Tests text generation
- Tests restricted token prediction (for classification tasks)

## Compatibility

### Backward Compatibility
✓ **All original GPT-2 functionality preserved**
- Original scripts (`run_agnews.sh`, etc.) still work
- Can still use: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`

### New Model Support
✓ **Any Hugging Face causal language model**
- Qwen 2.5: `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-3B`, `Qwen/Qwen2.5-7B`
- Other models: Llama, Mistral, Phi, etc. (untested but should work)

## Research Implications

### Original Paper (2021)
The ACL 2022 paper found that prompt ordering has **significant impact** on GPT-2 performance:
- Same examples in different orders: near SOTA → random guess performance
- Effect exists across all model sizes
- Not transferable between models
- 13% relative improvement possible with entropy-based ordering selection

### New Research Question
**Does this phenomenon still exist in 2025 with modern models like Qwen 2.5?**

Possible outcomes:
1. **Still exists**: Ordering sensitivity is fundamental to few-shot learning
2. **Reduced**: Better training/architecture has made models more robust
3. **Different patterns**: Sensitivity exists but manifests differently

## Running Experiments

### Quick Test (Recommended First Step)
```bash
# Test that Qwen loads correctly
python test_qwen.py --model Qwen/Qwen2.5-0.5B

# Run a single inference test
python main.py --config config/agnews.yaml \
  --model Qwen/Qwen2.5-0.5B \
  --nshot 4 \
  --seed 1 \
  --output test_output
```

### Full Comparison
```bash
# Original GPT-2 experiments
cd script
bash run_agnews.sh

# New Qwen 2.5 experiments
bash run_agnews_qwen.sh

# Results will be in:
# - experiment/agnews/     (GPT-2)
# - experiment/agnews_qwen/ (Qwen)
```

### Memory Requirements

**Qwen 2.5 Models:**
- 0.5B: ~1-2GB VRAM (works on most GPUs)
- 1.5B: ~3-4GB VRAM
- 3B: ~6-8GB VRAM
- 7B: ~14-16GB VRAM

**Tips for limited memory:**
- Start with Qwen/Qwen2.5-0.5B
- Use smaller batch sizes
- The code includes OOM handling that will split batches automatically

## Expected Workflow

1. **Install dependencies:** `pip install -r requirements.txt`
2. **Test Qwen loading:** `python test_qwen.py`
3. **Run quick test:** Single experiment with 1 seed
4. **Run full pipeline:** `bash script/run_agnews_qwen.sh`
5. **Analyze results:** Compare entropy patterns between GPT-2 and Qwen
6. **Extend to other datasets:** Modify config files in `config/`

## Troubleshooting

### "No module named 'transformers'"
```bash
pip install -r requirements.txt
```

### "CUDA out of memory"
- Try smaller model: `Qwen/Qwen2.5-0.5B`
- The code will automatically split batches for large models

### Slow downloads
Qwen models will be downloaded from Hugging Face on first use. This is a one-time download:
- 0.5B: ~1GB
- 1.5B: ~3GB
- 7B: ~14GB

### Trust remote code warning
The code uses `trust_remote_code=True` for Qwen models. This is safe for official Qwen models from `Qwen/` organization on Hugging Face.

## Next Steps

Consider testing:
1. **Multiple datasets**: SST-2, RTE, TREC (configs already exist)
2. **Different n-shot values**: 1, 2, 4, 8, 16
3. **Different model sizes**: Compare 0.5B vs 7B
4. **Cross-model transfer**: Do good orderings transfer between Qwen sizes?
