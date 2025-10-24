# Ordered-Prompt

Code for ACL 2022 paper "Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity".

Run scripts under script folder to get ordered prompt results. Or download the [prompt checkpoints](https://drive.google.com/file/d/1DFLdX0DPfSUqmiSxU2dxzJrHfGJSl-XK/view?usp=sharing) to start analysis immediately.

## Modern Model Support (Qwen 2.5)

This codebase has been updated to support modern language models, specifically **Qwen 2.5**, to test whether prompt ordering sensitivity still exists with newer models.

### Installation

```bash
pip install -r requirements.txt
```

### Using Qwen 2.5 Models

The codebase now uses `AutoModelForCausalLM` and `AutoTokenizer` from Hugging Face Transformers, making it compatible with any causal language model, including:

**Qwen 2.5 Models:**
- `Qwen/Qwen2.5-0.5B` (smallest, ~1GB VRAM)
- `Qwen/Qwen2.5-1.5B` (~3GB VRAM)
- `Qwen/Qwen2.5-3B` (~6GB VRAM)
- `Qwen/Qwen2.5-7B` (~14GB VRAM)

**Original GPT-2 Models:**
- `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`

### Running Experiments with Qwen 2.5

#### Quick Test (Single Run)

```bash
# Run inference with Qwen 2.5 (0.5B model)
python main.py --config config/agnews.yaml \
  --model Qwen/Qwen2.5-0.5B \
  --nshot 4 \
  --seed 1 \
  --output results_qwen

# Run with larger model
python main.py --config config/agnews.yaml \
  --model Qwen/Qwen2.5-1.5B \
  --nshot 4 \
  --seed 1 \
  --output results_qwen
```

#### Full Experimental Pipeline

Use the provided script to run the complete pipeline (generation → augmentation → evaluation → entropy analysis):

```bash
cd script
bash run_agnews_qwen.sh
```

This script will:
1. Generate synthetic examples using Qwen 2.5
2. Augment the dataset with generated examples
3. Run inference on multiple prompt orderings
4. Analyze entropy to identify ordering sensitivity
5. Compare results across different seeds and model sizes

### Comparing GPT-2 vs Qwen 2.5

To test if ordering sensitivity has changed with modern models:

```bash
# Run with GPT-2 (original)
bash script/run_agnews.sh

# Run with Qwen 2.5 (modern)
bash script/run_agnews_qwen.sh

# Compare results in experiment/agnews/ and experiment/agnews_qwen/
```

### Research Question

Does prompt ordering sensitivity still significantly affect modern models like Qwen 2.5, or have architectural improvements and training methods reduced this phenomenon? 
