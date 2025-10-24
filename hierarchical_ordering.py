# ==== DBPEDIA ordering sensitivity (L2/L3) with HWE over L1-match scores ====
# Drop this in AFTER your model/tokenizer are loaded.

import math, random, itertools, numpy as np
from datasets import load_dataset, concatenate_datasets
import torch

# ---------------------- Config ----------------------
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # informational
# Using local hierarchical JSONL files
train_file = "data/dbpedia/train_hierarchical.jsonl"
test_file = "data/dbpedia/test_hierarchical.jsonl"
demo_source_split_name = "train"
test_source_split_name = "test"

k_examples = 3
num_permutations_to_test = math.factorial(k_examples) if k_examples <= 5 else 24
num_test_instances_to_run = 200
print(f"[CFG] shots={k_examples} perms/test={num_permutations_to_test} tests={num_test_instances_to_run}")

# ---------------------- Load model and tokenizer ----------------------
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use a smaller model for testing (change to model_id for production)
test_model_id = "gpt2"  # Using gpt2 for faster testing
print(f"[Load] Loading model: {test_model_id}")
tokenizer = AutoTokenizer.from_pretrained(test_model_id)
model = AutoModelForCausalLM.from_pretrained(test_model_id, torch_dtype=torch.float16, device_map="auto")
print(f"[Load] Model loaded on device: {model.device}")

device = model.device

# ---------------------- Load dataset ----------------------
print("[Load] DBPEDIA hierarchical dataset")
dataset_full = load_dataset(
    "json",
    data_files={
        "train": train_file,
        "test": test_file
    }
)

available_splits = list(dataset_full.keys())
if demo_source_split_name not in dataset_full or test_source_split_name not in dataset_full:
    only_split = available_splits[0]
    print(f"[Split] Re-splitting '{only_split}' 80/20 into train/test")
    tmp = dataset_full[only_split].train_test_split(test_size=0.2, seed=42)
    dataset_full = {"train": tmp["train"], "test": tmp["test"]}

demo_source_data = dataset_full[demo_source_split_name]
test_source_data = dataset_full[test_source_split_name]

need_cols = {"text", "l1", "l2", "l3"}
have_cols = set(demo_source_data.features.keys())
if not need_cols.issubset(have_cols):
    raise ValueError(f"Dataset must contain columns {need_cols}, got {have_cols}")

print(f"[Info] demos={len(demo_source_data)} tests={len(test_source_data)}")

full_for_vocab = concatenate_datasets([demo_source_data, test_source_data])
all_l1 = sorted(set(full_for_vocab["l1"]))
all_l2 = sorted(set(full_for_vocab["l2"]))
all_l3 = sorted(set(full_for_vocab["l3"]))
print(f"[Vocab] L1={len(all_l1)} L2={len(all_l2)} L3={len(all_l3)}")

# ---------------------- Token map helpers ----------------------
def build_label_token_map(label_list, tokenizer):
    m = {}
    for s in label_list:
        with_space = tokenizer.encode(" " + s, add_special_tokens=False)
        no_space  = tokenizer.encode(s, add_special_tokens=False)
        if with_space and len(with_space) == 1:
            m[s] = with_space[0]
        elif no_space and len(no_space) == 1:
            m[s] = no_space[0]
        elif with_space:
            m[s] = with_space[0]
        elif no_space:
            m[s] = no_space[0]
    return m

l2_token_map = build_label_token_map(all_l2, tokenizer)
l3_token_map = build_label_token_map(all_l3, tokenizer)
print(f"[Tokenizable] L2: {len(l2_token_map)}/{len(all_l2)} | L3: {len(l3_token_map)}/{len(all_l3)}")

# ---------------------- Prompting & logits ----------------------
def format_dbpedia_prompt(test_row, demos, target_level):
    assert target_level in ("l2", "l3")
    q_label = "Level 2" if target_level == "l2" else "Level 3"

    parts = []
    for d in demos:
        parts.append(
            f"Text: {d['text']}\n"
            f"Level 1: {d['l1']}\n"
            f"Question: What is the {q_label} category?\n"
            f"Answer: {d[target_level]}"
        )
    parts.append(
        f"Text: {test_row['text']}\n"
        f"Level 1: {test_row['l1']}\n"
        f"Question: What is the {q_label} category?\n"
        f"Answer:"
    )
    return "\n\n".join(parts).strip() + " "

@torch.no_grad()
def get_label_logits(prompt_str, label_token_map):
    if not label_token_map:
        return None
    inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
    max_len = tokenizer.model_max_length
    if inputs['input_ids'].shape[1] >= max_len:
        original_len = inputs['input_ids'].shape[1]
        to_trunc = original_len - max_len + 5
        if to_trunc > 0:
            inputs['input_ids'] = inputs['input_ids'][:, to_trunc:]
            inputs['attention_mask'] = inputs['attention_mask'][:, to_trunc:]
    outputs = model(**inputs)
    last = outputs.logits[0, -1, :]
    return {lbl: last[tok].item() for lbl, tok in label_token_map.items()}

def compute_hwe_from_l1_scores(l1_match_scores, n):
    if n <= 0 or not l1_match_scores:
        return {'HWE': None, 'HWE_normalized': None, 'total_score': None}
    D = sum(l1_match_scores)
    if D == 0:
        hwe = 1.0 / n
        return {'HWE': hwe, 'HWE_normalized': 0.0, 'total_score': 0.0}
    hwe = 0.0
    for k, s_k in enumerate(l1_match_scores, start=1):
        E_k = (n + 1 - k) / n
        hwe += (s_k / D) * E_k
    min_v = 1.0 / n
    hwe_norm = (hwe - min_v) / (1.0 - min_v)
    return {'HWE': hwe, 'HWE_normalized': hwe_norm, 'total_score': float(D)}

def predict_from_logits(label_logits):
    items = sorted(label_logits.items(), key=lambda t: t[1], reverse=True)
    return items[0][0], items

# ---------------------- Experiment ----------------------
random.seed(42)
targets = [("l2", l2_token_map, all_l2), ("l3", l3_token_map, all_l3)]
selected_test_indices = random.sample(
    range(len(test_source_data)),
    min(num_test_instances_to_run, len(test_source_data))
)

def choose_demos(k, pool):
    if len(pool) < k:
        raise ValueError("Not enough demos.")
    idxs = random.sample(range(len(pool)), k)
    return [pool[i] for i in idxs]

overall_results = {"l2": [], "l3": []}

print(f"\n==== RUNNING EXPERIMENT: tests={len(selected_test_indices)} shots={k_examples} perms={num_permutations_to_test} ====")
for target_level, token_map, label_vocab in targets:
    print(f"\n-- Target: {target_level.upper()} --")
    for run_idx, test_i in enumerate(selected_test_indices, 1):
        test_row = test_source_data[test_i]
        gold = test_row[target_level]
        demos = choose_demos(k_examples, demo_source_data)
        l1_scores = [1 if d["l1"] == test_row["l1"] else 0 for d in demos]

        perms = list(itertools.permutations(range(k_examples)))
        if len(perms) > num_permutations_to_test:
            perms = random.sample(perms, num_permutations_to_test)

        perm_results = []
        for perm in perms:
            ordered_demos = [{k: demos[j][k] for k in ("text","l1","l2","l3")} for j in perm]
            ordered_scores = [l1_scores[j] for j in perm]
            hwe = compute_hwe_from_l1_scores(ordered_scores, k_examples)

            prompt = format_dbpedia_prompt(test_row, ordered_demos, target_level)
            logits = get_label_logits(prompt, token_map)
            if not logits:
                continue
            pred, sorted_logits = predict_from_logits(logits)

            perm_results.append({
                "order": [(ordered_demos[j]["l1"], ordered_demos[j][target_level]) for j in range(k_examples)],
                "prediction": pred,
                "correct": pred == gold,
                "gold": gold,
                "HWE": hwe["HWE"],
                "HWE_normalized": hwe["HWE_normalized"],
                "total_l1_score": hwe["total_score"],
                "logits_top3": sorted_logits[:3],
            })

        overall_results[target_level].append({
            "test_index": int(test_i),
            "test_l1": test_row["l1"],
            "gold": gold,
            "permutations": perm_results
        })
    print(f"[Done] {target_level.upper()} on {len(selected_test_indices)} tests.")

# ---------------------- Summaries ----------------------
def summarize(results_for_level, level_name):
    print(f"\n=== Summary: {level_name.upper()} ===")
    total_perms = sum(len(x["permutations"]) for x in results_for_level)
    total_correct = sum(sum(1 for p in x["permutations"] if p["correct"]) for x in results_for_level)
    if total_perms == 0:
        print("No permutations evaluated."); return

    all_hwe = [p["HWE"] for x in results_for_level for p in x["permutations"] if p["HWE"] is not None]
    all_hwe_n = [p["HWE_normalized"] for x in results_for_level for p in x["permutations"] if p["HWE_normalized"] is not None]
    sensitive = sum(1 for x in results_for_level if len({p["prediction"] for p in x["permutations"]}) > 1)

    print(f"Permutations total: {total_perms}")
    print(f"Accuracy across permutations: {total_correct/total_perms:.2%}")
    print(f"Sensitive test instances (order changes prediction): {sensitive}/{len(results_for_level)}")
    if all_hwe:
        print(f"HWE  mean={np.mean(all_hwe):.3f} median={np.median(all_hwe):.3f} std={np.std(all_hwe):.3f}")
    if all_hwe_n:
        print(f"HWE* mean={np.mean(all_hwe_n):.3f} median={np.median(all_hwe_n):.3f} std={np.std(all_hwe_n):.3f}")

    # Optional corr between HWE* and correctness
    xs, ys = [], []
    for x in results_for_level:
        for p in x["permutations"]:
            if p["HWE_normalized"] is not None:
                xs.append(p["HWE_normalized"])
                ys.append(1 if p["correct"] else 0)
    if xs:
        try:
            from scipy.stats import pearsonr
            r, pv = pearsonr(xs, ys)
            print(f"Corr(HWE*, correctness): r={r:.3f}, p={pv:.4f}")
        except Exception:
            pass

summarize(overall_results["l2"], "l2")
summarize(overall_results["l3"], "l3")
print("\n[Finished] DBPEDIA ordering experiment.")