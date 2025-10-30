# ==== DBPEDIA ordering sensitivity (0/1/2 label templating) ====
# This experiment probes ordering effects with 30-shot prompts composed of
# three affinity buckets relative to the test example (different L1 = 0,
# same L1 = 1, same L1 & L2 = 2).

import random
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------- Config ----------------------
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # informational only
train_file = "data/dbpedia/train_hierarchical.jsonl"
test_file = "data/dbpedia/test_hierarchical.jsonl"
demo_source_split_name = "train"
test_source_split_name = "test"

shots_per_label = 10
label_values = (0, 1, 2)
k_examples = shots_per_label * len(label_values)
num_random_baseline_templates = 3
num_test_instances_to_run = 200
topk_logit_report = 5
print(
    f"[CFG] shots={k_examples} ({shots_per_label} per label) "
    f"random_templates={num_random_baseline_templates} tests_target={num_test_instances_to_run}"
)

# ---------------------- Load model and tokenizer ----------------------
# Use a smaller model for testing (change to model_id for production)
test_model_id = "gpt2"
print(f"[Load] Loading model: {test_model_id}")
tokenizer = AutoTokenizer.from_pretrained(test_model_id)
model = AutoModelForCausalLM.from_pretrained(
    test_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
print(f"[Load] Model loaded on device: {model.device}")

device = model.device

# ---------------------- Load dataset ----------------------
print("[Load] DBPEDIA hierarchical dataset")
dataset_full = load_dataset(
    "json",
    data_files={"train": train_file, "test": test_file},
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
    mapping = {}
    for label in label_list:
        with_space = tokenizer.encode(" " + label, add_special_tokens=False)
        no_space = tokenizer.encode(label, add_special_tokens=False)
        if with_space and len(with_space) == 1:
            mapping[label] = with_space[0]
        elif no_space and len(no_space) == 1:
            mapping[label] = no_space[0]
        elif with_space:
            mapping[label] = with_space[0]
        elif no_space:
            mapping[label] = no_space[0]
    return mapping


l2_token_map = build_label_token_map(all_l2, tokenizer)
l3_token_map = build_label_token_map(all_l3, tokenizer)
print(
    f"[Tokenizable] L2: {len(l2_token_map)}/{len(all_l2)} | "
    f"L3: {len(l3_token_map)}/{len(all_l3)}"
)


# ---------------------- Prompt & logits ----------------------
def format_dbpedia_prompt(test_row, demos, target_level):
    assert target_level in ("l2", "l3")
    q_label = "Level 2" if target_level == "l2" else "Level 3"

    parts = []
    for demo in demos:
        parts.append(
            f"Text: {demo['text']}\n"
            f"Level 1: {demo['l1']}\n"
            f"Question: What is the {q_label} category?\n"
            f"Answer: {demo[target_level]}"
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
    if inputs["input_ids"].shape[1] >= max_len:
        original_len = inputs["input_ids"].shape[1]
        to_trunc = original_len - max_len + 5
        if to_trunc > 0:
            inputs["input_ids"] = inputs["input_ids"][:, to_trunc:]
            inputs["attention_mask"] = inputs["attention_mask"][:, to_trunc:]
    outputs = model(**inputs)
    last = outputs.logits[0, -1, :]
    return {label: last[token].item() for label, token in label_token_map.items()}


def predict_from_logits(label_logits):
    ordered = sorted(label_logits.items(), key=lambda item: item[1], reverse=True)
    return ordered[0][0], ordered


# ---------------------- Demo sampling helpers ----------------------
def prepare_demo_lookup(dataset):
    by_l1 = defaultdict(list)
    by_l1_l2 = defaultdict(list)
    for idx in range(len(dataset)):
        row = dataset[idx]
        entry = {
            "text": row["text"],
            "l1": row["l1"],
            "l2": row["l2"],
            "l3": row["l3"],
            "_dataset_index": idx,
        }
        by_l1[entry["l1"]].append(entry)
        by_l1_l2[(entry["l1"], entry["l2"])] += [entry]
    label0_cache = {}
    for l1_val in by_l1:
        others = []
        for other_l1, rows in by_l1.items():
            if other_l1 != l1_val:
                others.extend(rows)
        label0_cache[l1_val] = others
    return by_l1, by_l1_l2, label0_cache


def sample_labelled_demos(test_row, by_l1, by_l1_l2, label0_cache, samples_per_label):
    test_l1 = test_row["l1"]
    test_l2 = test_row["l2"]

    label2_pool = by_l1_l2.get((test_l1, test_l2), [])
    label1_pool = [row for row in by_l1.get(test_l1, []) if row["l2"] != test_l2]
    label0_pool = label0_cache.get(test_l1, [])

    pools = {2: label2_pool, 1: label1_pool, 0: label0_pool}
    for label, pool in pools.items():
        if len(pool) < samples_per_label:
            return None
    return {label: random.sample(pool, samples_per_label) for label, pool in pools.items()}


def instantiate_order(label_sequence, labelled_samples):
    counters = {label: 0 for label in label_values}
    ordered = []
    for label in label_sequence:
        idx = counters[label]
        ordered.append(labelled_samples[label][idx])
        counters[label] += 1
    return ordered


def make_wrap_sequence(shots_per_label: int) -> List[int]:
    prefix_pairs = min(3, shots_per_label)
    middle_pairs = min(3, shots_per_label - prefix_pairs)
    remaining_pairs = shots_per_label - prefix_pairs - middle_pairs

    sequence: List[int] = []
    sequence.extend([value for _ in range(prefix_pairs) for value in (1, 2)])
    sequence.extend([0] * shots_per_label)
    sequence.extend([value for _ in range(middle_pairs) for value in (1, 2)])
    sequence.extend([value for _ in range(remaining_pairs) for value in (1, 2)])
    return sequence


def build_ordering_templates(shots_per_label: int, num_random: int) -> List[Tuple[str, Tuple[int, ...]]]:
    total_shots = shots_per_label * len(label_values)
    templates: List[Tuple[str, Tuple[int, ...]]] = []

    def add_template(name: str, seq: List[int]):
        if len(seq) != total_shots:
            raise ValueError(f"Template '{name}' has len={len(seq)}, expected {total_shots}")
        counts = {label: seq.count(label) for label in label_values}
        expected = {label: shots_per_label for label in label_values}
        if counts != expected:
            raise ValueError(f"Template '{name}' counts {counts}, expected {expected}")
        templates.append((name, tuple(seq)))

    add_template("sorted_ascending", [0] * shots_per_label + [1] * shots_per_label + [2] * shots_per_label)
    add_template("sorted_descending", [2] * shots_per_label + [1] * shots_per_label + [0] * shots_per_label)
    add_template(
        "interleaved_012",
        [value for _ in range(shots_per_label) for value in (0, 1, 2)],
    )
    add_template(
        "interleaved_210",
        [value for _ in range(shots_per_label) for value in (2, 1, 0)],
    )
    add_template("wrap_1212_zero_middle", make_wrap_sequence(shots_per_label))
    add_template(
        "alternating_then_zeros",
        [value for _ in range(shots_per_label) for value in (1, 2)] + [0] * shots_per_label,
    )
    add_template(
        "zeros_then_alternating",
        [0] * shots_per_label + [value for _ in range(shots_per_label) for value in (1, 2)],
    )
    zero_front = shots_per_label // 2
    zero_back = shots_per_label - zero_front
    add_template(
        "zero_sandwich",
        [0] * zero_front + [value for _ in range(shots_per_label) for value in (1, 2)] + [0] * zero_back,
    )
    add_template("cluster_twos_middle", [0] * shots_per_label + [2] * shots_per_label + [1] * shots_per_label)
    edge_block = shots_per_label // 2
    center_block = shots_per_label - edge_block
    add_template(
        "mirror_blocks",
        [2] * edge_block + [1] * edge_block + [0] * shots_per_label + [1] * center_block + [2] * center_block,
    )
    add_template(
        "twos_edges",
        [2] * edge_block + [0] * shots_per_label + [1] * shots_per_label + [2] * center_block,
    )

    base = [0] * shots_per_label + [1] * shots_per_label + [2] * shots_per_label
    for ridx in range(1, num_random + 1):
        seq = base[:]
        random.shuffle(seq)
        add_template(f"random_baseline_{ridx}", seq)

    return templates


# ---------------------- Prepare caches ----------------------
by_l1, by_l1_l2, label0_cache = prepare_demo_lookup(demo_source_data)
random.seed(42)
ordering_templates = build_ordering_templates(shots_per_label, num_random_baseline_templates)
print(f"[Templates] Using {len(ordering_templates)} orderings: {[name for name, _ in ordering_templates]}")
for template_name, label_sequence in ordering_templates:
    seq_preview = "".join(str(label) for label in label_sequence)
    print(f"  {template_name}: {seq_preview}")
random_templates = [
    (name, "".join(str(label) for label in sequence))
    for name, sequence in ordering_templates
    if name.startswith("random_baseline_")
]
if random_templates:
    print("[Templates] Random baseline permutations:")
    for template_name, seq_preview in random_templates:
        print(f"  {template_name}: {seq_preview}")

candidate_indices = list(range(len(test_source_data)))
random.shuffle(candidate_indices)
sampled_demos_by_test: Dict[int, Dict[int, List[Dict]]] = {}
test_row_cache: Dict[int, Dict] = {}
skipped_due_to_pool = 0

for idx in candidate_indices:
    if len(sampled_demos_by_test) >= num_test_instances_to_run:
        break
    raw_row = test_source_data[idx]
    test_row = {key: raw_row[key] for key in ("text", "l1", "l2", "l3")}
    sampled = sample_labelled_demos(test_row, by_l1, by_l1_l2, label0_cache, shots_per_label)
    if sampled is None:
        skipped_due_to_pool += 1
        continue
    sampled_demos_by_test[idx] = sampled
    test_row_cache[idx] = test_row

selected_test_indices = list(sampled_demos_by_test.keys())
print(
    f"[Select] Prepared {len(selected_test_indices)} test rows "
    f"(skipped {skipped_due_to_pool} lacking 10-per-bucket demos)"
)

if not selected_test_indices:
    raise RuntimeError("No test instances satisfy the 10-per-label sampling requirement.")

# ---------------------- Experiment ----------------------
targets = [("l3", l3_token_map, all_l3)]
overall_results = {level_name: [] for level_name, _, _ in targets}

print(
    f"\n==== RUNNING EXPERIMENT: tests={len(selected_test_indices)} "
    f"shots={k_examples} orderings={len(ordering_templates)} ===="
)
printed_prompt_templates = set()
for target_level, token_map, label_vocab in targets:
    print(f"\n-- Target: {target_level.upper()} --")
    for run_idx, test_idx in enumerate(selected_test_indices, 1):
        test_row = test_row_cache[test_idx]
        gold = test_row[target_level]
        labelled_shots = sampled_demos_by_test[test_idx]

        order_results = []
        for order_name, label_sequence in ordering_templates:
            ordered_demos = instantiate_order(label_sequence, labelled_shots)
            prompt = format_dbpedia_prompt(test_row, ordered_demos, target_level)
            logits = get_label_logits(prompt, token_map)
            if logits is None:
                continue
            if order_name not in printed_prompt_templates:
                print(f"\n[Prompt Template][{target_level.upper()}] {order_name}:\n{prompt}")
                printed_prompt_templates.add(order_name)
            prediction, sorted_logits = predict_from_logits(logits)
            label_names = list(logits.keys())
            logit_tensor = torch.tensor([logits[label] for label in label_names], dtype=torch.float32)
            log_probs = torch.log_softmax(logit_tensor, dim=0)
            gold_loss = None
            if gold in logits:
                gold_idx = label_names.index(gold)
                gold_loss = float(-log_probs[gold_idx])
            else:
                print(
                    f"[Warn] Gold label '{gold}' missing from logits for template '{order_name}' "
                    f"test_idx={test_idx}"
                )
            order_results.append(
                {
                    "name": order_name,
                    "label_sequence": list(label_sequence),
                    "prediction": prediction,
                    "correct": prediction == gold,
                    "gold": gold,
                    "logits_topk": sorted_logits[:topk_logit_report],
                    "cross_entropy": gold_loss,
                    "order_trace": [
                        {
                            "label": label_sequence[pos],
                            "l1": ordered_demos[pos]["l1"],
                            "l2": ordered_demos[pos]["l2"],
                            "l3": ordered_demos[pos]["l3"],
                        }
                        for pos in range(k_examples)
                    ],
                }
            )

        overall_results[target_level].append(
            {
                "test_index": int(test_idx),
                "test_l1": test_row["l1"],
                "test_l2": test_row["l2"],
                "gold": gold,
                "orders": order_results,
            }
        )
    print(f"[Done] {target_level.upper()} on {len(selected_test_indices)} tests.")


# ---------------------- Summaries ----------------------
def summarize(results_for_level, level_name, ordering_templates):
    if not results_for_level:
        print(f"No results to summarize for {level_name}.")
        return

    template_names = [name for name, _ in ordering_templates]
    template_stats = {
        name: {"correct": 0, "total": 0, "loss_sum": 0.0, "loss_count": 0}
        for name in template_names
    }
    sensitive_instances = 0
    total_orders = 0
    total_correct = 0
    total_loss_sum = 0.0
    total_loss_count = 0

    for test_result in results_for_level:
        preds = set()
        for order in test_result["orders"]:
            name = order["name"]
            template_stats[name]["total"] += 1
            total_orders += 1
            if order["correct"]:
                template_stats[name]["correct"] += 1
                total_correct += 1
            loss = order.get("cross_entropy")
            if loss is not None:
                template_stats[name]["loss_sum"] += loss
                template_stats[name]["loss_count"] += 1
                total_loss_sum += loss
                total_loss_count += 1
            preds.add(order["prediction"])
        if len(preds) > 1:
            sensitive_instances += 1

    print(f"\n=== Summary: {level_name.upper()} ===")
    print(f"Orders evaluated: {total_orders}")
    if total_orders:
        print(f"Aggregate accuracy: {total_correct / total_orders:.2%} ({total_correct}/{total_orders})")
    if total_loss_count:
        print(
            f"Aggregate cross-entropy: {total_loss_sum / total_loss_count:.4f} "
            f"(n={total_loss_count})"
        )
    print(
        f"Sensitive test instances (order changes prediction): "
        f"{sensitive_instances}/{len(results_for_level)}"
    )
    print("Per-template accuracy:")
    for name in template_names:
        stats = template_stats[name]
        if stats["total"] == 0:
            print(f"  {name}: n/a")
            continue
        acc = stats["correct"] / stats["total"]
        if stats["loss_count"]:
            avg_loss = stats["loss_sum"] / stats["loss_count"]
            print(
                f"  {name}: {acc:.2%} ({stats['correct']}/{stats['total']}) | "
                f"xent={avg_loss:.4f} (n={stats['loss_count']})"
            )
        else:
            print(f"  {name}: {acc:.2%} ({stats['correct']}/{stats['total']}) | xent=n/a")


for target_level, _, _ in targets:
    summarize(overall_results[target_level], target_level, ordering_templates)
print("\n[Finished] DBPEDIA ordering experiment.")
