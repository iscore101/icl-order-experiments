#!/bin/bash
# Example script to run AGNews experiments with Qwen 2.5 models
# This tests if prompt ordering sensitivity still exists with modern models

MAIN_DIR=$(pwd)
DATASET=agnews
LOGDIR=experiment/${DATASET}_qwen;
SAMPLE_MODE=balance

mkdir -p $LOGDIR

# Test with different Qwen 2.5 model sizes
# Adjust models based on your available GPU memory:
# - Qwen/Qwen2.5-0.5B: smallest, fastest, ~1GB
# - Qwen/Qwen2.5-1.5B: small, ~3GB
# - Qwen/Qwen2.5-3B: medium, ~6GB
# - Qwen/Qwen2.5-7B: large, ~14GB

for NSHOT in 4;
do

# Start with smaller models for testing
for MODEL in "Qwen/Qwen2.5-0.5B" "Qwen/Qwen2.5-1.5B";
do
for SEED in 1 2 3;  # Reduced seeds for initial testing
do

  for N in 3 5;
  do
     python main.py --config config/$DATASET.yaml \
     --nshot $NSHOT --model $MODEL --output $LOGDIR --seed $SEED \
     --ngram $N --generate --temperature 2.0 --topk 20 --do_sample

     echo "python main.py --config config/$DATASET.yaml \
     --nshot $NSHOT --model $MODEL --output $LOGDIR --seed $SEED \
     --ngram $N --generate --temperature 2.0 --topk 20 --do_sample"
  done;

  cd $LOGDIR;

  for f in generate*;
  do
      python "$MAIN_DIR"/augment.py $f
  done;

  mkdir -p ckpt;
  mv *.pkl ckpt;

  # Clean model name for filename (replace / with _)
  MODEL_CLEAN=$(echo $MODEL | tr '/' '_')

  OUTPUT=dev_${DATASET}_${NSHOT}_shot_${MODEL_CLEAN}_seed${SEED}.jsonl
  cat augment_*.jsonl > $OUTPUT
  mv augment_*.jsonl ckpt

  cd "${MAIN_DIR}" || exit;

  echo "python main.py --config config/$DATASET.yaml \
     --nshot $NSHOT --model $MODEL --output $LOGDIR --seed $SEED --test_data_path $LOGDIR/$OUTPUT;"

  python main.py --config config/$DATASET.yaml \
     --nshot $NSHOT --model $MODEL --output $LOGDIR --seed $SEED --test_data_path $LOGDIR/$OUTPUT

  mv $LOGDIR/${DATASET}_${NSHOT}_shot_${MODEL_CLEAN}_seed${SEED}_*.pkl $LOGDIR/fake_${DATASET}_${NSHOT}_shot_${MODEL_CLEAN}_seed${SEED}.pkl
  python main.py --config config/$DATASET.yaml \
     --nshot $NSHOT --model $MODEL --output $LOGDIR --seed $SEED
  mv $LOGDIR/${DATASET}_${NSHOT}_shot_${MODEL_CLEAN}_seed${SEED}_*.pkl $LOGDIR/true_${DATASET}_${NSHOT}_shot_${MODEL_CLEAN}_seed${SEED}.pkl

  python entropy.py --true $LOGDIR/true_${DATASET}_${NSHOT}_shot_${MODEL_CLEAN}_seed${SEED}.pkl \
                    --fake $LOGDIR/fake_${DATASET}_${NSHOT}_shot_${MODEL_CLEAN}_seed${SEED}.pkl \
                    --topk 4 --save $LOGDIR/result_${DATASET}_${NSHOT}_shot_${MODEL_CLEAN}_seed${SEED}.json

done;

done;

done;
