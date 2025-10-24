#!/usr/bin/env python
"""
Simple test script to verify Qwen 2.5 model loading and basic functionality.
Usage: python test_qwen.py [--model MODEL_NAME]
"""

import argparse
import torch
from model import ImmutableLM

def test_model_loading(model_name):
    """Test if the model can be loaded successfully."""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}\n")

    try:
        print("1. Loading model...")
        model = ImmutableLM(model_path=model_name)
        print(f"   ✓ Model loaded successfully")
        print(f"   - Model class: {model.backbone.__class__.__name__}")
        print(f"   - Tokenizer class: {model.tokenizer.__class__.__name__}")
        print(f"   - Vocab size: {len(model.tokenizer)}")

        # Move to GPU if available
        if torch.cuda.is_available():
            model.cuda()
            print(f"   - Device: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            print(f"   - Device: CPU")

        # Test tokenization
        print("\n2. Testing tokenization...")
        test_text = "input: This is a test sentence.\ntype: positive\n\n"
        tokens = model.tokenizer.encode(test_text)
        decoded = model.tokenizer.decode(tokens)
        print(f"   ✓ Tokenization works")
        print(f"   - Input text: {repr(test_text)}")
        print(f"   - Token count: {len(tokens)}")
        print(f"   - Decoded text: {repr(decoded)}")

        # Test generation
        print("\n3. Testing text generation...")
        input_ids = torch.LongTensor([tokens]).to(model.backbone.device)
        with torch.no_grad():
            outputs = model.backbone(input_ids)
            logits = outputs[0]
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            next_word = model.tokenizer.decode([next_token])
        print(f"   ✓ Generation works")
        print(f"   - Next predicted token: {repr(next_word)}")

        # Test label token restriction (for classification)
        print("\n4. Testing restricted token prediction...")
        label_words = ['positive', 'negative', 'neutral']
        label_tokens = [model.tokenizer.encode(word)[0] for word in label_words]
        print(f"   - Test labels: {label_words}")
        print(f"   - Label token IDs: {label_tokens}")

        # Get probabilities for restricted tokens
        probs = torch.softmax(next_token_logits, dim=-1)
        label_probs = [(word, probs[token].item()) for word, token in zip(label_words, label_tokens)]
        label_probs.sort(key=lambda x: x[1], reverse=True)
        print(f"   ✓ Restricted prediction works")
        for word, prob in label_probs:
            print(f"     - {word}: {prob:.4f}")

        print(f"\n{'='*60}")
        print(f"✓ All tests passed for {model_name}!")
        print(f"{'='*60}\n")

        return True

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print(f"\nFull traceback:")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test Qwen 2.5 model loading and functionality")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                       help="Model name to test (default: Qwen/Qwen2.5-0.5B)")
    args = parser.parse_args()

    success = test_model_loading(args.model)

    if success:
        print("\nYou can now run experiments with this model:")
        print(f"  python main.py --config config/agnews.yaml --model {args.model} --nshot 4 --seed 1")

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
