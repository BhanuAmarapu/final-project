try:
    from transformers import pipeline
    import torch
    print("[OK] Can import transformers and torch")
    device = 0 if torch.cuda.is_available() else -1
    print(f"Device: {device} (CUDA: {torch.cuda.is_available()})")
except Exception as e:
    print(f"[X] Failed: {e}")
