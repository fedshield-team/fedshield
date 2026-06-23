"""
FedShield — AES-256 Weight Encryption
Encrypts model weights before transmission between federated nodes and aggregator.

Usage:
    from encrypt_weights import encrypt_weights, decrypt_weights

    # On sending node — encrypt before sending
    encrypted = encrypt_weights(weights, key)

    # On aggregator — decrypt after receiving
    weights = decrypt_weights(encrypted, key)
"""

import os
import io
import torch
import base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes


# ── Key management ────────────────────────────────────────────────────────────
KEY_FILE = "models/fedshield_aes.key"

def generate_key() -> bytes:
    """Generate a new 256-bit AES key and save it."""
    key = get_random_bytes(32)  # 256-bit
    os.makedirs("models", exist_ok=True)
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    print(f"[Crypto] AES-256 key generated and saved to {KEY_FILE}")
    return key

def load_key() -> bytes:
    """Load the shared AES key. Generate one if it doesn't exist."""
    if not os.path.exists(KEY_FILE):
        return generate_key()
    with open(KEY_FILE, "rb") as f:
        key = f.read()
    if len(key) != 32:
        raise ValueError(f"Invalid key length: {len(key)} bytes (expected 32)")
    return key


# ── Encrypt / Decrypt ─────────────────────────────────────────────────────────
def encrypt_weights(weights: list, key: bytes) -> dict:
    """
    Encrypt a list of PyTorch weight tensors using AES-256-CBC.

    Args:
        weights: list of torch.Tensor (model.parameters())
        key:     32-byte AES key

    Returns:
        dict with keys: iv (hex), ciphertext (base64), shape_info
    """
    # Serialize weights to bytes
    buffer = io.BytesIO()
    torch.save(weights, buffer)
    plaintext = buffer.getvalue()

    # AES-256-CBC encryption
    iv = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))

    return {
        "iv":         iv.hex(),
        "ciphertext": base64.b64encode(ciphertext).decode("utf-8"),
        "num_layers": len(weights)
    }


def decrypt_weights(payload: dict, key: bytes) -> list:
    """
    Decrypt AES-256-CBC encrypted weights back to a list of tensors.

    Args:
        payload: dict returned by encrypt_weights
        key:     32-byte AES key

    Returns:
        list of torch.Tensor
    """
    iv         = bytes.fromhex(payload["iv"])
    ciphertext = base64.b64decode(payload["ciphertext"])

    cipher    = AES.new(key, AES.MODE_CBC, iv)
    plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)

    buffer  = io.BytesIO(plaintext)
    weights = torch.load(buffer, weights_only=False)
    return weights


# ── Convenience wrappers for federated nodes ──────────────────────────────────
def secure_send(weights: list) -> dict:
    """Encrypt weights using the shared key. Call this before sending."""
    key = load_key()
    payload = encrypt_weights(weights, key)
    print(f"[Crypto] Weights encrypted — {len(payload['ciphertext'])} chars, IV: {payload['iv'][:8]}...")
    return payload


def secure_receive(payload: dict) -> list:
    """Decrypt weights using the shared key. Call this after receiving."""
    key = load_key()
    weights = decrypt_weights(payload, key)
    print(f"[Crypto] Weights decrypted — {payload['num_layers']} layers restored")
    return weights


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch.nn as nn

    print("=" * 50)
    print("FedShield AES-256 Weight Encryption — Self Test")
    print("=" * 50)

    # Build a small test model
    model = nn.Sequential(nn.Linear(41, 128), nn.ReLU(), nn.Linear(128, 5))
    original_weights = [p.data.clone() for p in model.parameters()]

    # Generate key
    key = generate_key()
    print(f"\n[Key] {key.hex()[:32]}... ({len(key)*8}-bit)")

    # Encrypt
    payload = encrypt_weights(original_weights, key)
    print(f"\n[Encrypt] IV:         {payload['iv']}")
    print(f"[Encrypt] Ciphertext: {payload['ciphertext'][:64]}...")
    print(f"[Encrypt] Layers:     {payload['num_layers']}")

    # Decrypt
    recovered = decrypt_weights(payload, key)

    # Verify
    all_match = all(
        torch.equal(orig, rec)
        for orig, rec in zip(original_weights, recovered)
    )

    print(f"\n[Verify] All weights match after decrypt: {all_match}")
    print("\n✅ AES-256 encryption working correctly!" if all_match else "\n❌ MISMATCH — something went wrong")

    # Show overhead
    import sys
    original_size = sum(p.numel() * 4 for p in model.parameters())
    encrypted_size = len(payload["ciphertext"])
    print(f"\n[Overhead] Original: {original_size:,} bytes | Encrypted: {encrypted_size:,} bytes")
    print("=" * 50)