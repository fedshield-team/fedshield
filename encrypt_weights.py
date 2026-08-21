"""FedShield — AES-256-GCM authenticated weight encryption."""

import base64
import io
import os

import torch

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes


KEY_FILE = os.path.join(
    "models",
    "fedshield_aes.key"
)

NONCE_SIZE = 12
TAG_SIZE = 16


def generate_key() -> bytes:
    """Generate a new 256-bit AES key."""

    key = get_random_bytes(32)

    os.makedirs(
        os.path.dirname(KEY_FILE),
        exist_ok=True
    )

    with open(KEY_FILE, "wb") as f:
        f.write(key)

    try:
        os.chmod(
            KEY_FILE,
            0o600
        )
    except OSError:
        pass

    print(
        f"[Crypto] AES-256 key generated "
        f"and saved to {KEY_FILE}"
    )

    return key


def load_key() -> bytes:
    """Load the shared AES-256 key."""

    if not os.path.exists(KEY_FILE):
        return generate_key()

    with open(KEY_FILE, "rb") as f:
        key = f.read()

    if len(key) != 32:
        raise ValueError(
            f"Invalid key length: "
            f"{len(key)} bytes "
            f"(expected 32)"
        )

    return key


def _serialize(weights: list) -> bytes:
    """Serialize tensors safely to CPU."""

    buffer = io.BytesIO()

    torch.save(
        [
            w.detach().cpu()
            for w in weights
        ],
        buffer
    )

    return buffer.getvalue()


def encrypt_weights(
    weights: list,
    key: bytes
) -> dict:
    """Encrypt model weights using AES-256-GCM."""

    if len(key) != 32:
        raise ValueError(
            "AES-256 requires a 32-byte key"
        )

    nonce = get_random_bytes(
        NONCE_SIZE
    )

    cipher = AES.new(
        key,
        AES.MODE_GCM,
        nonce=nonce,
        mac_len=TAG_SIZE
    )

    ciphertext, tag = (
        cipher.encrypt_and_digest(
            _serialize(weights)
        )
    )

    return {
        "version": 2,
        "algorithm": "AES-256-GCM",
        "nonce": base64.b64encode(
            nonce
        ).decode("ascii"),
        "ciphertext": base64.b64encode(
            ciphertext
        ).decode("ascii"),
        "tag": base64.b64encode(
            tag
        ).decode("ascii"),
        "num_layers": len(weights)
    }


def decrypt_weights(
    payload: dict,
    key: bytes
) -> list:
    """Decrypt and authenticate encrypted model weights."""

    if len(key) != 32:
        raise ValueError(
            "AES-256 requires a 32-byte key"
        )

    if payload.get("algorithm") != "AES-256-GCM":
        raise ValueError(
            "Unsupported or legacy encryption payload; "
            "re-encrypt with AES-256-GCM"
        )

    try:

        nonce = base64.b64decode(
            payload["nonce"]
        )

        ciphertext = base64.b64decode(
            payload["ciphertext"]
        )

        tag = base64.b64decode(
            payload["tag"]
        )

    except (
        KeyError,
        ValueError
    ) as e:

        raise ValueError(
            "Malformed encrypted payload"
        ) from e

    cipher = AES.new(
        key,
        AES.MODE_GCM,
        nonce=nonce,
        mac_len=len(tag)
    )

    try:

        plaintext = (
            cipher.decrypt_and_verify(
                ciphertext,
                tag
            )
        )

    except ValueError as e:

        raise ValueError(
            "Encrypted weights failed "
            "authentication or were tampered with"
        ) from e

    return torch.load(
        io.BytesIO(plaintext),
        map_location="cpu",
        weights_only=True
    )


def secure_send(weights: list) -> dict:
    """Encrypt weights using the shared key."""

    return encrypt_weights(
        weights,
        load_key()
    )


def secure_receive(payload: dict) -> list:
    """Decrypt weights using the shared key."""

    return decrypt_weights(
        payload,
        load_key()
    )


if __name__ == "__main__":

    import torch.nn as nn

    print("=" * 60)
    print("FedShield AES-256-GCM Self Test")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(41, 16),
        nn.ReLU(),
        nn.Linear(16, 5)
    )

    original = [
        p.detach().clone()
        for p in model.parameters()
    ]

    key = generate_key()

    payload = encrypt_weights(
        original,
        key
    )

    recovered = decrypt_weights(
        payload,
        key
    )

    ok = all(
        torch.equal(a, b)
        for a, b in zip(
            original,
            recovered
        )
    )

    print(
        f"AES-256-GCM round-trip: "
        f"{'PASS' if ok else 'FAIL'}"
    )

    # Tamper test
    tampered = dict(payload)

    tampered["ciphertext"] = (
        tampered["ciphertext"][:-2]
        + (
            "AA"
            if tampered["ciphertext"][-2:] != "AA"
            else "BB"
        )
    )

    try:

        decrypt_weights(
            tampered,
            key
        )

        print(
            "Tamper detection: FAIL"
        )

    except ValueError:

        print(
            "Tamper detection: PASS"
        )