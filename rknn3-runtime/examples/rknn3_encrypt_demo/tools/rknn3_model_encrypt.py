#!/usr/bin/env python3
"""
RKNN3 Model Encryption Tool

Encrypts .rknn and .weight files using AES-CTR, and wraps the user AES key
inside an RSA key envelope using RK's public key.

Usage:
    python3 rknn3_model_encrypt.py --model model.rknn [--weight model.weight] \
                                   --rk-pubkey rk_public_key.pem \
                                   [--output-dir ./encrypted]

Output files:
    <output_dir>/model.rknn.enc        - AES-CTR encrypted model
    <output_dir>/model.weight.enc      - AES-CTR encrypted weight (if provided)
    <output_dir>/user_key_iv.enc       - RSA-encrypted key envelope (RKKE format)

File formats are compatible with rknn3_set_decrypt_key_from_path() /
rknn3_load_model_from_path() APIs defined in rknn3_api.h.
"""

import argparse
import os
import struct
import sys
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes

# Magic constants (must match rknn_crypto.h)
RKNN3_CRYPT_MODEL_MAGIC = 0x45434B52  # "RKCE"
RKNN3_CRYPT_KEY_MAGIC   = 0x454B4B52  # "RKKE"
RKNN3_CRYPT_ALG_AES256_CTR = 0x02

AES_IV_LEN  = 16

# Encrypted model header: 32 bytes
# uint32 magic, uint32 version, uint32 algorithm, uint32 reserved0,
# uint64 original_size, uint8[8] reserved
MODEL_HEADER_FORMAT = '<IIII Q 8s'
MODEL_HEADER_SIZE   = struct.calcsize(MODEL_HEADER_FORMAT)  # 32

# Encrypted key header: 16 bytes
# uint32 magic, uint32 version, uint32 rsa_key_bits, uint32 reserved
KEY_HEADER_FORMAT = '<IIII'
KEY_HEADER_SIZE   = struct.calcsize(KEY_HEADER_FORMAT)  # 16


def aes_key_len(aes_bits: int) -> int:
    if aes_bits != 256:
        raise ValueError(f'Unsupported AES key bits: {aes_bits}')
    return aes_bits // 8


def aes_algorithm(aes_bits: int) -> int:
    return RKNN3_CRYPT_ALG_AES256_CTR


def aes_ctr_encrypt(key: bytes, iv: bytes, plaintext: bytes) -> bytes:
    """Encrypt data with AES-CTR."""
    cipher = AES.new(key, AES.MODE_CTR, nonce=b'', initial_value=iv)
    return cipher.encrypt(plaintext)


def build_model_header(original_size: int, aes_bits: int, version: int = 1) -> bytes:
    """Build the 32-byte encrypted model file header."""
    return struct.pack(
        MODEL_HEADER_FORMAT,
        RKNN3_CRYPT_MODEL_MAGIC,
        version,
        aes_algorithm(aes_bits),
        0,                      # reserved0
        original_size,
        b'\x00' * 8             # reserved
    )


def build_key_envelope(aes_key: bytes, aes_iv: bytes, rk_pubkey_path: str) -> bytes:
    """
    Build the key envelope file:
      RKKE header (16 bytes) + RSA-encrypted (AES key + IV)
    """
    with open(rk_pubkey_path, 'r') as f:
        rk_pubkey = RSA.import_key(f.read())

    rsa_key_bits = rk_pubkey.size_in_bits()

    rsa_cipher = PKCS1_OAEP.new(rk_pubkey)
    plaintext = aes_key + aes_iv
    rsa_encrypted = rsa_cipher.encrypt(plaintext)

    header = struct.pack(
        KEY_HEADER_FORMAT,
        RKNN3_CRYPT_KEY_MAGIC,
        1,              # version
        rsa_key_bits,
        0               # reserved
    )

    return header + rsa_encrypted


def encrypt_file(input_path: str, aes_key: bytes, aes_iv: bytes, aes_bits: int) -> bytes:
    """Read a file and return its encrypted representation (header + ciphertext)."""
    with open(input_path, 'rb') as f:
        plaintext = f.read()

    original_size = len(plaintext)
    header = build_model_header(original_size, aes_bits)
    ciphertext = aes_ctr_encrypt(aes_key, aes_iv, plaintext)

    return header + ciphertext


def main():
    parser = argparse.ArgumentParser(
        description='RKNN3 Model Encryption Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--model', required=True, help='Path to .rknn model file')
    parser.add_argument('--weight', default=None, help='Path to .weight file (optional)')
    parser.add_argument('--rk-pubkey', required=True, help='Path to RK RSA public key (PEM)')
    parser.add_argument('--output-dir', default='.', help='Output directory (default: current)')
    parser.add_argument('--aes-bits', type=int, choices=(256,), default=256,
                        help='AES key size in bits (default: 256)')
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f'Error: model file not found: {args.model}', file=sys.stderr)
        sys.exit(1)
    if args.weight and not os.path.isfile(args.weight):
        print(f'Error: weight file not found: {args.weight}', file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.rk_pubkey):
        print(f'Error: RK public key not found: {args.rk_pubkey}', file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Generate random AES key and IV
    aes_key = get_random_bytes(aes_key_len(args.aes_bits))
    aes_iv  = get_random_bytes(AES_IV_LEN)
    # print(f'Generated AES-{args.aes_bits} key: {aes_key.hex()}')
    # print(f'Generated IV:          {aes_iv.hex()}')

    # 2. Encrypt model file
    model_basename = os.path.basename(args.model)
    enc_model_path = os.path.join(args.output_dir, model_basename + '.enc')
    enc_model_data = encrypt_file(args.model, aes_key, aes_iv, args.aes_bits)
    with open(enc_model_path, 'wb') as f:
        f.write(enc_model_data)
    # print(f'Encrypted model:  {enc_model_path} ({len(enc_model_data)} bytes)')

    # 3. Encrypt weight file (if provided)
    if args.weight:
        weight_basename = os.path.basename(args.weight)
        enc_weight_path = os.path.join(args.output_dir, weight_basename + '.enc')
        enc_weight_data = encrypt_file(args.weight, aes_key, aes_iv, args.aes_bits)
        with open(enc_weight_path, 'wb') as f:
            f.write(enc_weight_data)
        print(f'Encrypted weight: {enc_weight_path} ({len(enc_weight_data)} bytes)')

    # 4. Build key envelope
    key_env_path = os.path.join(args.output_dir, 'user_key_iv.enc')
    key_env_data = build_key_envelope(aes_key, aes_iv, args.rk_pubkey)
    with open(key_env_path, 'wb') as f:
        f.write(key_env_data)
    print(f'Key envelope:     {key_env_path} ({len(key_env_data)} bytes)')

    print('\nDone. Deploy these files to the board:')
    print(f'  {enc_model_path}')
    if args.weight:
        print(f'  {enc_weight_path}')
    print(f'  {key_env_path}')


if __name__ == '__main__':
    main()
