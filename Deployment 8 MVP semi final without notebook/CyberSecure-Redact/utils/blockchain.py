"""
utils/blockchain.py
Very small block / chain helper to simulate immutable threat log chaining.
Each log entry becomes a block containing:
- timestamp
- data (the detection entry)
- previous_hash
- hash
"""

import hashlib
import json
import time
from typing import Dict

def compute_hash(entry: Dict, previous_hash: str = "") -> str:
    """Return SHA256 hex digest of entry + previous_hash."""
    # Ensure deterministic ordering
    s = json.dumps(entry, sort_keys=True, default=str) + str(previous_hash)
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def make_block(entry: Dict, previous_hash: str = "") -> Dict:
    """Create a block dictionary from entry and previous_hash."""
    block = {
        "timestamp": time.time(),
        "entry": entry,
        "previous_hash": previous_hash
    }
    block_hash = compute_hash(entry, previous_hash)
    block["hash"] = block_hash
    return block

def append_blockchain(log_path: str, entry: Dict) -> Dict:
    """
    Append a new block to the JSON chain file at log_path.
    Returns the newly created block.
    """
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            chain = json.load(f)
    except Exception:
        chain = []

    previous_hash = chain[-1]['hash'] if chain else ""
    block = make_block(entry, previous_hash)
    chain.append(block)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(chain, f, indent=2)
    return block
