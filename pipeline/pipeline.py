import json
import numpy as np
from pathlib import Path


class Pipeline:
    """Executes a linear chain of nodes and manages data flow."""

    def __init__(self, head):
        self.head = head

    def run(self, initial_data: dict = None) -> dict:
        """Execute all nodes in sequence, passing data through the chain."""
        data = initial_data or {}
        node = self.head
        while node is not None:
            print(f"[Pipeline] Running: {node.name}")
            node.validate(data)
            data = node.process(data)
            node = node._next
        print("[Pipeline] Done.")
        return data

    @staticmethod
    def save_manifest(data: dict, output_dir: str):
        """
        Serialize frame data to disk so the Blender script can read it.
        Splits into manifest.json (scalars/strings) and arrays.npz (numpy arrays).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_data = {}
        arrays = {}
        _split_data(data, json_data, arrays, prefix="")

        with open(output_dir / "manifest.json", "w") as f:
            json.dump(json_data, f, indent=2)

        np.savez(output_dir / "arrays.npz", **arrays)

    @staticmethod
    def load_manifest(output_dir: str) -> dict:
        """Reconstruct data dict from saved manifest (used inside Blender script)."""
        output_dir = Path(output_dir)
        with open(output_dir / "manifest.json") as f:
            json_data = json.load(f)
        arrays = dict(np.load(output_dir / "arrays.npz", allow_pickle=False))
        return _merge_data(json_data, arrays)


def _split_data(data: dict, json_out: dict, arrays_out: dict, prefix: str):
    """Recursively split a nested dict into JSON-serializable parts and numpy arrays."""
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, np.ndarray):
            arrays_out[full_key] = value
            json_out[key] = {"__array__": full_key}
        elif isinstance(value, dict):
            json_out[key] = {}
            _split_data(value, json_out[key], arrays_out, full_key)
        elif isinstance(value, (int, float, str, bool, list, type(None))):
            json_out[key] = value
        else:
            # Skip non-serializable values (e.g. raw audio data objects)
            pass


def _merge_data(json_data: dict, arrays: dict) -> dict:
    """Reconstruct nested dict by replacing array references with actual arrays."""
    result = {}
    for key, value in json_data.items():
        if isinstance(value, dict):
            if "__array__" in value:
                result[key] = arrays[value["__array__"]]
            else:
                result[key] = _merge_data(value, arrays)
        else:
            result[key] = value
    return result
