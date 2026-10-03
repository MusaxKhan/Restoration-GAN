"""Generate the deterministic validation / test corruption manifests and the split file."""
import json
from pathlib import Path

from . import pets

if __name__ == "__main__":
    out = Path("manifests")
    out.mkdir(exist_ok=True)
    (out / "pets_split.json").write_text(json.dumps(pets.get_split_indices()))
    val = pets.build_val_manifest()
    (out / "val_manifest.json").write_text(json.dumps(val, separators=(",", ":")))
    test = pets.build_test_manifest()
    (out / "test_manifest.json").write_text(json.dumps(test, separators=(",", ":")))
    print(f"val entries: {len(val)}, test entries: {len(test)}")
