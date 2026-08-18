# merge_chunks.py
import torch
from pathlib import Path
from torch_geometric.data import InMemoryDataset

chunks_dir = Path("panda_dataset/tmp_chunks")
all_data = []
for f in sorted(chunks_dir.glob("chunk_*.pt")):
    all_data.extend(torch.load(f, weights_only=False))

print(f"Total examples: {len(all_data)}")
data, slices = InMemoryDataset.collate(all_data)
output_dir = Path("panda_dataset/processed")
output_dir.mkdir(parents=True, exist_ok=True)
torch.save((data, slices), output_dir / "data.pt")
print("Merged dataset saved")