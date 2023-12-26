import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from glob import glob

import fire
import torch
import torch.utils
import torch.utils.data
from tokenizer import SpecialTokens, get_encoding


def partition_data(dataset, shard_index, num_shards, drop_last=False):
    if drop_last:
        chunk = len(dataset) // num_shards
    else:
        chunk = (len(dataset) + num_shards - 1) // num_shards
    data_slice = slice(shard_index * chunk, (shard_index + 1) * chunk)
    if isinstance(dataset, torch.utils.data.Dataset):
        return torch.utils.data.Subset(dataset, range(data_slice.start, data_slice.stop))
    else:
        return dataset[data_slice]


class TextTrainDataset(torch.utils.data.IterableDataset):
    def __init__(self, tensor: torch.Tensor, seq_length: int):
        self.tensor = tensor
        self.seq_length = seq_length

    def __iter__(self):
        return self

    def __next__(self):
        offset = torch.randint(len(self.tensor) - self.seq_length, (), dtype=torch.long)
        return self.tensor[offset : offset + self.seq_length]


class TextTestDataset(torch.utils.data.Dataset):
    def __init__(self, tensor: torch.Tensor, seq_length: int):
        size = (len(tensor) // seq_length) * seq_length
        self.tensor = tensor[:size].reshape(-1, seq_length)

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return self.tensor.size(0)


def create_dataset(
    path: str,
    train: bool,
    seq_length: int,
    shard_index: int = 0,
    num_shards: int = 1,
    data_dir: str = "./data",
):
    tokens = torch.cat([torch.load(path, weights_only=True) for path in glob(f"{data_dir}/{path}")])
    tokens = partition_data(tokens, shard_index, num_shards)
    print(tokens.shape)
    if train:
        dataset = TextTrainDataset(tokens, seq_length)
    else:
        dataset = TextTestDataset(tokens, seq_length)
    return dataset


def _process_file(item: tuple[str, str], local_dir: str, split: str):
    index, input_path = item
    output_path = f"{local_dir}/tokenized/{split}{index:02d}.pt"
    if os.path.exists(output_path):
        return
    print(f"Processing {input_path}")
    encoding = get_encoding()
    with open(input_path, "r", encoding="utf-8") as f:
        texts = []
        for item in json.load(f):
            story = item["story"].strip()
            text = f"{SpecialTokens.BEGIN}{story}{SpecialTokens.END}"
            texts.append(text)
        tokens = encoding.encode("".join(texts), allowed_special="all")
        tokens = torch.tensor(tokens, dtype=torch.long)
        torch.save(tokens, output_path)


def main(
    remote_url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz",
    local_dir="data/tinystories",
    split=0.98,
):
    filename = remote_url.split("/")[-1]
    os.makedirs(local_dir, exist_ok=True)
    if not os.path.exists(f"{local_dir}/{filename}"):
        print("Downloading the data...")
        os.system(f"wget -P {local_dir}/ {remote_url}")
    if not os.path.exists(f"{local_dir}/raw"):
        print("Extracting the data...")
        os.makedirs(f"{local_dir}/raw", exist_ok=True)
        os.system(f"tar -xvf {local_dir}/{filename} -C {local_dir}/raw")
    os.makedirs(f"{local_dir}/tokenized", exist_ok=True)
    paths = sorted(glob(f"{local_dir}/raw/*.json"))
    train_paths, test_paths = paths[: int(len(paths) * split)], paths[int(len(paths) * split) :]
    with ProcessPoolExecutor() as executor:
        executor.map(partial(_process_file, local_dir=local_dir, split="train"), enumerate(train_paths))
        executor.map(partial(_process_file, local_dir=local_dir, split="test"), enumerate(test_paths))


if __name__ == "__main__":
    fire.Fire(main)
