import logging
import os
import typing

import fire
import torch
import torch.nn as nn
import wandb
from dataset import create_dataset
from model import Transformer
from optimizer import LRScheduler


def repeat_iterator(iterator: typing.Iterable):
    while True:
        for element in iterator:
            yield element


def create_data_loader(path: str, train: bool, seq_length: int, batch_size: int, num_workers: int = 4):
    dataset = create_dataset(path, train=train, seq_length=seq_length)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    if train:
        loader = repeat_iterator(loader)
    return loader


def train(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    logger: typing.Any,
    step: int,
    train_steps: int,
    clip_grad: float = 1.0,
    logger_steps: int = 10,
):
    for i, batch in enumerate(loader):
        logging.debug(f"Train step {i}")
        batch = batch.cuda()
        x, y = batch[:, :-1], batch[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        if i % logger_steps == 0:
            logger.log({"train/loss": loss.item()}, step=step + i)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()
        scheduler.step()
        if i == train_steps - 1:
            break


def test(model: nn.Module, loader: torch.utils.data.DataLoader, logger: typing.Any, step: int):
    total_loss = 0
    count = 0
    for i, batch in enumerate(iter(loader)):
        logging.debug(f"Test step {i}")
        with torch.no_grad():
            batch = batch.cuda()
            x, y = batch[:, :-1], batch[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            total_loss += loss.item() * x.size(0)
            count += x.size(0)
    total_loss /= count
    logger.log({"test/loss": total_loss}, step=step)


def save_ckpt(model: nn.Module, step: int, output_dir: str = "./ckpts"):
    model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    model = model._orig_mod if hasattr(model, "_orig_mod") else model
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, f"model_{step:08d}.pt"))


def main(
    vocab_size: int = 50_304,
    seq_length: int = 128,
    max_seq_length: int = 1024,
    model_dims: int = 768,
    n_layers: int = 12,
    n_heads: int = 12,
    batch_size: int = 64,
    lr: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.999),
    weight_decay: float = 0.01,
    train_steps: int = 1_000_000,
    ckpt_steps: int = 2_000,
    warmup_steps: int = 10,
    cosine_decay: bool = False,
    compile: bool = True,
    precision: str = "high",
):
    logging.basicConfig(level=logging.INFO)

    torch.set_float32_matmul_precision(precision)

    logging.info("Creating the model")
    model = Transformer(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        model_dims=model_dims,
        n_layers=n_layers,
        n_heads=n_heads,
    )
    model = model.train().cuda()

    if compile:
        model = torch.compile(model)

    logging.info("Creating data loaders")
    train_loader = create_data_loader(
        "tinystories/tokenized/train*.pt",
        train=True,
        seq_length=seq_length,
        batch_size=batch_size,
    )
    test_loader = create_data_loader(
        "tinystories/tokenized/test*.pt",
        train=False,
        seq_length=seq_length,
        batch_size=batch_size,
    )

    logging.info("Creating the optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    scheduler = LRScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        max_decay_steps=train_steps,
        cosine_decay=cosine_decay,
    )

    logger = wandb.init(project="llm")

    logging.info("Starting training")
    for i in range(0, train_steps, ckpt_steps):
        train(model, train_loader, optimizer, scheduler, logger, step=i, train_steps=ckpt_steps)
        test(model, test_loader, logger, step=i + ckpt_steps)
        save_ckpt(model, step=i + ckpt_steps)


if __name__ == "__main__":
    fire.Fire(main)
