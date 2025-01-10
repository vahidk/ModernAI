import numpy as np
import torch
from dataset import generate_houses, generate_map, preprocess_data, split_data, visualize_data
from matplotlib import pyplot as plt
from model import MLP


def main():
    # generate the dataset
    map = generate_map(1920, 1080)
    houses = generate_houses(map, num_houses=10_000)
    train, test = split_data(houses, 0.75)
    x_train, y_train = preprocess_data(train)
    x_test, y_test = preprocess_data(test)

    # define the model
    model = MLP(4, 100, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    batch_size = 100

    # train the model
    for i in range(1000):
        for x_batch, y_batch in zip(x_train.split(batch_size), y_train.split(batch_size)):
            predictions = model(x_batch)
            loss = (predictions - y_batch).abs().mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

        loss = (model(x_train) - y_train).abs().mean()

        if (i + 1) % 100 == 0:
            predictions = model(x_test)
            test_error = (predictions - y_test).abs().mean()
            print(f"Epoch {i + 1}, training loss: {loss.item():.3f}, test loss: {test_error.item():.3f}")


if __name__ == "__main__":
    main()
