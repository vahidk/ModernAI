# %%
# Auxilary functions

import numpy as np
import torch
from dataset import generate_houses, generate_map, visualize_data
from matplotlib import pyplot as plt
from utils import figure_to_image, frames_to_video

# %%
# House price data

map = generate_map(1920, 1080)
houses = generate_houses(map, num_houses=20, gap=0.05)
visualize_data(map, houses)

# %%
# Pre-processing


def split_data(houses, ratio):
    idx = int(houses["price"].shape[0] * ratio)
    train = {k: v[:idx] for k, v in houses.items()}
    test = {k: v[idx:] for k, v in houses.items()}
    return train, test


def preprocess_inputs(houses):
    x = np.concatenate([houses["size"][:, None], houses["rooms"][:, None], houses["location"]], axis=-1).astype(
        np.float32
    )
    x = (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)
    return torch.from_numpy(x)


def preprocess_outputs(prices):
    y = prices[:, None].astype(np.float32)
    y = y / 1000.0
    return torch.from_numpy(y)


def preprocess_data(houses):
    return preprocess_inputs(houses), preprocess_outputs(houses["price"])


houses = generate_houses(map, num_houses=10_000)
train, test = split_data(houses, 0.75)
x_train, y_train = preprocess_data(train)
x_test, y_test = preprocess_data(test)
print("Input shape:", x_train.shape)
print("Target shape:", y_train.shape)


# %%
# Constant model

model = torch.mean(y_train)
print("Constant model:", model.item())

loss = (y_train - model).abs().mean()
print(f"Training loss: {loss:.03f}")

loss = (y_test - model).abs().mean()
print(f"Test loss: {loss:.03f}")


# %%


def compute_error(b):
    loss = abs(b - y_train).abs().mean()
    return loss


# plot the loss as a function of constant estimate b
plt.figure(figsize=(16, 9))
ws = torch.linspace(0, 1, 1000)
median = torch.median(y_train)
plt.plot(ws, [compute_error(w) for w in ws])
plt.plot(median, compute_error(median), ".", color="green")
plt.annotate(
    f"Median: {median:.03f}\nError: {compute_error(median):.03f}",
    xy=(median, compute_error(median)),
    xytext=(median, compute_error(median) + 0.05),
    arrowprops=dict(arrowstyle="->", color="green"),
    horizontalalignment="center",
)
plt.xlabel("w")
plt.ylabel("L")
plt.ylim(0, 0.6)
plt.show()

# %%
# Gradient descent


class ConstantModel(torch.nn.Module):
    def __init__(self, output_dims):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.randn([output_dims]))

    def forward(self):
        return self.bias


torch.manual_seed(21)
model = ConstantModel(1)
opt = torch.optim.SGD(model.parameters(), lr=0.001)

ws_train = []
const_ls_train = []

for i in range(1000):
    predictions = model()
    loss = (predictions - y_train).abs().mean()

    ws_train.append(model.bias.item())
    const_ls_train.append(loss.item())

    opt.zero_grad()
    loss.backward()
    opt.step()

    if (i + 1) % 100 == 0:
        predictions = model()
        test_error = (predictions - y_test).abs().mean()
        print(f"Step {i + 1}, training loss: {loss.item():.3f}, test loss: {test_error.item():.3f}")

# Create animation.
frames = []
ws = torch.linspace(0, 1, 1000)
ls = [compute_error(w) for w in ws]
for i in range(1000):
    plt.figure(figsize=(16, 9))
    plt.plot(ws, ls, "-")
    plt.plot(ws_train[:i], const_ls_train[:i], "-", color="r", lw=3)
    plt.xlabel("w")
    plt.ylabel("L")
    plt.ylim(0, 0.6)
    frames.append(figure_to_image(plt.gcf()))
    plt.close()

frames_to_video("./videos/gradient_descent.mp4", frames)

# %%
# Momentum

torch.manual_seed(21)
model = ConstantModel(1)
opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

ws_train = []
const_ls_train = []

for i in range(1000):
    predictions = model()
    loss = (predictions - y_train).abs().mean()

    ws_train.append(model.bias.item())
    const_ls_train.append(loss.item())

    opt.zero_grad()
    loss.backward()
    opt.step()

    if (i + 1) % 100 == 0:
        predictions = model()
        test_error = (predictions - y_test).abs().mean()
        print(f"Step {i + 1}, training loss: {loss.item():.3f}, test loss: {test_error.item():.3f}")

# Create animation.
frames = []
ws = torch.linspace(0, 1, 1000)
ls = [compute_error(w) for w in ws]
for i in range(1000):
    plt.figure(figsize=(16, 9))
    plt.plot(ws, ls, "-")
    plt.plot(ws_train[:i], const_ls_train[:i], "-", color="r", lw=3)
    plt.xlabel("w")
    plt.ylabel("L")
    plt.ylim(0, 0.6)
    frames.append(figure_to_image(plt.gcf()))
    plt.close()

frames_to_video("./videos/momentum.mp4", frames)

# %%
# Stochastic gradient descent

torch.manual_seed(21)
model = ConstantModel(1)
opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
batch_size = 1000

ws_train = []
const_ls_train = []

for i in range(1000):
    for x_batch, y_batch in zip(x_train.split(batch_size), y_train.split(batch_size)):
        loss = (model() - y_batch).abs().mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    loss = (model() - y_train).abs().mean()
    ws_train.append(model.bias.item())
    const_ls_train.append(loss.item())

    if (i + 1) % 100 == 0:
        test_error = (model() - y_test).abs().mean()
        print(f"Epoch {i + 1}, training loss: {loss.item():.3f}, test loss: {test_error.item():.3f}")

# %%
# Create animation.
frames = []
ws = torch.linspace(0, 1, 1000)
ls = [compute_error(w) for w in ws]
for i in range(1000):
    plt.figure(figsize=(16, 9))
    plt.plot(ws, ls, "-")
    plt.plot(ws_train[:i], const_ls_train[:i], "-", color="r", lw=3)
    plt.xlabel("w")
    plt.ylabel("L")
    plt.ylim(0, 0.6)
    frames.append(figure_to_image(plt.gcf()))
    plt.close()

frames_to_video("./videos/sgd.mp4", frames)

# %%
# Using a linear model

torch.manual_seed(2)
model = torch.nn.Linear(4, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

linear_ls_train = []

for i in range(1000):
    for x_batch, y_batch in zip(x_train.split(batch_size), y_train.split(batch_size)):
        predictions = model(x_batch)
        loss = (predictions - y_batch).abs().mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    loss = (model(x_train) - y_train).abs().mean()
    linear_ls_train.append(loss.item())

    if (i + 1) % 100 == 0:
        predictions = model(x_test)
        test_error = (predictions - y_test).abs().mean()
        print(f"Epoch {i + 1}, training loss: {loss.item():.3f}, test loss: {test_error.item():.3f}")

# Create animation.
frames = []
for i in range(100):
    plt.figure(figsize=(16, 9))
    plt.plot(const_ls_train[:i], "-", color="r", label="Constant model")
    plt.plot(linear_ls_train[:i], "-", color="b", label="Linear model")
    plt.xlabel("step")
    plt.ylabel("L")
    plt.xlim(0, 100)
    plt.ylim(0, 0.6)
    plt.legend()
    # legend
    frames.append(figure_to_image(plt.gcf()))
    plt.close()

frames_to_video("./videos/linear_model.mp4", frames)

# %%
# MLP


class MLP(torch.nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dims, hidden_dims)
        self.layer2 = torch.nn.Linear(hidden_dims, output_dims)

    def forward(self, x):
        h = torch.relu(self.layer1(x))
        return self.layer2(h)


torch.manual_seed(5)
model = MLP(4, 100, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
batch_size = 100

mlp_ls_train = []

for i in range(1000):
    for x_batch, y_batch in zip(x_train.split(batch_size), y_train.split(batch_size)):
        predictions = model(x_batch)
        loss = (predictions - y_batch).abs().mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    loss = (model(x_train) - y_train).abs().mean()
    mlp_ls_train.append(loss.item())

    if (i + 1) % 100 == 0:
        predictions = model(x_test)
        test_error = (predictions - y_test).abs().mean()
        print(f"Epoch {i + 1}, training loss: {loss.item():.3f}, test loss: {test_error.item():.3f}")

# Create animation.
frames = []
for i in range(100):
    plt.figure(figsize=(16, 9))
    plt.plot(const_ls_train[:i], "-", color="r", label="Constant model")
    plt.plot(linear_ls_train[:i], "-", color="b", label="Linear model")
    plt.plot(mlp_ls_train[:i], "-", color="g", label="MLP")
    plt.xlabel("step")
    plt.ylabel("L")
    plt.xlim(0, 100)
    plt.ylim(0, 0.6)
    plt.legend()
    # legend
    frames.append(figure_to_image(plt.gcf()))
    plt.close()

frames_to_video("./videos/mlp.mp4", frames)

# %%
