import numpy as np
import torch
from matplotlib import pyplot as plt


def generate_map(
    width,
    height,
    num_landmarks=3,
    landmark_radius=200,
    seed=0,
):
    # generate random landmark locations.
    rng = np.random.default_rng(seed)
    landmarks = np.stack(
        [
            rng.integers(0, width, size=(num_landmarks)),
            rng.integers(0, height, size=(num_landmarks)),
        ],
        axis=1,
    )

    return {
        "width": width,
        "height": height,
        "landmarks": landmarks,
        "landmark_radius": landmark_radius,
    }


def ground_truth_model(map, houses):
    landmarks = map["landmarks"]
    landmark_radius = map["landmark_radius"]
    locations = houses["location"]
    sizes = houses["size"]
    rooms = houses["rooms"]

    # Compute distance to the closest landmarks for each house.
    dist_to_landmarks = np.linalg.norm(locations[:, None, :] - landmarks[None, :, :], axis=-1).min(axis=-1)

    # generate house prices according to the size, number of rooms, distance to landmarks, and distance to ocean.
    prices = (sizes * 0.2 + rooms * 20 + (dist_to_landmarks < landmark_radius).astype(np.float32) * 100).astype(
        np.int32
    )

    return prices


def generate_houses(map, num_houses=20, margin=0.05, gap=None, seed=0):
    width = map["width"]
    height = map["height"]
    landmarks = map["landmarks"]

    # iteratively generate house locations that are not too close to each other.
    rng = np.random.default_rng(seed)
    locations = np.zeros((0, 2))
    while len(locations) < num_houses:
        new_location = np.array(
            [
                [
                    rng.integers(margin * width, width - margin * width, size=()),
                    rng.integers(margin * height, (1 - margin) * height, size=()),
                ]
            ]
        )

        if gap:
            if len(locations) > 0 and np.linalg.norm(locations - new_location, axis=-1).min() < gap * width:
                continue
            if len(landmarks) > 0 and np.linalg.norm(landmarks - new_location, axis=-1).min() < gap * width:
                continue

        locations = np.concatenate([locations, new_location], axis=0)

    # generate random house sizes.
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1000, 3000, size=num_houses)

    # generate number of rooms proportional to the size with some noise.
    rng = np.random.default_rng(seed)
    rooms = np.clip((sizes / 800).astype(int) + rng.integers(-1, 2, size=(num_houses)), 1, 5)

    houses = {
        "location": locations,
        "size": sizes,
        "rooms": rooms,
    }

    # compute house prices.
    rng = np.random.default_rng(seed)
    unknown_factors = rng.integers(-5, 6, size=num_houses)
    houses["price"] = ground_truth_model(map, houses) + unknown_factors

    return houses


def visualize_data(
    map,
    houses,
    show_price=True,
    show_size=True,
    show_rooms=True,
    show_landmarks=False,
    errors=None,
):
    plt.figure(figsize=(16, 9), dpi=300)
    if show_landmarks:
        for x, y in map["landmarks"]:
            plt.gca().add_artist(plt.Circle((x, y), map["landmark_radius"], color="red", alpha=0.1))
        plt.scatter(map["landmarks"][:, 0], map["landmarks"][:, 1], marker="*", s=200, c="red")
    if errors is not None:
        cs = errors.detach().numpy()
    else:
        cs = "black"
    plt.scatter(
        houses["location"][:, 0],
        houses["location"][:, 1],
        marker="^",
        s=houses["size"][:] * 0.1,
        c=cs,
    )
    for i, (x, y) in enumerate(houses["location"]):
        if show_price:
            plt.text(
                x + 20,
                y - 15,
                f"${houses['price'][i]}K",
                fontsize=10,
                color="darkgreen",
            )
        if show_size:
            plt.text(
                x + 20,
                y + 10,
                f"{houses['size'][i]} sqft",
                fontsize=10,
                color="darkblue",
            )
        if show_rooms:
            plt.text(
                x + 20,
                y + 35,
                f"{houses['rooms'][i]} rooms",
                fontsize=10,
                color="darkblue",
            )
        if errors is not None:
            plt.text(x + 20, y + 60, f"ε = {errors[i]:.2f}", fontsize=10, color="darkblue")
    plt.axis("off")
    plt.xlim(0, map["width"])
    plt.ylim(map["height"], 0)
    plt.tight_layout()
    plt.show()


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
