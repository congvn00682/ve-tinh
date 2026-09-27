"""Experiment constants shared by data, training, and inference code."""

CLASS_NAMES = [
    "airport",
    "baseball_diamond",
    "beach",
    "bridge",
    "church",
    "commercial_area",
    "dense_residential",
    "desert",
    "forest",
]

CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

