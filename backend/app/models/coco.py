"""COCO 80-class labels and convenience class groups for the class selector."""
from __future__ import annotations

COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

_by_name = {name: i for i, name in enumerate(COCO_CLASSES)}


def class_id(name: str) -> int:
    return _by_name[name]


# Named groups used by the class-selector quick actions.
CLASS_GROUPS: dict[str, list[str]] = {
    "people": ["person"],
    "vehicles": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
    "animals": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "indoor": [
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "bottle", "cup", "bowl",
    ],
}


def group_class_ids(group: str) -> list[int]:
    return [class_id(n) for n in CLASS_GROUPS.get(group, [])]
