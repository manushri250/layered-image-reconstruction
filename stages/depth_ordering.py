import os
import json
import numpy as np
from PIL import Image


def build_depth_order(
    segmentation_folder,
    depth_folder,
    output_folder
):
    os.makedirs(
        output_folder,
        exist_ok=True
    )

    with open(
        os.path.join(
            segmentation_folder,
            "labels.json"
        ),
        "r"
    ) as f:
        segments = json.load(f)

    depth_map = np.load(
        os.path.join(
            depth_folder,
            "depth_map.npy"
        )
    )

    masks_folder = os.path.join(
        segmentation_folder,
        "masks"
    )

    ordered_segments = []

    for segment in segments:

        mask_path = os.path.join(
            masks_folder,
            segment["mask_file"]
        )

        mask = np.array(
            Image.open(
                mask_path
            ).convert("L")
        ) > 0

        if mask.sum() == 0:
            continue

        median_depth = float(
            np.median(
                depth_map[mask]
            )
        )

        ordered_segments.append(
            {
                "segment_id": segment["segment_id"],
                "label_id": segment["label_id"],
                "class": segment["class"],
                "score": segment["score"],
                "mask_file": segment["mask_file"],
                "median_depth": median_depth
            }
        )

    ordered_segments = sorted(
        ordered_segments,
        key=lambda x: x["median_depth"],
        reverse=True
    )

    for rank, segment in enumerate(
        ordered_segments
    ):
        segment["depth_rank"] = rank

    output_path = os.path.join(
        output_folder,
        "depth_order.json"
    )

    with open(
        output_path,
        "w"
    ) as f:
        json.dump(
            ordered_segments,
            f,
            indent=2
        )

    return ordered_segments


def split_objects_and_stuff(
    depth_order,
    output_folder
):
    objects = []
    stuff = []

    for segment in depth_order:

        if segment["label_id"] < 80:
            objects.append(segment)
        else:
            stuff.append(segment)

    with open(
        os.path.join(
            output_folder,
            "object_instances.json"
        ),
        "w"
    ) as f:
        json.dump(
            objects,
            f,
            indent=2
        )

    with open(
        os.path.join(
            output_folder,
            "stuff_regions.json"
        ),
        "w"
    ) as f:
        json.dump(
            stuff,
            f,
            indent=2
        )

    return objects, stuff
