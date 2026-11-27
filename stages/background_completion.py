import os
import cv2
import json
import torch
import numpy as np
from PIL import Image


def build_background_removal_mask(
    segmentation_folder,
    ordering_folder,
    output_path
):
    with open(
        os.path.join(
            ordering_folder,
            "object_instances.json"
        ),
        "r"
    ) as f:
        objects = json.load(f)

    masks_folder = os.path.join(
        segmentation_folder,
        "masks"
    )

    if len(objects) == 0:
        return 0

    first_mask = np.array(
        Image.open(
            os.path.join(
                masks_folder,
                objects[0]["mask_file"]
            )
        ).convert("L")
    )

    removal_mask = np.zeros_like(
        first_mask,
        dtype=np.uint8
    )

    for obj in objects:

        mask = np.array(
            Image.open(
                os.path.join(
                    masks_folder,
                    obj["mask_file"]
                )
            ).convert("L")
        ) > 0

        removal_mask[
            mask
        ] = 255

    Image.fromarray(
        removal_mask
    ).save(
        output_path
    )

    return int(
        (removal_mask > 0).sum()
    )


def complete_background(
    image_path,
    segmentation_folder,
    ordering_folder,
    rgba_folder,
    inpaint_pipe,
    device="cuda"
):
    os.makedirs(
        rgba_folder,
        exist_ok=True
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    removal_mask_path = os.path.join(
        rgba_folder,
        "background_removal_mask.png"
    )

    removed_pixels = build_background_removal_mask(
        segmentation_folder,
        ordering_folder,
        removal_mask_path
    )

    background_path = os.path.join(
        rgba_folder,
        "background_completed.png"
    )

    if removed_pixels == 0:

        image.save(
            background_path
        )

        return background_path

    mask = np.array(
        Image.open(
            removal_mask_path
        ).convert("L")
    ) > 0

    kernel = np.ones(
        (15, 15),
        np.uint8
    )

    mask = cv2.dilate(
        mask.astype(np.uint8),
        kernel,
        iterations=1
    ) > 0

    mask_image = Image.fromarray(
        mask.astype(
            np.uint8
        ) * 255
    )

    width, height = image.size

    max_dimension = 1024

    scale = min(
        1.0,
        max_dimension / max(
            width,
            height
        )
    )

    resized_width = int(
        width * scale
    )

    resized_height = int(
        height * scale
    )

    resized_width = max(
        8,
        resized_width
        - resized_width % 8
    )

    resized_height = max(
        8,
        resized_height
        - resized_height % 8
    )

    resized_image = image.resize(
        (
            resized_width,
            resized_height
        ),
        Image.Resampling.LANCZOS
    )

    resized_mask = mask_image.resize(
        (
            resized_width,
            resized_height
        ),
        Image.Resampling.NEAREST
    )

    prompt = (
        "realistic empty background scene, "
        "continue the surrounding environment naturally, "
        "no people, no vehicles, no foreground objects"
    )

    generator = torch.Generator(
        device=device
    ).manual_seed(
        2000
    )

    with torch.no_grad():

        result = inpaint_pipe(
            prompt=prompt,
            image=resized_image,
            mask_image=resized_mask,
            height=resized_height,
            width=resized_width,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator
        ).images[0]

    completed = result.resize(
        image.size,
        Image.Resampling.LANCZOS
    )

    completed.save(
        background_path
    )

    return background_path
