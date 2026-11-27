import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import VitMatteImageProcessor, VitMatteForImageMatting


MODEL_NAME = "hustvl/vitmatte-small-composition-1k"


def load_matting_model(device="cuda"):

    processor = VitMatteImageProcessor.from_pretrained(
        MODEL_NAME
    )

    model = VitMatteForImageMatting.from_pretrained(
        MODEL_NAME
    ).to(device)

    model.eval()

    return processor, model


def run_alpha_matting(
    amodal_folder,
    rgba_folder,
    processor,
    model,
    device="cuda"
):

    os.makedirs(
        rgba_folder,
        exist_ok=True
    )

    masks_folder = os.path.join(
        amodal_folder,
        "masks"
    )

    objects_folder = os.path.join(
        amodal_folder,
        "completed_objects"
    )

    with open(
        os.path.join(
            amodal_folder,
            "validated_amodal_results.json"
        ),
        "r"
    ) as f:
        objects = json.load(f)

    for obj in objects:

        object_name = obj["object_name"]

        if obj["status"] == "accepted":

            image_path = os.path.join(
                objects_folder,
                f"{object_name}_completed.png"
            )

            mask_path = os.path.join(
                masks_folder,
                f"{object_name}_final.png"
            )

        else:

            image_path = os.path.join(
                objects_folder,
                f"{object_name}_crop.png"
            )

            mask_path = os.path.join(
                masks_folder,
                f"{object_name}_visible.png"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = np.array(
            Image.open(
                mask_path
            ).convert("L")
        ) > 0

        kernel = np.ones(
            (9, 9),
            np.uint8
        )

        foreground = cv2.erode(
            mask.astype(np.uint8),
            kernel,
            iterations=1
        ) > 0

        expanded = cv2.dilate(
            mask.astype(np.uint8),
            kernel,
            iterations=1
        ) > 0

        trimap = np.zeros(
            mask.shape,
            dtype=np.uint8
        )

        trimap[expanded] = 128
        trimap[foreground] = 255

        trimap_image = Image.fromarray(
            trimap
        )

        inputs = processor(
            images=image,
            trimaps=trimap_image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = model(
                **inputs
            )

        alpha = (
            outputs.alphas
            .squeeze()
            .cpu()
            .numpy()
        )

        alpha = np.clip(
            alpha,
            0.0,
            1.0
        )

        alpha_image = Image.fromarray(
            (
                alpha * 255
            ).astype(np.uint8)
        )

        alpha_image.save(
            os.path.join(
                rgba_folder,
                f"{object_name}_alpha.png"
            )
        )
