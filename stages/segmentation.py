import os
import json
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


MODEL_NAME = "facebook/mask2former-swin-small-coco-panoptic"


def load_segmentation_model(device="cuda"):
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        MODEL_NAME
    ).to(device)

    model.eval()

    return processor, model


def run_segmentation(
    image_path,
    output_folder,
    processor,
    model,
    device="cuda"
):
    os.makedirs(output_folder, exist_ok=True)

    masks_folder = os.path.join(
        output_folder,
        "masks"
    )

    os.makedirs(
        masks_folder,
        exist_ok=True
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=[image.size[::-1]]
    )[0]

    segmentation = result["segmentation"].cpu().numpy()
    segments_info = result["segments_info"]

    label_map = model.config.id2label

    saved_segments = []

    for segment in segments_info:
        segment_id = int(segment["id"])
        label_id = int(segment["label_id"])

        class_name = label_map.get(
            label_id,
            str(label_id)
        )

        score = float(
            segment.get(
                "score",
                1.0
            )
        )

        mask = (
            segmentation == segment_id
        ).astype(np.uint8) * 255

        safe_class_name = (
            class_name
            .replace(" ", "_")
            .replace("/", "_")
        )

        mask_file = (
            f"{segment_id:02d}_"
            f"{safe_class_name}.png"
        )

        Image.fromarray(mask).save(
            os.path.join(
                masks_folder,
                mask_file
            )
        )

        saved_segments.append(
            {
                "segment_id": segment_id,
                "label_id": label_id,
                "class": class_name,
                "score": score,
                "mask_file": mask_file
            }
        )

    labels_path = os.path.join(
        output_folder,
        "labels.json"
    )

    with open(
        labels_path,
        "w"
    ) as f:
        json.dump(
            saved_segments,
            f,
            indent=2
        )

    return saved_segments
