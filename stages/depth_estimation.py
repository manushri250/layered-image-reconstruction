import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"


def load_depth_model(device="cuda"):
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    model = AutoModelForDepthEstimation.from_pretrained(
        MODEL_NAME
    ).to(device)

    model.eval()

    return processor, model


def run_depth_estimation(
    image_path,
    output_folder,
    processor,
    model,
    device="cuda"
):
    os.makedirs(
        output_folder,
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

    predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False
    )

    depth_map = (
        prediction
        .squeeze()
        .cpu()
        .numpy()
    )

    np.save(
        os.path.join(
            output_folder,
            "depth_map.npy"
        ),
        depth_map
    )

    depth_min = depth_map.min()
    depth_max = depth_map.max()

    preview = (
        (depth_map - depth_min)
        /
        (depth_max - depth_min + 1e-8)
        * 255
    ).astype(np.uint8)

    Image.fromarray(preview).save(
        os.path.join(
            output_folder,
            "depth_preview.png"
        )
    )

    return depth_map
