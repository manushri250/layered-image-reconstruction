import os
import json
import math
import shutil
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def build_final_layer_stack(
    rgba_folder,
    ordering_folder,
    final_layers_folder
):
    os.makedirs(
        final_layers_folder,
        exist_ok=True
    )

    for file_name in os.listdir(
        final_layers_folder
    ):
        file_path = os.path.join(
            final_layers_folder,
            file_name
        )

        if os.path.isfile(
            file_path
        ):
            os.remove(
                file_path
            )

    background_source = os.path.join(
        rgba_folder,
        "background_completed.png"
    )

    background_target = os.path.join(
        final_layers_folder,
        "layer_00_background.png"
    )

    shutil.copy(
        background_source,
        background_target
    )

    with open(
        os.path.join(
            ordering_folder,
            "depth_order.json"
        ),
        "r"
    ) as f:
        depth_order = json.load(f)

    far_to_near = list(
        reversed(
            depth_order
        )
    )

    metadata = [
        {
            "layer_index": 0,
            "class": "background",
            "depth_rank": None,
            "file": "layer_00_background.png"
        }
    ]

    layer_index = 1

    for segment in far_to_near:

        safe_class = (
            segment["class"]
            .replace(" ", "_")
            .replace("/", "_")
        )

        source_name = (
            f'{segment["segment_id"]:02d}_'
            f'{safe_class}_rgba.png'
        )

        source_path = os.path.join(
            rgba_folder,
            source_name
        )

        if not os.path.exists(
            source_path
        ):
            continue

        output_name = (
            f"layer_{layer_index:02d}_"
            f"{safe_class}.png"
        )

        shutil.copy(
            source_path,
            os.path.join(
                final_layers_folder,
                output_name
            )
        )

        metadata.append(
            {
                "layer_index": layer_index,
                "segment_id": segment["segment_id"],
                "class": segment["class"],
                "depth_rank": segment["depth_rank"],
                "median_depth": segment["median_depth"],
                "file": output_name
            }
        )

        layer_index += 1

    with open(
        os.path.join(
            final_layers_folder,
            "layers_metadata.json"
        ),
        "w"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    return metadata


def recompose_layers(
    final_layers_folder,
    output_path
):
    layer_files = sorted(
        [
            file_name
            for file_name in os.listdir(
                final_layers_folder
            )
            if file_name.startswith(
                "layer_"
            )
            and file_name.endswith(
                ".png"
            )
        ]
    )

    base = Image.open(
        os.path.join(
            final_layers_folder,
            layer_files[0]
        )
    ).convert("RGBA")

    for file_name in layer_files[1:]:

        layer = Image.open(
            os.path.join(
                final_layers_folder,
                file_name
            )
        ).convert("RGBA")

        base = Image.alpha_composite(
            base,
            layer
        )

    base.convert(
        "RGB"
    ).save(
        output_path
    )

    return base.convert(
        "RGB"
    )


def evaluate_recomposition(
    original_path,
    recomposed_path,
    output_json
):
    original = np.array(
        Image.open(
            original_path
        ).convert("RGB")
    ).astype(
        np.float32
    ) / 255.0

    recomposed = np.array(
        Image.open(
            recomposed_path
        ).convert("RGB")
    ).astype(
        np.float32
    ) / 255.0

    mse = np.mean(
        (
            original
            -
            recomposed
        ) ** 2
    )

    if mse == 0:
        psnr_value = float(
            "inf"
        )
    else:
        psnr_value = (
            10
            *
            math.log10(
                1.0 / mse
            )
        )

    ssim_value = ssim(
        original,
        recomposed,
        channel_axis=2,
        data_range=1.0
    )

    metrics = {
        "psnr": float(
            psnr_value
        ),
        "ssim": float(
            ssim_value
        )
    }

    with open(
        output_json,
        "w"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2
        )

    return metrics
