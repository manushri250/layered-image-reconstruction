import os
import cv2
import numpy as np
from PIL import Image


def intrinsic_decomposition(
    rgba_folder,
    intrinsic_folder
):
    albedo_folder = os.path.join(
        intrinsic_folder,
        "albedo"
    )

    shading_folder = os.path.join(
        intrinsic_folder,
        "shading"
    )

    os.makedirs(
        albedo_folder,
        exist_ok=True
    )

    os.makedirs(
        shading_folder,
        exist_ok=True
    )

    layer_files = sorted(
        [
            file_name
            for file_name in os.listdir(
                rgba_folder
            )
            if file_name.endswith(
                "_rgba.png"
            )
        ]
    )

    for file_name in layer_files:

        layer_path = os.path.join(
            rgba_folder,
            file_name
        )

        layer = Image.open(
            layer_path
        ).convert("RGBA")

        layer_array = np.array(
            layer
        ).astype(
            np.float32
        ) / 255.0

        rgb = layer_array[
            :, :, :3
        ]

        alpha = layer_array[
            :, :, 3
        ]

        luminance = (
            0.299 * rgb[:, :, 0]
            +
            0.587 * rgb[:, :, 1]
            +
            0.114 * rgb[:, :, 2]
        )

        shading = cv2.GaussianBlur(
            luminance,
            (0, 0),
            sigmaX=5,
            sigmaY=5
        )

        shading = np.clip(
            shading,
            0.05,
            1.0
        )

        albedo = rgb / shading[
            :, :,
            None
        ]

        albedo = np.clip(
            albedo,
            0.0,
            1.0
        )

        albedo_rgba = np.dstack(
            (
                albedo,
                alpha
            )
        )

        shading_rgb = np.repeat(
            shading[:, :, None],
            3,
            axis=2
        )

        shading_rgba = np.dstack(
            (
                shading_rgb,
                alpha
            )
        )

        base_name = file_name.replace(
            "_rgba.png",
            ""
        )

        albedo_image = Image.fromarray(
            (
                albedo_rgba * 255
            ).astype(
                np.uint8
            ),
            mode="RGBA"
        )

        shading_image = Image.fromarray(
            (
                shading_rgba * 255
            ).astype(
                np.uint8
            ),
            mode="RGBA"
        )

        albedo_image.save(
            os.path.join(
                albedo_folder,
                f"{base_name}_albedo.png"
            )
        )

        shading_image.save(
            os.path.join(
                shading_folder,
                f"{base_name}_shading.png"
            )
        )
