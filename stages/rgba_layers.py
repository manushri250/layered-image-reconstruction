import os
import json
import numpy as np
from PIL import Image


def build_object_rgba_layers(
    image_path,
    segmentation_folder,
    amodal_folder,
    rgba_folder
):
    os.makedirs(
        rgba_folder,
        exist_ok=True
    )

    original_image = Image.open(
        image_path
    ).convert("RGB")

    original_array = np.array(
        original_image
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
            "crop_metadata.json"
        ),
        "r"
    ) as f:
        crop_metadata = json.load(f)

    with open(
        os.path.join(
            amodal_folder,
            "validated_amodal_results.json"
        ),
        "r"
    ) as f:
        validation_results = json.load(f)

    validation_lookup = {
        item["object_name"]: item
        for item in validation_results
    }

    for item in crop_metadata:

        object_name = item["object_name"]

        validation = validation_lookup[
            object_name
        ]

        x1 = item["crop_x1"]
        y1 = item["crop_y1"]
        x2 = item["crop_x2"]
        y2 = item["crop_y2"]

        original_crop = original_array[
            y1:y2,
            x1:x2
        ].copy()

        visible_mask = np.array(
            Image.open(
                os.path.join(
                    masks_folder,
                    f"{object_name}_visible.png"
                )
            ).convert("L")
        ) > 0

        final_mask = np.array(
            Image.open(
                os.path.join(
                    masks_folder,
                    f"{object_name}_final.png"
                )
            ).convert("L")
        ) > 0

        rgb_crop = original_crop.copy()

        if validation["status"] == "accepted":

            completed = np.array(
                Image.open(
                    os.path.join(
                        objects_folder,
                        f"{object_name}_completed.png"
                    )
                ).convert("RGB")
            )

            if completed.shape[:2] != rgb_crop.shape[:2]:

                completed = np.array(
                    Image.fromarray(
                        completed
                    ).resize(
                        (
                            rgb_crop.shape[1],
                            rgb_crop.shape[0]
                        ),
                        Image.Resampling.LANCZOS
                    )
                )

            hidden_region = (
                final_mask
                &
                ~visible_mask
            )

            rgb_crop[
                hidden_region
            ] = completed[
                hidden_region
            ]

        alpha = Image.open(
            os.path.join(
                rgba_folder,
                f"{object_name}_alpha.png"
            )
        ).convert("L")

        if alpha.size != (
            rgb_crop.shape[1],
            rgb_crop.shape[0]
        ):

            alpha = alpha.resize(
                (
                    rgb_crop.shape[1],
                    rgb_crop.shape[0]
                ),
                Image.Resampling.BILINEAR
            )

        rgba_crop = Image.fromarray(
            rgb_crop
        ).convert("RGBA")

        rgba_crop.putalpha(
            alpha
        )

        full_layer = Image.new(
            "RGBA",
            original_image.size,
            (
                0,
                0,
                0,
                0
            )
        )

        full_layer.paste(
            rgba_crop,
            (
                x1,
                y1
            )
        )

        full_layer.save(
            os.path.join(
                rgba_folder,
                f"{object_name}_rgba.png"
            )
        )


def build_stuff_rgba_layers(
    image_path,
    segmentation_folder,
    ordering_folder,
    rgba_folder
):
    original_image = Image.open(
        image_path
    ).convert("RGB")

    with open(
        os.path.join(
            ordering_folder,
            "stuff_regions.json"
        ),
        "r"
    ) as f:
        stuff_regions = json.load(f)

    masks_folder = os.path.join(
        segmentation_folder,
        "masks"
    )

    for region in stuff_regions:

        mask = Image.open(
            os.path.join(
                masks_folder,
                region["mask_file"]
            )
        ).convert("L")

        rgba = original_image.convert(
            "RGBA"
        )

        rgba.putalpha(
            mask
        )

        safe_class = (
            region["class"]
            .replace(" ", "_")
            .replace("/", "_")
        )

        layer_name = (
            f'{region["segment_id"]:02d}_'
            f'{safe_class}_rgba.png'
        )

        rgba.save(
            os.path.join(
                rgba_folder,
                layer_name
            )
        )
