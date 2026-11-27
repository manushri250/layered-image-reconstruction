import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


INPAINT_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"


def load_inpainting_model(device="cuda"):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        INPAINT_MODEL,
        torch_dtype=torch.float16
    ).to(device)

    pipe.enable_attention_slicing()

    return pipe


def prepare_object_crops(
    image_path,
    segmentation_folder,
    ordering_folder,
    amodal_folder
):
    with open(
        os.path.join(
            ordering_folder,
            "object_instances.json"
        ),
        "r"
    ) as f:
        objects = json.load(f)

    image = Image.open(
        image_path
    ).convert("RGB")

    image_array = np.array(image)

    masks_folder = os.path.join(
        segmentation_folder,
        "masks"
    )

    output_masks = os.path.join(
        amodal_folder,
        "masks"
    )

    output_objects = os.path.join(
        amodal_folder,
        "completed_objects"
    )

    os.makedirs(
        output_masks,
        exist_ok=True
    )

    os.makedirs(
        output_objects,
        exist_ok=True
    )

    crop_data = []

    for obj in objects:

        mask = np.array(
            Image.open(
                os.path.join(
                    masks_folder,
                    obj["mask_file"]
                )
            ).convert("L")
        ) > 0

        ys, xs = np.where(mask)

        if len(xs) == 0:
            continue

        x1 = xs.min()
        x2 = xs.max()
        y1 = ys.min()
        y2 = ys.max()

        object_width = x2 - x1 + 1
        object_height = y2 - y1 + 1

        margin = max(
            20,
            int(
                0.5 * max(
                    object_width,
                    object_height
                )
            )
        )

        crop_x1 = max(
            0,
            x1 - margin
        )

        crop_y1 = max(
            0,
            y1 - margin
        )

        crop_x2 = min(
            image.width,
            x2 + margin + 1
        )

        crop_y2 = min(
            image.height,
            y2 + margin + 1
        )

        image_crop = image_array[
            crop_y1:crop_y2,
            crop_x1:crop_x2
        ]

        mask_crop = mask[
            crop_y1:crop_y2,
            crop_x1:crop_x2
        ]

        safe_class = (
            obj["class"]
            .replace(" ", "_")
            .replace("/", "_")
        )

        object_name = (
            f'{obj["segment_id"]:02d}_'
            f'{safe_class}'
        )

        crop_path = os.path.join(
            output_objects,
            f"{object_name}_crop.png"
        )

        visible_path = os.path.join(
            output_masks,
            f"{object_name}_visible.png"
        )

        Image.fromarray(
            image_crop
        ).save(
            crop_path
        )

        Image.fromarray(
            mask_crop.astype(np.uint8) * 255
        ).save(
            visible_path
        )

        crop_data.append(
            {
                "segment_id": obj["segment_id"],
                "class": obj["class"],
                "object_name": object_name,
                "crop_x1": int(crop_x1),
                "crop_y1": int(crop_y1),
                "crop_x2": int(crop_x2),
                "crop_y2": int(crop_y2)
            }
        )

    with open(
        os.path.join(
            amodal_folder,
            "crop_metadata.json"
        ),
        "w"
    ) as f:
        json.dump(
            crop_data,
            f,
            indent=2
        )

    return crop_data


def run_amodal_completion(
    amodal_folder,
    inpaint_pipe,
    segmentation_processor,
    segmentation_model,
    device="cuda"
):
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
        objects = json.load(f)

    results = []

    for obj in objects:

        object_name = obj["object_name"]
        target_class = obj["class"]

        crop_path = os.path.join(
            objects_folder,
            f"{object_name}_crop.png"
        )

        visible_path = os.path.join(
            masks_folder,
            f"{object_name}_visible.png"
        )

        image = Image.open(
            crop_path
        ).convert("RGB")

        visible_mask = np.array(
            Image.open(
                visible_path
            ).convert("L")
        ) > 0

        ys, xs = np.where(
            visible_mask
        )

        object_width = (
            xs.max() - xs.min() + 1
        )

        object_height = (
            ys.max() - ys.min() + 1
        )

        kernel_size = max(
            15,
            int(
                0.1 * max(
                    object_width,
                    object_height
                )
            )
        )

        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = np.ones(
            (
                kernel_size,
                kernel_size
            ),
            np.uint8
        )

        dilated = cv2.dilate(
            visible_mask.astype(np.uint8),
            kernel,
            iterations=1
        ) > 0

        completion_region = (
            dilated
            &
            ~visible_mask
        )

        completion_mask = Image.fromarray(
            completion_region.astype(
                np.uint8
            ) * 255
        )

        width, height = image.size

        padded_width = (
            (width + 7) // 8
        ) * 8

        padded_height = (
            (height + 7) // 8
        ) * 8

        padded_image = Image.new(
            "RGB",
            (
                padded_width,
                padded_height
            )
        )

        padded_image.paste(
            image,
            (0, 0)
        )

        padded_mask = Image.new(
            "L",
            (
                padded_width,
                padded_height
            )
        )

        padded_mask.paste(
            completion_mask,
            (0, 0)
        )

        prompt = (
            f"complete the full {target_class} naturally, "
            f"same {target_class}, realistic photograph"
        )

        generator = torch.Generator(
            device=device
        ).manual_seed(
            1000 + int(
                obj["segment_id"]
            )
        )

        with torch.no_grad():

            completed = inpaint_pipe(
                prompt=prompt,
                image=padded_image,
                mask_image=padded_mask,
                height=padded_height,
                width=padded_width,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator
            ).images[0]

        completed = completed.crop(
            (
                0,
                0,
                width,
                height
            )
        )

        completed_path = os.path.join(
            objects_folder,
            f"{object_name}_completed.png"
        )

        completed.save(
            completed_path
        )

        inputs = segmentation_processor(
            images=completed,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = segmentation_model(
                **inputs
            )

        prediction = (
            segmentation_processor
            .post_process_panoptic_segmentation(
                outputs,
                target_sizes=[
                    (
                        height,
                        width
                    )
                ]
            )[0]
        )

        segmentation = (
            prediction["segmentation"]
            .cpu()
            .numpy()
        )

        best_mask = None
        best_overlap = 0.0

        for segment in prediction[
            "segments_info"
        ]:

            label_id = int(
                segment["label_id"]
            )

            class_name = (
                segmentation_model
                .config
                .id2label
                .get(
                    label_id,
                    str(label_id)
                )
            )

            if class_name != target_class:
                continue

            predicted_mask = (
                segmentation
                ==
                int(
                    segment["id"]
                )
            )

            overlap = (
                np.logical_and(
                    predicted_mask,
                    visible_mask
                ).sum()
                /
                max(
                    visible_mask.sum(),
                    1
                )
            )

            if overlap > best_overlap:

                best_overlap = overlap
                best_mask = predicted_mask

        if best_mask is None:

            raw_amodal = (
                visible_mask.copy()
            )

        else:

            raw_amodal = np.logical_or(
                visible_mask,
                best_mask
            )

        raw_path = os.path.join(
            masks_folder,
            f"{object_name}_amodal.png"
        )

        Image.fromarray(
            raw_amodal.astype(
                np.uint8
            ) * 255
        ).save(
            raw_path
        )

        added_pixels = int(
            np.logical_and(
                raw_amodal,
                ~visible_mask
            ).sum()
        )

        growth_ratio = (
            added_pixels
            /
            max(
                visible_mask.sum(),
                1
            )
        )

        completed_array = np.array(
            completed
        )

        black_ratio = np.mean(
            np.all(
                completed_array < 5,
                axis=2
            )
        )

        black_output = (
            black_ratio > 0.95
        )

        accepted = (
            best_overlap >= 0.50
            and
            growth_ratio <= 0.50
            and
            not black_output
        )

        if accepted:
            final_mask = raw_amodal
            status = "accepted"
        else:
            final_mask = visible_mask
            status = "fallback_visible"

        final_path = os.path.join(
            masks_folder,
            f"{object_name}_final.png"
        )

        Image.fromarray(
            final_mask.astype(
                np.uint8
            ) * 255
        ).save(
            final_path
        )

        results.append(
            {
                "segment_id": obj["segment_id"],
                "class": target_class,
                "object_name": object_name,
                "overlap_score": float(
                    best_overlap
                ),
                "growth_ratio": float(
                    growth_ratio
                ),
                "black_output": bool(
                    black_output
                ),
                "status": status,
                "final_mask": os.path.basename(
                    final_path
                )
            }
        )

    with open(
        os.path.join(
            amodal_folder,
            "validated_amodal_results.json"
        ),
        "w"
    ) as f:
        json.dump(
            results,
            f,
            indent=2
        )

    return results
