import os
import torch

from stages.segmentation import (
    load_segmentation_model,
    run_segmentation
)

from stages.depth_estimation import (
    load_depth_model,
    run_depth_estimation
)

from stages.depth_ordering import (
    build_depth_order,
    split_objects_and_stuff
)

from stages.amodal_completion import (
    load_inpainting_model,
    prepare_object_crops,
    run_amodal_completion
)

from stages.alpha_matting import (
    load_matting_model,
    run_alpha_matting
)

from stages.rgba_layers import (
    build_object_rgba_layers,
    build_stuff_rgba_layers
)

from stages.background_completion import (
    complete_background
)

from stages.intrinsic_decomposition import (
    intrinsic_decomposition
)

from stages.final_layers import (
    build_final_layer_stack,
    recompose_layers,
    evaluate_recomposition
)


PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

CASE_NAMES = [
    "park",
    "outdoortent",
    "street",
    "indoor",
    "dog"
]


def get_input_image(input_folder):
    files = [
        file_name
        for file_name in os.listdir(
            input_folder
        )
        if file_name.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        )
    ]

    if len(files) == 0:
        raise FileNotFoundError(
            f"No image found in {input_folder}"
        )

    return os.path.join(
        input_folder,
        files[0]
    )


def run_case(
    case_name,
    segmentation_processor,
    segmentation_model,
    depth_processor,
    depth_model,
    inpaint_pipe,
    matting_processor,
    matting_model,
    device
):
    print()
    print(
        f"Running case: {case_name}"
    )

    case_root = os.path.join(
        PROJECT_ROOT,
        "cases",
        case_name
    )

    input_folder = os.path.join(
        case_root,
        "input"
    )

    segmentation_folder = os.path.join(
        case_root,
        "segmentation"
    )

    depth_folder = os.path.join(
        case_root,
        "depth"
    )

    ordering_folder = os.path.join(
        case_root,
        "ordering"
    )

    amodal_folder = os.path.join(
        case_root,
        "amodal"
    )

    rgba_folder = os.path.join(
        case_root,
        "rgba_layers"
    )

    intrinsic_folder = os.path.join(
        case_root,
        "intrinsic"
    )

    evaluation_folder = os.path.join(
        case_root,
        "evaluation"
    )

    final_layers_folder = os.path.join(
        case_root,
        "final_layers"
    )

    for folder in [
        segmentation_folder,
        depth_folder,
        ordering_folder,
        amodal_folder,
        rgba_folder,
        intrinsic_folder,
        evaluation_folder,
        final_layers_folder
    ]:
        os.makedirs(
            folder,
            exist_ok=True
        )

    image_path = get_input_image(
        input_folder
    )

    print(
        "1. Segmentation"
    )

    run_segmentation(
        image_path,
        segmentation_folder,
        segmentation_processor,
        segmentation_model,
        device
    )

    print(
        "2. Depth estimation"
    )

    run_depth_estimation(
        image_path,
        depth_folder,
        depth_processor,
        depth_model,
        device
    )

    print(
        "3. Depth ordering"
    )

    depth_order = build_depth_order(
        segmentation_folder,
        depth_folder,
        ordering_folder
    )

    split_objects_and_stuff(
        depth_order,
        ordering_folder
    )

    print(
        "4. Preparing object crops"
    )

    prepare_object_crops(
        image_path,
        segmentation_folder,
        ordering_folder,
        amodal_folder
    )

    print(
        "5. Amodal completion"
    )

    run_amodal_completion(
        amodal_folder,
        inpaint_pipe,
        segmentation_processor,
        segmentation_model,
        device
    )

    print(
        "6. Alpha matting"
    )

    run_alpha_matting(
        amodal_folder,
        rgba_folder,
        matting_processor,
        matting_model,
        device
    )

    print(
        "7. RGBA object layers"
    )

    build_object_rgba_layers(
        image_path,
        segmentation_folder,
        amodal_folder,
        rgba_folder
    )

    print(
        "8. RGBA stuff layers"
    )

    build_stuff_rgba_layers(
        image_path,
        segmentation_folder,
        ordering_folder,
        rgba_folder
    )

    print(
        "9. Background completion"
    )

    complete_background(
        image_path,
        segmentation_folder,
        ordering_folder,
        rgba_folder,
        inpaint_pipe,
        device
    )

    print(
        "10. Intrinsic decomposition"
    )

    intrinsic_decomposition(
        rgba_folder,
        intrinsic_folder
    )

    print(
        "11. Building final layers"
    )

    build_final_layer_stack(
        rgba_folder,
        ordering_folder,
        final_layers_folder
    )

    recomposed_path = os.path.join(
        evaluation_folder,
        "recomposed.png"
    )

    print(
        "12. Recomposing layers"
    )

    recompose_layers(
        final_layers_folder,
        recomposed_path
    )

    metrics_path = os.path.join(
        evaluation_folder,
        "recomposition_metrics.json"
    )

    metrics = evaluate_recomposition(
        image_path,
        recomposed_path,
        metrics_path
    )

    print(
        f"Completed {case_name}"
    )

    print(
        f'PSNR: {metrics["psnr"]:.2f} dB'
    )

    print(
        f'SSIM: {metrics["ssim"]:.4f}'
    )


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Using device:",
        device
    )

    segmentation_processor, segmentation_model = (
        load_segmentation_model(
            device
        )
    )

    depth_processor, depth_model = (
        load_depth_model(
            device
        )
    )

    inpaint_pipe = (
        load_inpainting_model(
            device
        )
    )

    matting_processor, matting_model = (
        load_matting_model(
            device
        )
    )

    for case_name in CASE_NAMES:
        run_case(
            case_name,
            segmentation_processor,
            segmentation_model,
            depth_processor,
            depth_model,
            inpaint_pipe,
            matting_processor,
            matting_model,
            device
        )

    print()
    print(
        "All cases completed."
    )


if __name__ == "__main__":
    main()
