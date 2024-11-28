"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the Creative Commons Attribution 4.0 International License (see LICENSE for details)
"""
from mesonet.mask_functions import *
import os
from mesonet.utils import parse_yaml
from keras.models import load_model

#----------------------------------------预测特定区域-----------------------------------------------------------------
def predictRegion(
    input_file,
    num_images,
    model,
    output,
    mat_save,
    threshold,
    mask_generate,
    git_repo_base,
    atlas_to_brain_align,
    dlc_pts,
    atlas_pts,
    olfactory_check,
    use_unet,
    plot_landmarks,
    atlas_label_list,
    align_once,
    region_labels,
    original_label,
):
    """
    Segment brain images to predict the location of brain regions.
    :param input_file: Input folder containing brain images.
    :param num_images: Number of brain images to be analyzed.
    :param model: Prediction model (.hdf5 file) to be used.
    :param output: Overall output folder into which all files will be saved.
    :param mat_save: Choose whether or not to save each brain region contour and centre as a .mat file (for MATLAB).
    :param threshold: Threshold for segmentation algorithm.
    :param mask_generate: Choose whether or not to only generate masks of the brain contour from this function.
    :param git_repo_base: The path to the base git repository containing necessary resources for MesoNet (reference
    atlases, DeepLabCut config files, etc.)
    :param atlas_to_brain_align: If True, warp and register an atlas to the brain image; if False, warp and register a
    brain image to the atlas.
    :param dlc_pts: The landmarks for brain-atlas registration as determined by the DeepLabCut model.
    :param atlas_pts: The landmarks for brain-atlas registration from the original brain atlas.
    :param region_labels: Choose whether or not to attempt to label each region with its name from the Allen Institute
    Mouse Brain Atlas.
    :param olfactory_check: If True, draws olfactory bulb contours on the brain image.
    :param use_unet: Choose whether or not to identify the borders of the cortex using a U-net model.
    :param atlas_to_brain_align: If True, registers the atlas to each brain image. If False, registers each brain image
    to the atlas.
    :param plot_landmarks: If True, plots DeepLabCut landmarks (large circles) and original alignment landmarks (small
    circles) on final brain image.
    :param atlas_label_list: A list of aligned atlases in which each brain region is filled with a unique numeric label.
    This allows for consistent identification of brain regions across images. If original_label is True, this is an
    empty list.
    :param align_once: If True, carries out all alignments based on the alignment of the first atlas and brain. This can
    save time if you have many frames of the same brain with a fixed camera position.
    :param region_labels: choose whether to assign a name to each region based on an existing brain atlas (not currently
    implemented).
    :param original_label: If True, uses a brain region labelling approach that attempts to automatically sort brain
    regions in a consistent order (left to right by hemisphere, then top to bottom for vertically aligned regions). This
    approach may be more flexible if you're using a custom brain atlas (i.e. not one in which region is filled with a
    unique number).
    input_file: 包含脑图像的输入文件夹。
    num_images: 需要分析的脑图像数量。
    model: 要使用的预测模型（.hdf5 文件）。
    output: 所有文件将被保存的总体输出文件夹。
    mat_save: 选择是否将每个脑区轮廓和中心保存为 .mat 文件（用于 MATLAB）。
    threshold: 分割算法的阈值。
    mask_generate: 选择是否仅从该函数生成脑轮廓的掩码。
    git_repo_base: 包含 MesoNet 必要资源（参考图谱、DeepLabCut 配置文件等）的基本 Git 存储库路径。
    atlas_to_brain_align: 如果为 True，则将图谱变形并注册到脑图像；如果为 False，则将脑图像变形并注册到图谱。
    dlc_pts: 由 DeepLabCut 模型确定的脑图谱配准的地标。
    atlas_pts: 来自原始脑图谱的脑图谱配准的地标。
    region_labels: 选择是否尝试根据艾伦研究所小鼠脑图谱给每个区域命名。
    olfactory_check: 如果为 True，在脑图像上绘制嗅球轮廓。
    use_unet: 选择是否使用 U-net 模型识别皮层边界。
    plot_landmarks: 如果为 True，在最终脑图像上绘制 DeepLabCut 地标（大圆圈）和原始对齐地标（小圆圈）。
    atlas_label_list: 一个列表，其中包含每个脑区填充了唯一数字标签的对齐图谱。这允许在图像之间一致地识别脑区。如果 original_label 为 True，则这是一个空列表。
    align_once: 如果为 True，基于第一个图谱和脑的对齐执行所有对齐。如果相机位置固定且有许多同一脑的帧，这可以节省时间。
    region_labels: 选择是否根据现有的脑图谱给每个区域分配名称（目前未实现）。
    original_label: 如果为 True，使用一种试图自动按一致顺序排序脑区的脑区标记方法（从左到右按半球，然后从上到下对垂直排列的区域进行排序）。这种方法在使用自定义脑图谱时可能更灵活（即不是每个区域都填充了唯一数字的图谱）
    """
    # Create and define save folders for each output of the prediction
    # Output folder for basic mask (used later in prediction)
    output_mask_path = os.path.join(output, "output_mask")
    # Output folder for transparent masks and masks overlaid onto brain image
    output_overlay_path = os.path.join(output, "output_overlay")
    if not os.path.isdir(output_mask_path):
        os.mkdir(output_mask_path)
    if not os.path.isdir(output_overlay_path):
        os.mkdir(output_overlay_path)
    # Loads in existing model
    print(model)
    if not mask_generate:
        model_path = os.path.join(git_repo_base, "models", model)
        model_to_use = load_model(model_path)
    else:
        model_to_use = load_model(model)
    # Resizes and prepares images for prediction
    print(input_file)
    test_gen = testGenerator(    #调整图像和模型兼容
        input_file,
        output_mask_path,
        num_images,
        atlas_to_brain_align=atlas_to_brain_align,
    )
    # Makes predictions on each image
    results = model_to_use.predict(test_gen, steps=num_images, verbose=1)
    # Saves output mask
    saveResult(output_mask_path, results)
    if not mask_generate:
        plot_landmarks = False
        use_dlc = False
        use_voxelmorph = False
        # Predicts and identifies brain regions based on output mask
        applyMask(
            input_file,
            output_mask_path,
            output_overlay_path,
            output_overlay_path,
            mat_save,
            threshold,
            git_repo_base,
            atlas_to_brain_align,
            model,
            dlc_pts,
            atlas_pts,
            olfactory_check,
            use_unet,
            use_dlc,
            use_voxelmorph,
            plot_landmarks,
            align_once,
            atlas_label_list,
            [],
            region_labels,
            original_label,
        )


def predict_regions(config_file):
    """
    Loads parameters into predictRegion from config file.

    :param config_file: The full path to a MesoNet config file (generated using mesonet.config_project())
    """
    cwd = os.getcwd()
    cfg = parse_yaml(config_file)
    input_file = cfg["input_file"]
    num_images = cfg["num_images"]
    model = os.path.join(cwd, cfg["model"])
    output = cfg["output"]
    mat_save = cfg["mat_save"]
    threshold = cfg["threshold"]
    mask_generate = cfg["mask_generate"]
    git_repo_base = cfg["git_repo_base"]
    region_labels = cfg["region_labels"]
    atlas_to_brain_align = cfg["atlas_to_brain_align"]
    dlc_pts = []
    atlas_pts = []
    olfactory_check = cfg["olfactory_check"]
    use_unet = cfg["use_unet"]
    plot_landmarks = cfg["plot_landmarks"]
    align_once = cfg["align_once"]
    atlas_label_list = cfg["atlas_label_list"]
    original_label = cfg["original_label"]

    predictRegion(
        input_file,
        num_images,
        model,
        output,
        mat_save,
        threshold,
        mask_generate,
        git_repo_base,
        atlas_to_brain_align,
        dlc_pts,
        atlas_pts,
        olfactory_check,
        use_unet,
        plot_landmarks,
        align_once,
        atlas_label_list,
        region_labels,
        original_label,
    )
