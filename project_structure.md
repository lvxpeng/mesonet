# Table of Contents
- D:\MyCode\MesoNet\MesoNet\mesonet\atlas_brain_matching.py
- D:\MyCode\MesoNet\MesoNet\mesonet\data.py
- D:\MyCode\MesoNet\MesoNet\mesonet\dlc_predict.py
- D:\MyCode\MesoNet\MesoNet\mesonet\gui_start.py
- D:\MyCode\MesoNet\MesoNet\mesonet\gui_test.py
- D:\MyCode\MesoNet\MesoNet\mesonet\gui_train.py
- D:\MyCode\MesoNet\MesoNet\mesonet\img_augment.py
- D:\MyCode\MesoNet\MesoNet\mesonet\mask_functions.py
- D:\MyCode\MesoNet\MesoNet\mesonet\model.py
- D:\MyCode\MesoNet\MesoNet\mesonet\predict_regions.py
- D:\MyCode\MesoNet\MesoNet\mesonet\train_model.py
- D:\MyCode\MesoNet\MesoNet\mesonet\utils.py
- D:\MyCode\MesoNet\MesoNet\mesonet\voxelmorph_align.py
- D:\MyCode\MesoNet\MesoNet\mesonet\__init__.py

## File: D:\MyCode\MesoNet\MesoNet\mesonet\atlas_brain_matching.py

- Extension: .py
- Language: python
- Size: 57209 bytes
- Created: 2024-11-08 12:56:16
- Modified: 2024-11-08 16:15:42

### Code

```python

from mesonet.utils import natural_sort_key, convert_to_png
from mesonet.mask_functions import atlas_to_mask, applyMask

from mesonet.voxelmorph_align import voxelmorph_align, vxm_transform
import numpy as np
import pandas as pd
import cv2
import imutils
import math
from scipy.io import savemat, loadmat
import skimage.io as io
from skimage.transform import PiecewiseAffineTransform, warp
import imageio
import os
import fnmatch
import glob
from PIL import Image


def find_peaks(img):
    
    max_loc_arr = []
    img = cv2.imread(str(img), 0)
    im = img.copy()
    x_min = int(np.around(im.shape[0] / 2))
    im1 = im[:, x_min: im.shape[0]]
    im2 = im[:, 0:x_min]
    (minVal, max_val, minLoc, max_loc) = cv2.minMaxLoc(im1)
    (minVal2, maxVal2, minLoc2, maxLoc2) = cv2.minMaxLoc(im2)
    max_loc = list(max_loc)
    max_loc[0] = max_loc[0] + x_min
    max_loc = tuple(max_loc)
    max_loc_arr.append(max_loc)
    if (max_val - 30) <= maxVal2 <= (max_val + 30):
        max_loc_arr.append(maxLoc2)
    return max_loc_arr


def coords_to_mat(
        sub_dlc_pts, i, output_mask_path, bregma_present, bregma_index, landmark_arr
):
    if bregma_present:
        x_bregma, y_bregma = sub_dlc_pts[bregma_index]
        for pt, landmark in zip(sub_dlc_pts, landmark_arr):
            x_pt = pt[0]
            y_pt = pt[1]
            pt_adj = [landmark, x_pt - x_bregma, y_pt - y_bregma]
            pt_adj_to_mat = np.array(pt_adj, dtype=object)
            if not os.path.isdir(os.path.join(output_mask_path, "mat_coords")):
                os.mkdir(os.path.join(output_mask_path, "mat_coords"))
            savemat(
                os.path.join(
                    output_mask_path,
                    "mat_coords/landmarks_{}_{}.mat".format(i, landmark),
                ),
                {"landmark_coords_{}_{}".format(i, landmark): pt_adj_to_mat},
                appendmat=False,
            )


def sensory_to_mat(sub_dlc_pts, bregma_pt, i, output_mask_path):
    x_bregma, y_bregma = bregma_pt
    sensory_names = ["tail_left", "tail_right", "visual", "whisker"]
    for pt, landmark in zip(sub_dlc_pts, sensory_names):
        x_pt = pt[0]
        y_pt = pt[1]
        pt_adj = [landmark, x_pt - x_bregma, y_pt - y_bregma]
        pt_adj_to_mat = np.array(pt_adj, dtype=object)
        if not os.path.isdir(os.path.join(output_mask_path, "mat_coords")):
            os.mkdir(os.path.join(output_mask_path, "mat_coords"))
        savemat(
            os.path.join(
                output_mask_path,
                "mat_coords/sensory_peaks_{}_{}.mat".format(i, landmark),
            ),
            {"sensory_peaks_{}_{}".format(i, landmark): pt_adj_to_mat},
            appendmat=False,
        )


def atlas_from_mat(input_file, mat_cnt_list):
    
    file = input_file
    atlas_base = np.zeros((512, 512), dtype="uint8")
    if glob.glob(os.path.join(input_file, "*.mat")):
        mat = loadmat(file)
        mat_shape = mat[list(mat.keys())[3]]
        if len(mat_shape.shape) > 2:
            for val in range(0, mat_shape.shape[2]):
                mat_roi = mat_shape[:, :, val]
                mat_resize = cv2.resize(mat_roi, (512, 512))
                mat_resize = np.uint8(mat_resize)
                ret, thresh = cv2.threshold(mat_resize, 5, 255, cv2.THRESH_BINARY_INV)
                mat_roi_cnt = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                mat_roi_cnt = imutils.grab_contours(mat_roi_cnt)
                c_to_save = max(mat_roi_cnt, key=cv2.contourArea)
                mat_cnt_list.append(c_to_save)
                cv2.drawContours(atlas_base, mat_roi_cnt, -1, (255, 255, 255), 1)
            ret, thresh = cv2.threshold(atlas_base, 5, 255, cv2.THRESH_BINARY_INV)
            io.imsave("atlas_unresized_test.png", thresh)
        else:
            mat = mat["atlas"]
            mat_resize = cv2.resize(mat, (512, 512))
            ret, thresh = cv2.threshold(mat_resize, 5, 255, cv2.THRESH_BINARY_INV)
    else:
        atlas_im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        atlas_resize = np.uint8(atlas_im)
        ret, atlas_resize = cv2.threshold(atlas_resize, 127, 255, 0)
        io.imsave("atlas_unresized_test.png", atlas_resize)
        roi_cnt, hierarchy = cv2.findContours(
            atlas_resize, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )[-2:]
        for val in roi_cnt:
            c_to_save = max(val, key=cv2.contourArea)
            mat_cnt_list.append(c_to_save)
            cv2.drawContours(atlas_base, val, -1, (255, 255, 255), 1)
        ret, thresh = cv2.threshold(atlas_base, 5, 255, cv2.THRESH_BINARY_INV)
    return thresh


def atlas_rotate(dlc_pts, im):
    
    dlc_y_pts = [
        coord if (190 <= coord[0] <= 330) else (1000, 1000) for coord in dlc_pts
    ]
    dlc_y_pts = [coord for coord in dlc_y_pts if coord[0] < 1000]

    
    rotate_rad = math.atan2(0, (im.shape[1] / 2) - dlc_y_pts[-1][0])
    rotate_deg = -1 * (abs(math.degrees(rotate_rad)))
    im_rotate_mat = cv2.getRotationMatrix2D(
        (im.shape[1] / 2, im.shape[0] / 2), rotate_deg, 1.0
    )

    im_rotated = cv2.warpAffine(im, im_rotate_mat, (512, 512))
    x_min = int(np.around(im_rotated.shape[0] / 2))
    im_left = im_rotated[:, 0:x_min]
    im_right = im_rotated[:, x_min: im_rotated.shape[0]]
    return im_left, im_right


def getMaskContour(mask_dir, atlas_img, predicted_pts, actual_pts, cwd, n, main_mask):
    
    c_landmarks = np.empty([0, 2])
    c_atlas_landmarks = np.empty([0, 2])
    mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
    atlas_to_warp = atlas_img
    mask = np.uint8(mask)
    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    for cnt in cnts:
        cnt = cnt[:, 0, :]
        cnt = np.asarray(cnt).astype("float32")
        c_landmarks = np.concatenate((c_landmarks, cnt))
        c_atlas_landmarks = np.concatenate((c_atlas_landmarks, cnt))
    c_landmarks = np.concatenate((c_landmarks, predicted_pts))
    c_atlas_landmarks = np.concatenate((c_atlas_landmarks, actual_pts))
    tform = PiecewiseAffineTransform()
    tform.estimate(c_atlas_landmarks, c_landmarks)
    dst = warp(atlas_to_warp, tform, output_shape=(512, 512))
    if main_mask:
        io.imsave(os.path.join(cwd, "mask_{}.png".format(n)), mask)
    return dst


def atlasBrainMatch(
        brain_img_dir,
        sensory_img_dir,
        coords_input,
        sensory_match,
        mat_save,
        threshold,
        git_repo_base,
        region_labels,
        landmark_arr_orig,
        use_unet,
        use_dlc,
        atlas_to_brain_align,
        model,
        olfactory_check,
        plot_landmarks,
        align_once,
        original_label,
        use_voxelmorph,
        exist_transform,
        voxelmorph_model="motif_model_atlas.h5",
        vxm_template_path="templates",
        dlc_template_path="dlc_templates",
        flow_path="",
):
    
    brain_img_arr = []
    dlc_img_arr = []
    peak_arr = []
    atlas_label_list = []
    dst_list = []
    vxm_template_list = []
    br_list = []
    olfactory_bulbs_to_use_list = []
    olfactory_bulbs_to_use_pre_align_list = []

    voxelmorph_model_path = os.path.join(
        git_repo_base, "models", "voxelmorph", voxelmorph_model
    )
    convert_to_png(vxm_template_path)
    files = glob.glob(os.path.join(git_repo_base, "atlases", vxm_template_path, "*.png"))
    if files:
        vxm_template_orig = cv2.imread(files[0])
    else:
        
        raise FileNotFoundError(
            f"No .png files found in path: {os.path.join(git_repo_base, 'atlases', vxm_template_path)}")
    cwd = os.getcwd()
    output_mask_path = os.path.join(cwd, "../output_mask")
    output_overlay_path = os.path.join(cwd, "../output_overlay")
    if not os.path.isdir(output_mask_path):
        os.mkdir(output_mask_path)
    if not os.path.isdir(output_overlay_path):
        os.mkdir(output_overlay_path)

    if not atlas_to_brain_align:
        im = cv2.imread(
            os.path.join(git_repo_base, "atlases/Atlas_workflow2_binary.png")
        )
    else:
        if use_voxelmorph and not use_dlc:
            im = cv2.imread(
                os.path.join(git_repo_base, "atlases/Atlas_for_Voxelmorph_binary.png")
            )
        else:
            im = cv2.imread(
                os.path.join(git_repo_base, "atlases/Atlas_workflow1_binary.png")
            )
        im_left = cv2.imread(os.path.join(git_repo_base, "atlases/left_hemi.png"))
        ret, im_left = cv2.threshold(im_left, 5, 255, cv2.THRESH_BINARY_INV)
        im_right = cv2.imread(os.path.join(git_repo_base, "atlases/right_hemi.png"))
        ret, im_right = cv2.threshold(im_right, 5, 255, cv2.THRESH_BINARY_INV)
        im_left = np.uint8(im_left)
        im_right = np.uint8(im_right)
        im = np.uint8(im)
    atlas = im
    
    for num, file in enumerate(os.listdir(cwd)):
        if fnmatch.fnmatch(file, "*.png") and "mask" not in file:
            dlc_img_arr.append(os.path.join(cwd, file))
    for num, file in enumerate(os.listdir(brain_img_dir)):
        if fnmatch.fnmatch(file, "*.png"):
            brain_img_arr.append(os.path.join(brain_img_dir, file))
            brain_img_arr.sort(key=natural_sort_key)
        elif fnmatch.fnmatch(file, "*.tif"):
            tif_stack = imageio.mimread(os.path.join(brain_img_dir, file))
            for tif_im in tif_stack:
                brain_img_arr.append(tif_im)

    
    coord_circles_img = cv2.imread(
        os.path.join(
            git_repo_base, "atlases", "multi_landmark", "landmarks_new_binary.png"
        ),
        cv2.IMREAD_GRAYSCALE,
    )
    coord_circles_img = np.uint8(coord_circles_img)
    
    circles, hierarchy = cv2.findContours(
        coord_circles_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )[-2:]
    
    if circles is not None:
        
        atlas_arr = np.array(
            [
                (
                    int(cv2.moments(circle)["m10"] / cv2.moments(circle)["m00"]),
                    int(cv2.moments(circle)["m01"] / cv2.moments(circle)["m00"]),
                )
                for circle in circles
            ]
        )

    
    atlas_arr = np.array(
        [
            (102, 148),
            (166, 88),
            (214, 454),
            (256, 88),
            (256, 256),
            (256, 428),
            (410, 148),
            (346, 88),
            (298, 454),
        ]
    )

    peak_arr_flat = []
    peak_arr_total = []

    
    if sensory_match:
        for num, file in enumerate(brain_img_arr):
            img_name = str(os.path.splitext(os.path.basename(file))[0])
            sensory_img_for_brain = os.path.join(sensory_img_dir, img_name)
            if glob.glob(sensory_img_for_brain):
                sensory_img_for_brain_dir = os.listdir(sensory_img_for_brain)
                sensory_img_for_brain_dir.sort(key=natural_sort_key)
                for num_im, file_im in enumerate(sensory_img_for_brain_dir):
                    sensory_im = io.imread(
                        os.path.join(sensory_img_dir, img_name, file_im)
                    )
                    sensory_im = np.uint8(sensory_im)
                    sensory_im = cv2.resize(sensory_im, (512, 512))
                    io.imsave(
                        os.path.join(sensory_img_dir, img_name, file_im), sensory_im
                    )
                    peak = find_peaks(os.path.join(sensory_img_dir, img_name, file_im))
                    peak_arr.append(peak)
            for x in peak_arr:
                for y in x:
                    peak_arr_flat.append(y)
            peak_arr_total.append(peak_arr_flat)
            peak_arr_flat = []
            peak_arr = []

    
    dlc_pts = []
    atlas_pts = []
    sensory_peak_pts = []
    sensory_atlas_pts = []
    sub_dlc_pts = []
    sub_atlas_pts = []
    sub_sensory_peak_pts = []
    sub_sensory_atlas_pts = []

    bregma_index_list = []
    bregma_list = []
    if use_dlc:
        bregma_present = True
    else:
        bregma_present = False

    coords = pd.read_csv(coords_input)
    x_coord = coords.iloc[2:, 1::3]
    y_coord = coords.iloc[2:, 2::3]
    accuracy = coords.iloc[2:, 3::3]
    acc_left_total = accuracy.iloc[:, 0:5]
    acc_right_total = accuracy.iloc[:, 3:8]
    landmark_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]  
    for arr_index, i in enumerate(range(0, len(x_coord))):
        landmark_arr = landmark_arr_orig
        x_coord_flat = x_coord.iloc[i].values.astype("float32")
        y_coord_flat = y_coord.iloc[i].values.astype("float32")
        x_coord_flat = x_coord_flat[landmark_arr]
        y_coord_flat = y_coord_flat[landmark_arr]
        dlc_list = []
        atlas_list = []
        for coord_x, coord_y in zip(x_coord_flat, y_coord_flat):
            dlc_coord = (coord_x, coord_y)
            dlc_list.append(dlc_coord)
        for coord_atlas in atlas_arr:
            atlas_coord = (coord_atlas[0], coord_atlas[1])
            atlas_list.append(atlas_coord)
        atlas_list = [atlas_list[i] for i in landmark_arr]

        
        landmark_indices = landmark_indices[0: len(landmark_arr)]
        atlas_indices = landmark_arr

        pts_dist = np.absolute(
            np.asarray(atlas_list) - np.asarray((im.shape[0] / 2, im.shape[1] / 2))
        )
        pts_avg_dist = [np.mean(v) for v in pts_dist]
        bregma_index = np.argmin(np.asarray(pts_avg_dist))

        for j in landmark_indices:
            sub_dlc_pts.append([x_coord_flat[j], y_coord_flat[j]])
        for j in atlas_indices:
            sub_atlas_pts.append([atlas_arr[j][0], atlas_arr[j][1]])

        dlc_pts.append(sub_dlc_pts)
        atlas_pts.append(sub_atlas_pts)
        coords_to_mat(
            sub_dlc_pts, i, output_mask_path, bregma_present, bregma_index, landmark_arr
        )
        bregma_index_list.append(bregma_index)
        sub_dlc_pts = []
        sub_atlas_pts = []
    if sensory_match:
        k_coord, m_coord = np.array([(189, 323, 435, 348), (315, 315, 350, 460)])
        coords_peak = peak_arr_total
        for img_num, img in enumerate(brain_img_arr):
            for j in [1, 0, 3, 2]:  
                sub_sensory_peak_pts.append(
                    [coords_peak[img_num][j][0], coords_peak[img_num][j][1]]
                )
            for j in [0, 1, 2, 3]:  
                sub_sensory_atlas_pts.append([k_coord[j], m_coord[j]])
            sensory_peak_pts.append(sub_sensory_peak_pts)
            sensory_atlas_pts.append(sub_sensory_atlas_pts)
            sensory_to_mat(
                sub_sensory_peak_pts, dlc_pts[img_num][3], img_num, output_mask_path
            )
            sub_sensory_peak_pts = []
            sub_sensory_atlas_pts = []
        sensory_peak_pts, sensory_atlas_pts = (
            np.asarray(sensory_peak_pts).astype("float32"),
            np.asarray(sensory_atlas_pts).astype("float32"),
        )

    for n, br in enumerate(brain_img_arr):
        vxm_template = np.uint8(vxm_template_orig)
        vxm_template = cv2.resize(vxm_template, (512, 512))

        align_val = n
        if atlas_to_brain_align:
            im = np.uint8(im)
            br = cv2.imread(br)
            br = np.uint8(br)
            br = cv2.resize(br, (512, 512))
        else:
            
            if ".png" in br:
                im = cv2.imread(br)
            else:
                im = br
            im = np.uint8(im)
            im = cv2.resize(im, (512, 512))

        
        if atlas_to_brain_align:
            if use_voxelmorph and not use_dlc:
                atlas_mask_dir = os.path.join(
                    git_repo_base, "atlases/Atlas_for_Voxelmorph_border.png"
                )
            else:
                atlas_mask_dir = os.path.join(
                    git_repo_base, "atlases/atlas_smooth2_binary.png"
                )
        else:
            atlas_mask_dir = os.path.join(
                git_repo_base, "atlases/atlas_smooth2_binary.png"
            )
        atlas_mask_dir_left = os.path.join(
            git_repo_base, "atlases/left_hemisphere_smooth.png"
        )
        atlas_mask_dir_right = os.path.join(
            git_repo_base, "atlases/right_hemisphere_smooth.png"
        )
        atlas_label_mask_dir = os.path.join(
            git_repo_base, "atlases/diff_colour_regions/Common_atlas.mat"
        )
        atlas_label_mask_dir_left = os.path.join(
            git_repo_base, "atlases/diff_colour_regions/atlas_left_hemisphere.csv"
        )
        atlas_label_mask_dir_right = os.path.join(
            git_repo_base, "atlases/diff_colour_regions/atlas_right_hemisphere.csv"
        )
        atlas_label_mask_left = np.genfromtxt(atlas_label_mask_dir_left, delimiter=",")
        atlas_label_mask_right = np.genfromtxt(
            atlas_label_mask_dir_right, delimiter=","
        )
        atlas_mask_left = cv2.imread(atlas_mask_dir_left, cv2.IMREAD_UNCHANGED)
        atlas_mask_left = cv2.resize(atlas_mask_left, (im.shape[0], im.shape[1]))
        atlas_mask_left = np.uint8(atlas_mask_left)
        atlas_mask_right = cv2.imread(atlas_mask_dir_right, cv2.IMREAD_UNCHANGED)
        atlas_mask_right = cv2.resize(atlas_mask_right, (im.shape[0], im.shape[1]))
        atlas_mask_right = np.uint8(atlas_mask_right)

        atlas_mask = cv2.imread(atlas_mask_dir, cv2.IMREAD_UNCHANGED)
        atlas_mask = cv2.resize(atlas_mask, (im.shape[0], im.shape[1]))
        atlas_mask = np.uint8(atlas_mask)
        mask_dir = os.path.join(cwd, "../output_mask/{}.png".format(n))
        if use_voxelmorph and n == 1:
            try:
                os.remove(mask_dir)
            except:
                print("Cannot remove second U-Net output for VoxelMorph!")

        print("Performing first transformation of atlas {}...".format(n))

        mask_warped_path = os.path.join(
            output_mask_path, "{}_mask_warped.png".format(str(n))
        )
        mask_warped_path_alt_left = os.path.join(
            output_mask_path, "{}_mask_warped_left.png".format(str(n))
        )
        mask_warped_path_alt_right = os.path.join(
            output_mask_path, "{}_mask_warped_right.png".format(str(n))
        )
        brain_to_atlas_mask_path = os.path.join(
            output_mask_path, "{}_brain_to_atlas_mask.png".format(str(n))
        )
        if use_dlc:
            
            atlas_pts_for_input = np.array([atlas_pts[n][0: len(dlc_pts[n])]]).astype(
                "float32"
            )
            pts_for_input = np.array([dlc_pts[n]]).astype("float32")

            if align_once:
                align_val = 0
            else:
                align_val = n

            
            
            if len(atlas_pts_for_input[0]) == 2:
                atlas_pts_for_input = np.append(
                    atlas_pts_for_input[0], [[0, 0]], axis=0
                )
                pts_for_input = np.append(pts_for_input[0], [[0, 0]], axis=0)
            
            if len(atlas_pts_for_input[0]) <= 2:
                warp_coords = cv2.estimateAffinePartial2D(
                    atlas_pts_for_input, pts_for_input
                )[0]

                
                with open(os.path.join(output_mask_path, "{}_atlas_transform.txt".format(str(n))), "w") as f:
                    for row in warp_coords:
                        np.savetxt(f, [row], fmt='%f')  

                if atlas_to_brain_align:
                    atlas_warped_left = cv2.warpAffine(im_left, warp_coords, (512, 512))
                    atlas_warped_right = cv2.warpAffine(
                        im_right, warp_coords, (512, 512)
                    )
                    atlas_warped = cv2.bitwise_or(atlas_warped_left, atlas_warped_right)
                    ret, atlas_warped = cv2.threshold(
                        atlas_warped, 5, 255, cv2.THRESH_BINARY_INV
                    )
                    atlas_left_transform_path = os.path.join(
                        output_mask_path, "{}_atlas_left_transform.png".format(str(n))
                    )
                    atlas_right_transform_path = os.path.join(
                        output_mask_path, "{}_atlas_right_transform.png".format(str(n))
                    )
                    io.imsave(atlas_left_transform_path, atlas_warped_left)
                    io.imsave(atlas_right_transform_path, atlas_warped_right)
                else:
                    atlas_warped = cv2.warpAffine(im, warp_coords, (512, 512))
            
            elif len(atlas_pts_for_input[0]) == 3:
                warp_coords = cv2.getAffineTransform(atlas_pts_for_input, pts_for_input)
                
                with open(os.path.join(output_mask_path, "{}_atlas_transform.txt".format(str(n))), "w") as f:
                    for row in warp_coords:
                        np.savetxt(f, [row], fmt='%f')  

                atlas_warped = cv2.warpAffine(im, warp_coords, (512, 512))
            
            elif len(atlas_pts_for_input[0]) >= 4:
                im_final_size = (512, 512)

                left = acc_left_total.iloc[n, :].values.astype("float32").tolist()
                right = acc_right_total.iloc[n, :].values.astype("float32").tolist()
                left = np.argsort(left).tolist()
                right = np.argsort(right).tolist()
                right = [x + 1 for x in right]
                
                if {0, 3, 5, 6}.issubset(landmark_arr) and len(landmark_arr) >= 7:
                    left = [0, 3, 5]
                    right = [3, 5, 6]    
                else:  
                    left = [
                               landmark_arr.index(x)
                               for x in landmark_arr
                               if x in [0, 1, 2, 3, 4, 5]
                           ][0:3]
                    
                    right = [
                                landmark_arr.index(x)
                                for x in landmark_arr
                                if x in [3, 4, 5, 6, 7, 8]
                            ][-3:]
                    print(landmark_indices)
                    print(landmark_arr)
                    print(left)
                    print(right)

                
                atlas_pts_left = np.array(
                    [
                        atlas_pts[align_val][left[0]],
                        atlas_pts[align_val][left[1]],
                        atlas_pts[align_val][left[2]],
                    ],
                    dtype=np.float32,
                )
                atlas_pts_right = np.array(
                    [
                        atlas_pts[align_val][right[0]],
                        atlas_pts[align_val][right[1]],
                        atlas_pts[align_val][right[2]],
                    ],
                    dtype=np.float32,
                )
                dlc_pts_left = np.array(
                    [
                        dlc_pts[align_val][left[0]],
                        dlc_pts[align_val][left[1]],
                        dlc_pts[align_val][left[2]],
                    ],
                    dtype=np.float32,
                )
                dlc_pts_right = np.array(
                    [
                        dlc_pts[align_val][right[0]],
                        dlc_pts[align_val][right[1]],
                        dlc_pts[align_val][right[2]],
                    ],
                    dtype=np.float32,
                )

                warp_coords_left = cv2.getAffineTransform(atlas_pts_left, dlc_pts_left)

                
                with open(os.path.join(output_mask_path, "{}_atlas_transform_left.txt".format(str(n))), "w") as f:
                    for row in warp_coords_left:
                        np.savetxt(f, [row], fmt='%f')  

                warp_coords_right = cv2.getAffineTransform(
                    atlas_pts_right, dlc_pts_right
                )

                
                with open(os.path.join(output_mask_path, "{}_atlas_transform_right.txt".format(str(n))), "w") as f:
                    for row in warp_coords_right:
                        np.savetxt(f, [row], fmt='%f')  

                warp_coords_brain_atlas_left = cv2.getAffineTransform(
                    dlc_pts_left, atlas_pts_left
                )
                warp_coords_brain_atlas_right = cv2.getAffineTransform(
                    dlc_pts_right, atlas_pts_right
                )
                if atlas_to_brain_align:
                    atlas_warped_left = cv2.warpAffine(
                        im_left, warp_coords_left, im_final_size
                    )
                    atlas_warped_right = cv2.warpAffine(
                        im_right, warp_coords_right, im_final_size
                    )
                    atlas_warped = cv2.bitwise_or(atlas_warped_left, atlas_warped_right)
                    ret, atlas_warped = cv2.threshold(
                        atlas_warped, 5, 255, cv2.THRESH_BINARY_INV
                    )
                    if not original_label:
                        atlas_label_left = cv2.warpAffine(
                            atlas_label_mask_left, warp_coords_left, im_final_size
                        )
                        atlas_label_right = cv2.warpAffine(
                            atlas_label_mask_right, warp_coords_right, im_final_size
                        )
                        atlas_label = cv2.bitwise_or(
                            atlas_label_left, atlas_label_right
                        )

                else:
                    pts_np = np.array(
                        [
                            dlc_pts[align_val][0],
                            dlc_pts[align_val][1],
                            dlc_pts[align_val][2],
                        ],
                        dtype=np.float32,
                    )
                    atlas_pts_np = np.array(
                        [
                            atlas_pts[align_val][0],
                            atlas_pts[align_val][1],
                            atlas_pts[align_val][2],
                        ],
                        dtype=np.float32,
                    )
                    warp_coords = cv2.getAffineTransform(pts_np, atlas_pts_np)
                    atlas_warped = cv2.warpAffine(im, warp_coords, (512, 512))

            
            
            if len(atlas_pts_for_input[0]) == 2:
                atlas_mask_left_warped = cv2.warpAffine(
                    atlas_mask_left, warp_coords, (512, 512)
                )
                atlas_mask_right_warped = cv2.warpAffine(
                    atlas_mask_right, warp_coords, (512, 512)
                )
                atlas_mask_warped = cv2.bitwise_or(
                    atlas_mask_left_warped, atlas_mask_right_warped
                )
            
            if len(atlas_pts_for_input[0]) == 3:
                atlas_mask_warped = cv2.warpAffine(atlas_mask, warp_coords, (512, 512))
            
            if len(atlas_pts_for_input[0]) >= 4:
                atlas_mask_left_warped = cv2.warpAffine(
                    atlas_mask_left, warp_coords_left, (512, 512)
                )
                atlas_mask_right_warped = cv2.warpAffine(
                    atlas_mask_right, warp_coords_right, (512, 512)
                )
                atlas_mask_warped = cv2.bitwise_or(
                    atlas_mask_left_warped, atlas_mask_right_warped
                )
            atlas_mask_warped = np.uint8(atlas_mask_warped)
            io.imsave(mask_warped_path_alt_left, atlas_mask_left_warped)
            io.imsave(mask_warped_path_alt_right, atlas_mask_right_warped)

            hemispheres = ["left", "right"]

            
            if olfactory_check and use_unet:
                if align_once and n != 0:
                    olfactory_bulbs = olfactory_bulbs_to_use_pre_align_list[0]
                else:
                    mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
                    cnts_for_olfactory = cv2.findContours(
                        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                    )
                    cnts_for_olfactory = imutils.grab_contours(cnts_for_olfactory)
                    if len(cnts_for_olfactory) == 3:
                        olfactory_bulbs = sorted(
                            cnts_for_olfactory, key=cv2.contourArea, reverse=True
                        )[1:3]
                    else:
                        olfactory_bulbs = sorted(
                            cnts_for_olfactory, key=cv2.contourArea, reverse=True
                        )[2:4]
                if align_once and n == 0:
                    olfactory_bulbs_to_use_pre_align_list.append(olfactory_bulbs)
                if len(olfactory_bulbs) == 0:
                    
                    
                    olfactory_check = False
            for hemisphere in hemispheres:
                new_data = []
                if hemisphere == "left":
                    mask_path = mask_warped_path_alt_left
                    mask_warped_to_use = atlas_mask_left_warped
                    if olfactory_check and use_unet:
                        if len(olfactory_bulbs) >= 1:
                            bulb = olfactory_bulbs[0]
                        
                else:
                    mask_path = mask_warped_path_alt_right
                    mask_warped_to_use = atlas_mask_right_warped
                    if olfactory_check and use_unet:
                        if len(olfactory_bulbs) > 1:
                            bulb = olfactory_bulbs[1]
                        
                if olfactory_check and use_unet:
                    
                    try:
                        cv2.fillPoly(
                            mask_warped_to_use, pts=[bulb], color=[255, 255, 255]
                        )
                        io.imsave(mask_path, mask_warped_to_use)
                    except:
                        print("No olfactory bulb found!")
                    mask_warped_to_use = cv2.cvtColor(
                        mask_warped_to_use, cv2.COLOR_BGR2GRAY
                    )
                img_edited = Image.open(mask_path)
                
                img_rgba = img_edited.convert("RGBA")
                data = img_rgba.getdata()
                for pixel in data:
                    if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
                        new_data.append((pixel[0], pixel[1], pixel[2], 0))
                    else:
                        new_data.append(pixel)
                img_rgba.putdata(new_data)
                img_rgba.save(
                    os.path.join(
                        output_mask_path,
                        "{}_brain_atlas_mask_transparent_{}.png".format(n, hemisphere),
                    )
                )
                img_transparent = cv2.imread(
                    os.path.join(
                        output_mask_path,
                        "{}_brain_atlas_mask_transparent_{}.png".format(n, hemisphere),
                    )
                )
                brain_atlas_transparent = cv2.bitwise_and(im, img_transparent)

                io.imsave(
                    os.path.join(
                        output_mask_path,
                        "{}_brain_atlas_transparent_{}.png".format(n, hemisphere),
                    ),
                    brain_atlas_transparent,
                )
                if hemisphere == "left":
                    if atlas_to_brain_align:
                        brain_to_atlas_warped_left = cv2.warpAffine(
                            brain_atlas_transparent, warp_coords_left, (512, 512)
                        )
                        if olfactory_check and use_unet:
                            olfactory_warped_left = cv2.warpAffine(
                                mask_warped_to_use, warp_coords_left, (512, 512)
                            )
                            olfactory_warped_cnts_left = cv2.findContours(
                                olfactory_warped_left.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE,
                            )
                            olfactory_warped_cnts_left = imutils.grab_contours(
                                olfactory_warped_cnts_left
                            )
                            olfactory_warped_left = sorted(
                                olfactory_warped_cnts_left,
                                key=cv2.contourArea,
                                reverse=True,
                            )[-1]
                    else:
                        brain_to_atlas_warped_left = cv2.warpAffine(
                            brain_atlas_transparent,
                            warp_coords_brain_atlas_left,
                            (512, 512),
                        )
                        if olfactory_check and use_unet:
                            olfactory_warped_left = cv2.warpAffine(
                                mask_warped_to_use,
                                warp_coords_brain_atlas_left,
                                (512, 512),
                            )
                            olfactory_warped_cnts_left = cv2.findContours(
                                olfactory_warped_left.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE,
                            )
                            olfactory_warped_cnts_left = imutils.grab_contours(
                                olfactory_warped_cnts_left
                            )
                            olfactory_warped_left = sorted(
                                olfactory_warped_cnts_left,
                                key=cv2.contourArea,
                                reverse=True,
                            )[-1]
                else:
                    if atlas_to_brain_align:
                        brain_to_atlas_warped_right = cv2.warpAffine(
                            brain_atlas_transparent, warp_coords_right, (512, 512)
                        )
                        olfactory_warped_right = cv2.warpAffine(
                            mask_warped_to_use, warp_coords_right, (512, 512)
                        )
                        if olfactory_check and use_unet:
                            olfactory_warped_right = cv2.warpAffine(
                                mask_warped_to_use, warp_coords_right, (512, 512)
                            )
                            olfactory_warped_cnts_right = cv2.findContours(
                                olfactory_warped_right.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE,
                            )
                            olfactory_warped_cnts_right = imutils.grab_contours(
                                olfactory_warped_cnts_right
                            )
                            olfactory_warped_right = sorted(
                                olfactory_warped_cnts_right,
                                key=cv2.contourArea,
                                reverse=True,
                            )[-1]
                    else:
                        brain_to_atlas_warped_right = cv2.warpAffine(
                            brain_atlas_transparent,
                            warp_coords_brain_atlas_right,
                            (512, 512),
                        )
                        olfactory_warped_right = cv2.warpAffine(
                            mask_warped_to_use,
                            warp_coords_brain_atlas_right,
                            (512, 512),
                        )
                        if olfactory_check and use_unet:
                            olfactory_warped_right = cv2.warpAffine(
                                mask_warped_to_use,
                                warp_coords_brain_atlas_right,
                                (512, 512),
                            )
                            olfactory_warped_cnts_right = cv2.findContours(
                                olfactory_warped_right.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE,
                            )
                            olfactory_warped_cnts_right = imutils.grab_contours(
                                olfactory_warped_cnts_right
                            )
                            olfactory_warped_right = sorted(
                                olfactory_warped_cnts_right,
                                key=cv2.contourArea,
                                reverse=True,
                            )[-1]
            if olfactory_check and use_unet:
                olfactory_bulbs_to_use_list.append(
                    [olfactory_warped_left, olfactory_warped_right]
                )
            brain_to_atlas_mask = cv2.bitwise_or(
                brain_to_atlas_warped_left, brain_to_atlas_warped_right
            )
            io.imsave(brain_to_atlas_mask_path, brain_to_atlas_mask)

            
            print("Performing second transformation of atlas {}...".format(n))

            atlas_first_transform_path = os.path.join(
                output_mask_path, "{}_atlas_first_transform.png".format(str(n))
            )

            if atlas_to_brain_align:
                dst = atlas_warped
            else:
                dst = brain_to_atlas_mask

            io.imsave(atlas_first_transform_path, dst)

            
            
            if sensory_match:
                original_label = True
                if atlas_to_brain_align:
                    dst = getMaskContour(
                        mask_dir,
                        atlas_warped,
                        sensory_peak_pts[align_val],
                        sensory_atlas_pts[align_val],
                        cwd,
                        align_val,
                        False,
                    )
                    atlas_mask_warped = getMaskContour(
                        atlas_first_transform_path,
                        atlas_mask_warped,
                        sensory_peak_pts[align_val],
                        sensory_atlas_pts[align_val],
                        cwd,
                        align_val,
                        False,
                    )
                    print(np.shape(atlas_mask_warped))
                    atlas_mask_warped = np.uint8(atlas_mask_warped)
                    atlas_mask_warped = cv2.resize(
                        atlas_mask_warped, (im.shape[0], im.shape[1])
                    )
                    atlas_mask_warped = (atlas_mask_warped * 255).astype(np.uint8)
                else:
                    dst = atlas_warped
        else:
            
            if atlas_to_brain_align and not use_voxelmorph:
                dst = cv2.bitwise_or(im_left, im_right)
                ret, dst = cv2.threshold(dst, 5, 255, cv2.THRESH_BINARY_INV)
            else:
                dst = im
            dst = np.uint8(dst)

        if use_dlc:
            if atlas_to_brain_align:
                io.imsave(mask_warped_path, atlas_mask_warped)
            else:
                io.imsave(mask_warped_path, atlas_mask)
        else:
            io.imsave(mask_warped_path, dst)
            if use_voxelmorph:
                atlas_mask_warped = atlas_mask
            else:
                atlas_mask_warped = cv2.bitwise_or(atlas_mask_left, atlas_mask_right)
            atlas_mask_warped = cv2.cvtColor(atlas_mask_warped, cv2.COLOR_BGR2GRAY)
            ret, atlas_mask_warped = cv2.threshold(
                atlas_mask_warped, 5, 255, cv2.THRESH_BINARY
            )
            atlas_mask_warped = np.uint8(atlas_mask_warped)
            original_label = True
            io.imsave(mask_warped_path, atlas_mask_warped)

        
        dst = cv2.resize(dst, (im.shape[0], im.shape[1]))
        atlas_path = os.path.join(output_mask_path, "{}_atlas.png".format(str(n)))

        vxm_template_output_path = os.path.join(
            output_mask_path, "{}_vxm_template.png".format(str(n))
        )

        dst_list.append(dst)
        if use_voxelmorph:
            vxm_template_list.append(vxm_template)
            io.imsave(vxm_template_output_path, vxm_template_list[n])

        if atlas_to_brain_align:
            dst = (dst * 255).astype(np.uint8)
            io.imsave(atlas_path, dst)
            br_list.append(br)
        else:
            brain_warped_path = os.path.join(
                output_mask_path, "{}_brain_warp.png".format(str(n))
            )
            io.imsave(brain_warped_path, dst)
            io.imsave(atlas_path, atlas)

        if atlas_to_brain_align:
            if original_label:
                atlas_label = []
            atlas_label = atlas_to_mask(
                atlas_path,
                mask_dir,
                mask_warped_path,
                output_mask_path,
                n,
                use_unet,
                use_voxelmorph,
                atlas_to_brain_align,
                git_repo_base,
                olfactory_check,
                [],
                atlas_label,
            )
            atlas_label_list.append(atlas_label)
        elif not use_dlc:
            io.imsave(os.path.join(output_mask_path, "{}.png".format(n)), dst)
        if bregma_present:
            bregma_val = int(bregma_index_list[n])
            bregma_list.append(dlc_pts[n][bregma_val])

    
    if use_voxelmorph:
        align_once = True
        if len(dst_list) == 1:
            n_to_use = 0
        else:
            n_to_use = 1
        for (n_post, dst_post), vxm_template_post in zip(
                enumerate([dst_list[n_to_use]]), [vxm_template_list[n_to_use]]
        ):
            output_img, flow_post = voxelmorph_align(
                voxelmorph_model_path,
                dst_post,
                vxm_template_post,
                exist_transform,
                flow_path,
            )
            flow_path_after = os.path.join(
                output_mask_path, "{}_flow.npy".format(str(n_post))
            )
            np.save(flow_path_after, flow_post)
            output_path_after = os.path.join(
                output_mask_path, "{}_output_img.png".format(str(n_post))
            )
            output_img = (output_img * 255).astype(np.uint8)
            io.imsave(output_path_after, output_img)
            if not exist_transform:
                dst_gray = cv2.cvtColor(atlas, cv2.COLOR_BGR2GRAY)
                dst_post = vxm_transform(dst_gray, flow_path_after)
                ret, dst_post = cv2.threshold(
                    dst_post, 1, 255, cv2.THRESH_BINARY
                )  
                dst_post = np.uint8(dst_post)

            mask_warped_path = os.path.join(
                output_mask_path, "{}_mask_warped.png".format(str(n_post))
            )

            atlas_first_transform_path_post = os.path.join(
                output_mask_path, "{}_atlas_first_transform.png".format(str(n_post))
            )

            io.imsave(atlas_first_transform_path_post, dst_post)
            if align_once and use_voxelmorph:
                atlas_path = os.path.join(
                    output_mask_path, "{}_atlas.png".format(str(0))
                )

            brain_warped_path = os.path.join(
                output_mask_path, "{}_brain_warp.png".format(str(n_post))
            )
            mask_dir = os.path.join(cwd, "../output_mask/{}.png".format(n_post))
            dst_post = cv2.resize(dst_post, (im.shape[0], im.shape[1]))
            if not atlas_to_brain_align:
                atlas_to_brain_align = True
                original_label = True
            if atlas_to_brain_align:
                io.imsave(atlas_path, dst_post)
            else:
                io.imsave(brain_warped_path, dst_post)
            if atlas_to_brain_align:
                if original_label:
                    atlas_label = []
                if use_voxelmorph and olfactory_check and use_unet and use_dlc:
                    if align_once:
                        olfactory_bulbs_to_use_check = olfactory_bulbs_to_use_list[0]
                    else:
                        olfactory_bulbs_to_use_check = olfactory_bulbs_to_use_list[
                            n_post
                        ]
                else:
                    olfactory_bulbs_to_use_check = []
                atlas_label = atlas_to_mask(
                    atlas_path,
                    mask_dir,
                    mask_warped_path,
                    output_mask_path,
                    n_post,
                    use_unet,
                    use_voxelmorph,
                    atlas_to_brain_align,
                    git_repo_base,
                    olfactory_check,
                    olfactory_bulbs_to_use_check,
                    atlas_label,
                )
                atlas_label_list.append(atlas_label)

    
    applyMask(
        brain_img_dir,
        output_mask_path,
        output_overlay_path,
        output_overlay_path,
        mat_save,
        threshold,
        git_repo_base,
        bregma_list,
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
        olfactory_bulbs_to_use_list,
        region_labels,
        original_label,
    )

```

## File: D:\MyCode\MesoNet\MesoNet\mesonet\data.py

- Extension: .py
- Language: python
- Size: 3473 bytes
- Created: 2024-11-08 12:56:16
- Modified: 2024-11-08 12:56:16

### Code

```python

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans


def adjustData(img, mask, flag_multi_class, num_class):
    if flag_multi_class:
        img = img / 255
        mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i, i] = 1
        new_mask = (
            np.reshape(
                new_mask,
                (
                    new_mask.shape[0],
                    new_mask.shape[1] * new_mask.shape[2],
                    new_mask.shape[3],
                ),
            )
            if flag_multi_class
            else np.reshape(
                new_mask, (new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2])
            )
        )
        mask = new_mask
    elif np.max(img) > 1:
        img = img / 255
        mask = mask / 255
        mask[mask > 0.7] = 1
        mask[mask <= 0.7] = 0
    return img, mask


def trainGenerator(
    batch_size,
    train_path,
    image_folder,
    mask_folder,
    aug_dict,
    image_color_mode="grayscale",
    mask_color_mode="grayscale",
    image_save_prefix="image",
    mask_save_prefix="mask",
    flag_multi_class=False,
    num_class=2,
    save_to_dir=None,
    target_size=(512, 512),
    seed=1,
):
    
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
    )
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed,
    )
    train_generator = zip(image_generator, mask_generator)
    for img, mask in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield img, mask


def testGenerator(
    test_path,
    num_image=60,
    target_size=(512, 512),
    flag_multi_class=False,
    as_gray=True,
):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%d.png" % i), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img

```

## File: D:\MyCode\MesoNet\MesoNet\mesonet\dlc_predict.py

- Extension: .py
- Language: python
- Size: 16527 bytes
- Created: 2024-11-08 12:56:16
- Modified: 2024-11-08 12:56:16

### Code

```python

import deeplabcut
from mesonet.atlas_brain_matching import atlasBrainMatch
from mesonet.utils import parse_yaml, natural_sort_key
from deeplabcut.utils.auxiliaryfunctions import read_config, write_config
import cv2
import glob
import imageio
import os
import numpy as np
import re
from sys import platform


def DLCPredict(
    config,
    input_file,
    output,
    atlas,
    sensory_match,
    sensory_path,
    mat_save,
    threshold,
    git_repo_base,
    region_labels,
    landmark_arr,
    use_unet,
    use_dlc,
    atlas_to_brain_align,
    model,
    olfactory_check,
    plot_landmarks,
    align_once,
    original_label,
    use_voxelmorph,
    exist_transform,
    voxelmorph_model,
    template_path,
    flow_path,
    coords_input_file,
):
    
    img_array = []
    if sensory_match == 1:
        sensory_match = True
    else:
        sensory_match = False
    if sensory_match:
        sensory_img_dir = sensory_path
    else:
        sensory_img_dir = ""
    tif_list = glob.glob(os.path.join(input_file, "*tif"))
    if tif_list:
        print(tif_list)
        tif_stack = imageio.mimread(os.path.join(input_file, tif_list[0]))
        filenames = tif_stack
    else:
        filenames = glob.glob(os.path.join(input_file, "*.png"))
        filenames.sort(key=natural_sort_key)

    size = (512, 512)
    print(len(filenames))
    for filename in filenames:
        print(filename)
        if tif_list:
            img = filename
            img = np.uint8(img)
            img = cv2.resize(img, size)
            height, width = img.shape
        else:
            img = cv2.imread(filename)
            img = np.uint8(img)
            img = cv2.resize(img, size)
            height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    if len(img_array) > 0:
        video_output_path = os.path.join(output, "dlc_output")

        video_name = os.path.join(video_output_path, "tmp_video.mp4")

        if not os.path.isdir(video_output_path):
            os.mkdir(video_output_path)
        if not coords_input_file:
            
            if platform == "linux" or platform == "linux2" or platform == "darwin":
                fourcc = cv2.VideoWriter_fourcc("M", "P", "E", "G")
            else:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_name, fourcc, 30, size)
            for i in img_array:
                
                out.write(i)
            out.release()

            print("DLC config file path: {}".format(config))

            deeplabcut.analyze_videos(
                config, [video_output_path], videotype=".mp4", save_as_csv=True
            )
            deeplabcut.create_labeled_video(config, [video_name], filtered=True)
            
            
            
            scorer_name = "DLC"
            output_video_name = ""
            coords_input = ""
            for filename in glob.glob(
                os.path.join(video_output_path, "tmp_video" + scorer_name + "*.*")
            ):
                try:
                    if ".mp4" in filename:
                        output_video_name = filename
                    elif ".csv" in filename:
                        coords_input = filename
                except FileNotFoundError:
                    print(
                        "Please ensure that an output video and corresponding datafile from DeepLabCut are in the folder!"
                    )

            cap = cv2.VideoCapture(output_video_name)
            i = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(
                    os.path.join(video_output_path, "{}.png".format(str(i))), frame
                )
                i += 1
        else:
            coords_input = coords_input_file

        os.chdir(video_output_path)

        print("Landmark prediction complete!")
        if not atlas:
            atlasBrainMatch(
                input_file,
                sensory_img_dir,
                coords_input,
                sensory_match,
                mat_save,
                threshold,
                git_repo_base,
                region_labels,
                landmark_arr,
                use_unet,
                use_dlc,
                atlas_to_brain_align,
                model,
                olfactory_check,
                plot_landmarks,
                align_once,
                original_label,
                use_voxelmorph,
                exist_transform,
                voxelmorph_model,
                template_path,
                flow_path,
            )


def DLCPredictBehavior(config, input_file, output):
    
    video_array = []
    img_array = []
    print(input_file)
    for filename in glob.glob(os.path.join(input_file, "*.mp4")):
        video_array.append(os.path.join(input_file, filename))

    for s in glob.glob(os.path.join(input_file, "*.png")):
        _nsre = re.compile("([0-9]+)")
        return [
            int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)
        ]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    if len(img_array) > 0:
        video_output_path = os.path.join(output, "dlc_output", "behavior")
        video_name = os.path.join(video_output_path, "behavior_video.mp4")
        video_output_path = [video_output_path]
        video_name = [video_name]

        if not os.path.isdir(video_output_path[0]):
            os.mkdir(video_output_path[0])

        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MP4V"), 30, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
    elif len(video_array) > 0:
        video_output_path = video_array
        video_name = video_array

    deeplabcut.analyze_videos(
        config, video_output_path, videotype=".mp4", save_as_csv=True, destfolder=output
    )
    deeplabcut.create_labeled_video(
        config, video_name, filtered=True, destfolder=output
    )
    cv2.destroyAllWindows()


def DLCPrep(project_name, your_name, img_path, output_dir_base, copy_videos_bool=True):
    img_array = []
    filenames = glob.glob(os.path.join(img_path, "*.png"))
    filenames.sort(key=natural_sort_key)
    size = (512, 512)
    for filename in filenames:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    if len(img_array) > 0:
        if not os.path.isdir(os.path.join(output_dir_base, "img_for_label")):
            os.mkdir(os.path.join(output_dir_base, "img_for_label"))
        video_output_path = os.path.join(output_dir_base, "img_for_label")
        video_name = os.path.join(video_output_path, "video_for_label.mp4")

        if not os.path.isdir(video_output_path):
            os.mkdir(video_output_path)
        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MP4V"), 30, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

        config_path = deeplabcut.create_new_project(
            project_name,
            your_name,
            [video_name],
            copy_videos=copy_videos_bool,
            working_directory=output_dir_base,
        )
        return config_path


def DLCLabel(config_path):
    
    deeplabcut.extract_frames(config_path, crop=False)
    deeplabcut.label_frames(config_path)
    deeplabcut.check_labels(config_path)


def DLCTrain(config_path, displayiters, saveiters, maxiters):
    
    deeplabcut.create_training_dataset(config_path)
    deeplabcut.train_network(
        config_path, displayiters=displayiters, saveiters=saveiters, maxiters=maxiters
    )


def DLC_edit_bodyparts(config_path, new_bodyparts):
    
    dlc_cfg = read_config(config_path)
    dlc_cfg["bodyparts"] = new_bodyparts
    write_config(config_path, dlc_cfg)


def predict_dlc(config_file):
    
    cwd = os.getcwd()
    cfg = parse_yaml(config_file)
    config = cfg["config"]
    atlas = cfg["atlas"]
    sensory_match = cfg["sensory_match"]
    sensory_path = cfg["sensory_path"]
    input_file = cfg["input_file"]
    output = cfg["output"]
    mat_save = cfg["mat_save"]
    threshold = cfg["threshold"]
    git_repo_base = cfg["git_repo_base"]
    region_labels = cfg["region_labels"]
    landmark_arr = cfg["landmark_arr"]
    use_unet = cfg["use_unet"]
    use_dlc = cfg["use_dlc"]
    atlas_to_brain_align = cfg["atlas_to_brain_align"]
    model = os.path.join(cwd, cfg["model"])
    olfactory_check = cfg["olfactory_check"]
    plot_landmarks = cfg["plot_landmarks"]
    align_once = cfg["align_once"]
    original_label = cfg["original_label"]
    use_voxelmorph = cfg["use_voxelmorph"]
    exist_transform = cfg["exist_transform"]
    voxelmorph_model = cfg["voxelmorph_model"]
    template_path = cfg["template_path"]
    flow_path = cfg["flow_path"]
    coords_input_file = cfg["coords_input_file"]
    DLCPredict(
        config,
        input_file,
        output,
        atlas,
        sensory_match,
        sensory_path,
        mat_save,
        threshold,
        git_repo_base,
        region_labels,
        landmark_arr,
        use_unet,
        use_dlc,
        atlas_to_brain_align,
        model,
        olfactory_check,
        plot_landmarks,
        align_once,
        original_label,
        use_voxelmorph,
        exist_transform,
        voxelmorph_model,
        template_path,
        flow_path,
        coords_input_file,
    )

```

## File: D:\MyCode\MesoNet\MesoNet\mesonet\gui_start.py

- Extension: .py
- Language: python
- Size: 1251 bytes
- Created: 2024-11-08 12:56:16
- Modified: 2024-11-08 12:56:16

### Code

```python

from mesonet import gui_test, gui_train


def gui_start(gui_type="test", git_repo="", config_file=""):
    
    if gui_type == "test":
        gui_test.gui(git_repo, config_file)
    elif gui_type == "train":
        gui_train.gui()


if __name__ == "__main__":
    gui_start()

```

## File: D:\MyCode\MesoNet\MesoNet\mesonet\gui_test.py

- Extension: .py
- Language: python
- Size: 56062 bytes
- Created: 2024-11-08 12:56:16
- Modified: 2024-11-08 12:56:16

### Code

```python

import fnmatch
import glob
import os
from tkinter import *  
from tkinter import filedialog

from PIL import Image, ImageTk
import imageio
import threading

from mesonet.dlc_predict import DLCPredict, DLCPredictBehavior
from mesonet.predict_regions import predictRegion
from mesonet.utils import (
    config_project,
    find_git_repo,
    natural_sort_key,
    convert_to_png,
    parse_yaml,
)


class Gui(object):
    

    def __init__(self, git_repo, config_file):
        
        self.existTransformCheck = None
        self.flowEntryBox = None
        self.flowEntryButton = None
        self.flowName_str = None
        self.flowEntryLabel = None
        self.templateEntryBox = None
        self.templateEntryButton = None
        self.templateName_str = None
        self.templateEntryLabel = None
        self.vxmModelListBox = None
        self.vxmModelLabel = None
        self.vxm_model_select = None
        self.vxm_window = None
        self.root = Tk()
        self.root.resizable(False, False)

        if config_file:
            self.cwd = os.getcwd()
            cfg = parse_yaml(config_file)
            self.atlas = cfg["atlas"]
            self.sensory_align = cfg["sensory_match"]
            self.sensoryName = cfg["sensory_path"]
            self.folderName = cfg["input_file"]
            self.saveFolderName = cfg["output"]
            self.mat_save = cfg["mat_save"]
            self.threshold = cfg["threshold"]
            self.git_repo_base = cfg["git_repo_base"]
            self.region_labels = cfg["region_labels"]
            self.landmark_arr = cfg["landmark_arr"]
            self.unet_select = cfg["use_unet"]
            self.dlc_select = cfg["use_dlc"]
            self.atlas_to_brain_align = cfg["atlas_to_brain_align"]
            self.model = cfg["model"]
            self.olfactory_check = cfg["olfactory_check"]
            self.plot_landmarks = cfg["plot_landmarks"]
            self.align_once = cfg["align_once"]
            self.original_label = cfg["original_label"]
            self.vxm_select = cfg["use_voxelmorph"]
            self.exist_transform = cfg["exist_transform"]
            self.vxm_model = cfg["voxelmorph_model"]
            self.templateName = cfg["template_path"]
            self.flowName = cfg["flow_path"]
        else:
            self.cwd = os.getcwd()
            self.folderName = self.cwd
            self.sensoryName = self.cwd
            self.saveFolderName = self.cwd
            self.threshold = 0.01  
            self.vxm_model = "motif_model_atlas.h5"
            self.flowName = self.cwd
            self.landmark_arr = []

        self.BFolderName = self.cwd
        self.saveBFolderName = self.cwd

        self.j = -1
        self.delta = 0
        self.imgDisplayed = 0
        self.picLen = 0
        self.imageFileName = ""
        self.model = "unet_bundary.hdf5"
        self.status = 'Please select a folder with brain images at "Input Folder".'
        self.status_str = StringVar(self.root, value=self.status)
        self.haveMasks = False
        self.imgDisplayed = 0

        self.config_dir = "dlc"
        self.model_dir = "models"

        if git_repo == "" and not config_file:
            self.git_repo_base = find_git_repo()
        else:
            self.git_repo_base = os.path.join(git_repo, "mesonet")
        self.config_path = os.path.join(
            self.git_repo_base, self.config_dir, "atlas-DongshengXiao-2020-08-03", "config.yaml"
        )
        self.behavior_config_path = os.path.join(
            self.git_repo_base, self.config_dir, "behavior", " config.yaml"
        )
        self.model_top_dir = os.path.join(self.git_repo_base, self.model_dir)
        self.templateName = os.path.join(self.git_repo_base, "atlases", "templates")

        self.Title = self.root.title("MesoNet Analyzer")

        self.canvas = Canvas(self.root, width=512, height=512)
        self.canvas.grid(row=8, column=0, columnspan=4, rowspan=15, sticky=N + S + W)

        
        self.modelSelect = []
        for file in os.listdir(self.model_top_dir):
            if fnmatch.fnmatch(file, "*.hdf5"):
                self.modelSelect.append(file)

        self.modelLabel = Label(
            self.root,
            text="If using U-net, select a model to analyze the brain regions:",
        )
        self.modelListBox = Listbox(self.root, exportselection=0)
        self.modelLabel.grid(row=0, column=4, columnspan=5, sticky=W + E + S)
        self.modelListBox.grid(
            row=1, rowspan=4, column=4, columnspan=5, sticky=W + E + N
        )
        for item in self.modelSelect:
            self.modelListBox.insert(END, item)

        if len(self.modelSelect) > 0:
            self.modelListBox.bind("<<ListboxSelect>>", self.onSelect)

        
        self.fileEntryLabel = Label(self.root, text="Input folder")
        self.fileEntryLabel.grid(row=0, column=0, sticky=E + W)
        self.fileEntryButton = Button(
            self.root, text="Browse...", command=lambda: self.OpenFile(0)
        )

        self.folderName_str = StringVar(self.root, value=self.folderName)
        self.fileEntryButton.grid(row=0, column=2, sticky=E)
        self.fileEntryBox = Entry(self.root, textvariable=self.folderName_str, width=50)
        self.fileEntryBox.grid(row=0, column=1, padx=5, pady=5)

        self.fileSaveLabel = Label(self.root, text="Save folder")
        self.fileSaveLabel.grid(row=1, column=0, sticky=E + W)
        self.fileSaveButton = Button(
            self.root, text="Browse...", command=lambda: self.OpenFile(1)
        )

        self.saveFolderName_str = StringVar(self.root, value=self.saveFolderName)
        self.fileSaveButton.grid(row=1, column=2, sticky=E)
        self.fileSaveBox = Entry(
            self.root, textvariable=self.saveFolderName_str, width=50
        )
        self.fileSaveBox.grid(row=1, column=1, padx=5, pady=5)

        self.sensoryEntryLabel = Label(self.root, text="Sensory map folder")
        self.sensoryEntryLabel.grid(row=2, column=0, sticky=E + W)
        self.sensoryEntryButton = Button(
            self.root, text="Browse...", command=lambda: self.OpenFile(2)
        )

        self.sensoryName_str = StringVar(self.root, value=self.sensoryName)
        self.sensoryEntryButton.grid(row=2, column=2, sticky=E)
        self.sensoryEntryBox = Entry(
            self.root, textvariable=self.sensoryName_str, width=50
        )
        self.sensoryEntryBox.grid(row=2, column=1, padx=5, pady=5)

        self.configDLCLabel = Label(self.root, text="DLC config folder")
        self.configDLCLabel.grid(row=3, column=0, sticky=E + W)
        self.configDLCButton = Button(
            self.root, text="Browse...", command=lambda: self.OpenFile(3)
        )

        self.configDLCName_str = StringVar(self.root, value=self.config_path)
        self.configDLCButton.grid(row=3, column=2, sticky=E)
        self.configDLCEntryBox = Entry(
            self.root, textvariable=self.configDLCName_str, width=50
        )
        self.configDLCEntryBox.grid(row=3, column=1, padx=5, pady=5)

        self.gitLabel = Label(self.root, text="MesoNet git repo folder")
        self.gitLabel.grid(row=4, column=0, sticky=E + W)
        self.gitButton = Button(
            self.root, text="Browse...", command=lambda: self.OpenFile(4)
        )

        self.git_str = StringVar(self.root, value=self.git_repo_base)
        self.gitButton.grid(row=4, column=2, sticky=E)
        self.gitEntryBox = Entry(self.root, textvariable=self.git_str, width=50)
        self.gitEntryBox.grid(row=4, column=1, padx=5, pady=5)

        
        self.BfileEntryLabel = Label(self.root, text="Behavior input folder")
        self.BfileEntryLabel.grid(row=5, column=0, sticky=E + W)
        self.BfileEntryButton = Button(
            self.root, text="Browse...", command=lambda: self.OpenBFile(0)
        )

        self.BfolderName_str = StringVar(self.root, value=self.folderName)
        self.BfileEntryButton.grid(row=5, column=2, sticky=E)
        self.BfileEntryBox = Entry(
            self.root, textvariable=self.BfolderName_str, width=50
        )
        self.BfileEntryBox.grid(row=5, column=1, padx=5, pady=5)

        self.BfileSaveLabel = Label(self.root, text="Behavior Save folder")
        self.BfileSaveLabel.grid(row=6, column=0, sticky=E + W)
        self.BfileSaveButton = Button(
            self.root, text="Browse...", command=lambda: self.OpenBFile(1)
        )

        self.saveBFolderName_str = StringVar(self.root, value=self.saveBFolderName)
        self.BfileSaveButton.grid(row=6, column=2, sticky=E)
        self.BfileSaveBox = Entry(
            self.root, textvariable=self.saveBFolderName_str, width=50
        )
        self.BfileSaveBox.grid(row=6, column=1, padx=5, pady=5)

        
        
        if not config_file:
            self.mat_save = BooleanVar()
            self.mat_save.set(True)
            self.atlas = BooleanVar()
            self.atlas.set(False)
            self.sensory_align = BooleanVar()
            self.sensory_align.set(False)
            self.region_labels = BooleanVar()
            self.region_labels.set(False)
            self.unet_select = BooleanVar()
            self.unet_select.set(True)
            self.dlc_select = BooleanVar()
            self.dlc_select.set(True)
            self.vxm_select = BooleanVar()
            self.vxm_select.set(True)
            self.olfactory_check = BooleanVar()
            self.olfactory_check.set(True)
            self.atlas_to_brain_align = BooleanVar()
            self.atlas_to_brain_align.set(True)
            self.plot_landmarks = BooleanVar()
            self.plot_landmarks.set(True)
            self.align_once = BooleanVar()
            self.align_once.set(False)
            self.original_label = BooleanVar()
            self.original_label.set(False)
            self.exist_transform = BooleanVar()
            self.exist_transform.set(False)

        self.landmark_left = BooleanVar()
        self.landmark_left.set(True)
        self.landmark_right = BooleanVar()
        self.landmark_right.set(True)
        self.landmark_bregma = BooleanVar()
        self.landmark_bregma.set(True)
        self.landmark_lambda = BooleanVar()
        self.landmark_lambda.set(True)

        self.landmark_top_left = BooleanVar()
        self.landmark_top_left.set(True)
        self.landmark_top_centre = BooleanVar()
        self.landmark_top_centre.set(True)
        self.landmark_top_right = BooleanVar()
        self.landmark_top_right.set(True)
        self.landmark_bottom_left = BooleanVar()
        self.landmark_bottom_left.set(True)
        self.landmark_bottom_right = BooleanVar()
        self.landmark_bottom_right.set(True)

        self.saveMatFileCheck = Checkbutton(
            self.root,
            text="Save predicted regions as .mat files",
            variable=self.mat_save,
            onvalue=True,
            offvalue=False,
        )
        self.saveMatFileCheck.grid(
            row=7, column=4, columnspan=5, padx=2, sticky=N + S + W
        )
        
        
        
        self.uNetCheck = Checkbutton(
            self.root,
            text="Use U-net for alignment",
            variable=self.unet_select,
            onvalue=True,
            offvalue=False,
        )
        self.uNetCheck.grid(row=8, column=4, columnspan=5, padx=2, sticky=N + S + W)

        self.dlcCheck = Checkbutton(
            self.root,
            text="Use DeepLabCut for alignment",
            variable=self.dlc_select,
            onvalue=True,
            offvalue=False,
        )
        self.dlcCheck.grid(row=9, column=4, columnspan=5, padx=2, sticky=N + S + W)

        self.vxmCheck = Checkbutton(
            self.root,
            text="Use VoxelMorph for alignment",
            variable=self.vxm_select,
            onvalue=True,
            offvalue=False,
        )
        self.vxmCheck.grid(row=10, column=4, columnspan=5, padx=2, sticky=N + S + W)

        self.olfactoryCheck = Checkbutton(
            self.root,
            text="Draw olfactory bulbs\n(uncheck if no olfactory bulb visible "
            "in all images)",
            variable=self.olfactory_check,
            onvalue=True,
            offvalue=False,
        )
        self.olfactoryCheck.grid(
            row=11, column=4, columnspan=5, padx=2, sticky=N + S + W
        )

        self.atlasToBrainCheck = Checkbutton(
            self.root,
            text="Align atlas to brain",
            variable=self.atlas_to_brain_align,
            onvalue=True,
            offvalue=False,
        )
        self.atlasToBrainCheck.grid(
            row=12, column=4, columnspan=5, padx=2, sticky=N + S + W
        )

        self.sensoryMapCheck = Checkbutton(
            self.root,
            text="Align using sensory map",
            variable=self.sensory_align,
            onvalue=True,
            offvalue=False,
        )
        self.sensoryMapCheck.grid(
            row=13, column=4, columnspan=5, padx=2, sticky=N + S + W
        )

        self.landmarkPlotCheck = Checkbutton(
            self.root,
            text="Plot DLC landmarks on final image",
            variable=self.plot_landmarks,
            onvalue=True,
            offvalue=False,
        )
        self.landmarkPlotCheck.grid(
            row=14, column=4, columnspan=5, padx=2, sticky=N + S + W
        )

        self.alignOnceCheck = Checkbutton(
            self.root,
            text="Align based on first brain image only",
            variable=self.align_once,
            onvalue=True,
            offvalue=False,
        )
        self.alignOnceCheck.grid(
            row=15, column=4, columnspan=5, padx=2, sticky=N + S + W
        )

        self.origLabelCheck = Checkbutton(
            self.root,
            text="Use old label consistency method\n(less consistent)",
            variable=self.original_label,
            onvalue=True,
            offvalue=False,
        )
        self.origLabelCheck.grid(
            row=16, column=4, columnspan=5, padx=2, sticky=N + S + W
        )

        
        self.landmarkLeftCheck = Checkbutton(
            self.root,
            text="Left",
            variable=self.landmark_left,
            onvalue=True,
            offvalue=False,
        )
        self.landmarkLeftCheck.grid(row=17, column=4, padx=2, sticky=N + S + W)
        self.landmarkRightCheck = Checkbutton(
            self.root,
            text="Right",
            variable=self.landmark_right,
            onvalue=True,
            offvalue=False,
        )
        self.landmarkRightCheck.grid(row=17, column=5, padx=2, sticky=N + S + W)
        self.landmarkBregmaCheck = Checkbutton(
            self.root,
            text="Bregma",
            variable=self.landmark_bregma,
            onvalue=True,
            offvalue=False,
        )
        self.landmarkBregmaCheck.grid(row=17, column=6, padx=2, sticky=N + S + W)
        self.landmarkLambdaCheck = Checkbutton(
            self.root,
            text="Lambda",
            variable=self.landmark_lambda,
            onvalue=True,
            offvalue=False,
        )
        self.landmarkLambdaCheck.grid(row=17, column=7, padx=2, sticky=N + S + W)

        self.landmarkTopLeftCheck = Checkbutton(
            self.root,
            text="Top left",
            variable=self.landmark_top_left,
            onvalue=True,
            offvalue=False,
        )
        self.landmarkTopLeftCheck.grid(row=18, column=4, padx=2, sticky=N + S + W)
        self.landmarkTopCentreCheck = Checkbutton(
            self.root,
            text="Top centre",
            variable=self.landmark_top_centre,
            onvalue=True,
            offvalue=False,
        )
        self.landmarkTopCentreCheck.grid(row=18, column=5, padx=2, sticky=N + S + W)
        self.landmarkTopRightCheck = Checkbutton(
            self.root,
            text="Top right",
            variable=self.landmark_top_right,
            onvalue=True,
            offvalue=False,
        )
        self.landmarkTopRightCheck.grid(row=18, column=6, padx=2, sticky=N + S + W)
        self.landmarkBottomLeftCheck = Checkbutton(
            self.root,
            text="Bottom left",
            variable=self.landmark_bottom_left,
            onvalue=True,
            offvalue=False,
        )
        self.landmarkBottomLeftCheck.grid(row=18, column=7, padx=2, sticky=N + S + W)
        self.landmarkBottomRightCheck = Checkbutton(
            self.root,
            text="Bottom right",
            variable=self.landmark_bottom_right,
            onvalue=True,
            offvalue=False,
        )
        self.landmarkBottomRightCheck.grid(row=18, column=8, padx=2, sticky=N + S + W)

        self.vxm_window_open = False
        if self.vxm_window_open:
            print("TEST")

        self.vxmSettingsButton = Button(
            self.root,
            text="Open VoxelMorph settings",
            command=lambda: self.open_voxelmorph_window(),
        )
        self.vxmSettingsButton.grid(
            row=19, column=4, columnspan=5, padx=2, sticky=N + S + W + E
        )

        self.predictDLCButton = Button(
            self.root,
            text="Predict brain regions\nusing landmarks",
            command=lambda: self.EnterThread("predict_dlc"),
        )
        self.predictDLCButton.grid(
            row=20, column=4, columnspan=5, padx=2, sticky=N + S + W + E
        )

        self.predictAllImButton = Button(
            self.root,
            text="Predict brain regions directly\nusing pretrained U-net model",
            command=lambda: self.EnterThread("predict_regions"),
        )
        self.predictAllImButton.grid(
            row=21, column=4, columnspan=5, padx=2, sticky=N + S + W + E
        )
        self.predictBehaviourButton = Button(
            self.root,
            text="Predict animal movements",
            command=lambda: DLCPredictBehavior(
                self.behavior_config_path, self.BFolderName, self.saveBFolderName
            ),
        )
        self.predictBehaviourButton.grid(
            row=22, column=4, columnspan=5, padx=2, sticky=N + S + W + E
        )

        
        
        self.nextButton = Button(
            self.root,
            text="->",
            command=lambda: self.ImageDisplay(1, self.folderName, 0),
        )
        self.nextButton.grid(row=23, column=2, columnspan=2, sticky=E)
        self.previousButton = Button(
            self.root,
            text="<-",
            command=lambda: self.ImageDisplay(-1, self.folderName, 0),
        )
        self.previousButton.grid(row=23, column=0, columnspan=2, sticky=W)

        self.statusBar = Label(
            self.root, textvariable=self.status_str, bd=1, relief=SUNKEN, anchor=W
        )
        self.statusBar.grid(row=24, column=0, columnspan=9, sticky="we")

        
        self.root.bind("<Right>", self.forward)
        self.root.bind("<Left>", self.backward)

        
        
        self.pipelinesLabel = Label(self.root, text="Quick Start: Automated pipelines")

        self.pipelinesLabel.grid(
            row=0, column=9, columnspan=1, padx=2, sticky=N + S + W + E
        )

        
        self.atlasToBrainButton = Button(
            self.root,
            text="1 - Atlas to brain",
            command=lambda: self.EnterThread("atlas_to_brain"),
        )
        self.atlasToBrainButton.grid(
            row=1, column=9, columnspan=1, padx=2, sticky=N + S + W + E
        )

        
        self.brainToAtlasButton = Button(
            self.root,
            text="2 - Brain to atlas",
            command=lambda: self.EnterThread("brain_to_atlas"),
        )
        self.brainToAtlasButton.grid(
            row=2, column=9, columnspan=1, padx=2, sticky=N + S + W + E
        )

        
        self.atlasToBrainSensoryButton = Button(
            self.root,
            text="3 - Atlas to brain +\nsensory maps",
            command=lambda: self.EnterThread("atlas_to_brain_sensory"),
        )
        self.atlasToBrainSensoryButton.grid(
            row=3, column=9, columnspan=1, padx=2, sticky=N + S + W + E
        )

        
        self.MBFMUNetButton = Button(
            self.root,
            text="4 - Motif-based functional maps (MBFMs) +\nMBFM-U-Net",
            command=lambda: self.EnterThread("MBFM_U_Net"),
        )
        self.MBFMUNetButton.grid(
            row=4, column=9, columnspan=1, padx=2, sticky=N + S + W + E
        )

        
        self.MBFMBrainToAtlasVxmButton = Button(
            self.root,
            text="5 - Motif-based functional maps (MBFMs) +\nBrain-to-atlas + VoxelMorph",
            command=lambda: self.EnterThread("MBFM_brain_to_atlas_vxm"),
        )
        self.MBFMBrainToAtlasVxmButton.grid(
            row=5, column=9, columnspan=1, padx=2, sticky=N + S + W + E
        )

        if self.saveFolderName == "" or self.imgDisplayed == 0:
            self.predictAllImButton.config(state="disabled")
            self.predictDLCButton.config(state="disabled")
            self.saveMatFileCheck.config(state="disabled")
            
            self.uNetCheck.config(state="disabled")
            self.dlcCheck.config(state="disabled")
            self.vxmCheck.config(state="disabled")
            self.olfactoryCheck.config(state="disabled")
            self.sensoryMapCheck.config(state="disabled")
            self.atlasToBrainCheck.config(state="disabled")
            self.predictBehaviourButton.config(state="disabled")
            self.landmarkPlotCheck.config(state="disabled")
            self.alignOnceCheck.config(state="disabled")
            self.origLabelCheck.config(state="disabled")

            self.landmarkLeftCheck.config(state="disabled")
            self.landmarkRightCheck.config(state="disabled")
            self.landmarkBregmaCheck.config(state="disabled")
            self.landmarkLambdaCheck.config(state="disabled")

            self.landmarkTopLeftCheck.config(state="disabled")
            self.landmarkTopCentreCheck.config(state="disabled")
            self.landmarkTopRightCheck.config(state="disabled")
            self.landmarkBottomLeftCheck.config(state="disabled")
            self.landmarkBottomRightCheck.config(state="disabled")

            self.atlasToBrainButton.config(state="disabled")
            self.brainToAtlasButton.config(state="disabled")
            self.atlasToBrainSensoryButton.config(state="disabled")
            self.MBFMUNetButton.config(state="disabled")
            self.MBFMBrainToAtlasVxmButton.config(state="disabled")

        if config_file:
            self.ImageDisplay(1, self.folderName, 1)
            self.predictAllImButton.config(state="normal")
            self.predictDLCButton.config(state="normal")
            self.saveMatFileCheck.config(state="normal")
            
            self.uNetCheck.config(state="normal")
            self.dlcCheck.config(state="normal")
            self.vxmCheck.config(state="normal")
            self.olfactoryCheck.config(state="normal")
            self.atlasToBrainCheck.config(state="normal")
            self.sensoryMapCheck.config(state="normal")
            self.landmarkPlotCheck.config(state="normal")
            self.alignOnceCheck.config(state="normal")
            self.origLabelCheck.config(state="normal")

            self.landmarkLeftCheck.config(state="normal")
            self.landmarkRightCheck.config(state="normal")
            self.landmarkBregmaCheck.config(state="normal")
            self.landmarkLambdaCheck.config(state="normal")

            self.landmarkTopLeftCheck.config(state="normal")
            self.landmarkTopCentreCheck.config(state="normal")
            self.landmarkTopRightCheck.config(state="normal")
            self.landmarkBottomLeftCheck.config(state="normal")
            self.landmarkBottomRightCheck.config(state="normal")

    def OpenFile(self, openOrSave):
        if openOrSave == 0:
            newFolderName = filedialog.askdirectory(
                initialdir=self.cwd,
                title="Choose folder containing the brain images you want to "
                "analyze",
            )
            
            try:
                self.folderName_str.set(newFolderName)
                self.folderName = newFolderName
                self.ImageDisplay(1, self.folderName, 1)
                self.statusHandler(
                    'Please select a folder to save your images to at "Save Folder".'
                )
            except:
                if self.folderName_str.get != newFolderName:
                    self.folderName_str.set(self.cwd)
                img_path_err = "No image file selected!"
                self.statusHandler(img_path_err)
        elif openOrSave == 1:
            newSaveFolderName = filedialog.askdirectory(
                initialdir=self.cwd, title="Choose folder for saving files"
            )
            
            try:
                self.saveFolderName_str.set(newSaveFolderName)
                self.saveFolderName = newSaveFolderName
                self.predictAllImButton.config(state="normal")
                self.predictDLCButton.config(state="normal")
                self.saveMatFileCheck.config(state="normal")
                
                self.uNetCheck.config(state="normal")
                self.dlcCheck.config(state="normal")
                self.vxmCheck.config(state="normal")
                self.olfactoryCheck.config(state="normal")
                self.atlasToBrainCheck.config(state="normal")
                self.sensoryMapCheck.config(state="normal")
                self.landmarkPlotCheck.config(state="normal")
                self.alignOnceCheck.config(state="normal")
                self.origLabelCheck.config(state="normal")

                self.landmarkLeftCheck.config(state="normal")
                self.landmarkRightCheck.config(state="normal")
                self.landmarkBregmaCheck.config(state="normal")
                self.landmarkLambdaCheck.config(state="normal")

                self.landmarkTopLeftCheck.config(state="normal")
                self.landmarkTopCentreCheck.config(state="normal")
                self.landmarkTopRightCheck.config(state="normal")
                self.landmarkBottomLeftCheck.config(state="normal")
                self.landmarkBottomRightCheck.config(state="normal")

                self.atlasToBrainButton.config(state="normal")
                self.brainToAtlasButton.config(state="normal")
                self.atlasToBrainSensoryButton.config(state="normal")
                self.MBFMUNetButton.config(state="normal")
                self.MBFMBrainToAtlasVxmButton.config(state="normal")

                self.statusHandler(
                    "Save folder selected! Choose an option on the right to begin your analysis."
                )
            except:
                if self.saveFolderName_str.get != newSaveFolderName:
                    self.saveFolderName_str.set(self.cwd)
                save_path_err = "No save file selected!"
                print(save_path_err)
                self.statusHandler(save_path_err)
        elif openOrSave == 2:
            newSensoryName = filedialog.askdirectory(
                initialdir=self.cwd,
                title="Choose folder containing the sensory images you want to use",
            )
            try:
                self.sensoryName_str.set(newSensoryName)
                self.sensoryName = newSensoryName
                self.root.update()
            except:
                if self.sensoryName_str.get != newSensoryName:
                    self.sensoryName_str.set(self.cwd)
                sensory_path_err = "No sensory image file selected!"
                print(sensory_path_err)
                self.statusHandler(sensory_path_err)
        elif openOrSave == 3:
            newDLCName = filedialog.askopenfilename(
                initialdir=self.cwd,
                title="Choose folder containing the DLC config file",
            )
            try:
                self.configDLCName_str.set(newDLCName)
                self.config_path = newDLCName
                self.root.update()
            except:
                dlc_path_err = "No DLC config file selected!"
                self.statusHandler(dlc_path_err)
        elif openOrSave == 4:
            newGitName = filedialog.askdirectory(
                initialdir=self.cwd, title="Choose MesoNet git repository"
            )
            try:
                newGitName = os.path.join(newGitName, "mesonet")
                self.git_str.set(newGitName)
                self.git_repo_base = newGitName
                self.config_path = os.path.join(
                    self.git_repo_base, self.config_dir, "config.yaml"
                )
                self.configDLCName_str.set(self.config_path)
                self.behavior_config_path = os.path.join(
                    self.git_repo_base, self.config_dir, "behavior", " config.yaml"
                )
                self.model_top_dir = os.path.join(self.git_repo_base, self.model_dir)
                self.modelSelect = []
                for file in os.listdir(self.model_top_dir):
                    if fnmatch.fnmatch(file, "*.hdf5"):
                        self.modelSelect.append(file)
                self.modelListBox.delete(0, END)
                for item in self.modelSelect:
                    self.modelListBox.insert(END, item)
                self.root.update()
            except:
                dlc_path_err = "No git repo selected!"
                self.statusHandler(dlc_path_err)
        elif openOrSave == 5:
            newVxmTemplateName = filedialog.askdirectory(
                initialdir=self.cwd, title="Select VoxelMorph template directory"
            )
            try:
                self.templateName_str.set(newVxmTemplateName)
                self.templateName = newVxmTemplateName
            except:
                template_err = "No template folder selected!"
                self.statusHandler(template_err)
        elif openOrSave == 6:
            newVxmFlowName = filedialog.askopenfilename(
                initialdir=self.cwd, title="Select VoxelMorph flow file"
            )
            try:
                self.flowName_str.set(newVxmFlowName)
                self.flowName = newVxmFlowName
            except:
                template_err = "No template file selected!"
                self.statusHandler(template_err)

    def OpenBFile(self, openOrSave):
        if openOrSave == 0:
            newBFolderName = filedialog.askdirectory(
                initialdir=self.cwd,
                title="Choose folder containing the brain images you want to "
                "analyze",
            )
            
            try:
                self.BfolderName_str.set(newBFolderName)
                self.BFolderName = newBFolderName
                self.root.update()
            except:
                print("No image file selected!")

        elif openOrSave == 1:
            newSaveBFolderName = filedialog.askdirectory(
                initialdir=self.cwd, title="Choose folder for saving files"
            )
            
            try:
                self.saveBFolderName_str.set(newSaveBFolderName)
                self.saveBFolderName = newSaveBFolderName
                self.predictBehaviourButton.config(state="normal")
                self.statusHandler(
                    'Save folder selected! Click "Predict animal movements" to begin your analysis.'
                )
            except:
                save_path_err = "No save file selected!"
                print(save_path_err)
                self.statusHandler(save_path_err)

    def open_voxelmorph_window(self):
        
        self.vxm_window_open = True
        self.vxm_window = Toplevel(self.root)
        self.vxm_window.title("VoxelMorph settings")
        self.vxm_window.resizable(False, False)

        
        self.vxm_model_select = []
        for file in os.listdir(os.path.join(self.model_top_dir, "voxelmorph")):
            if fnmatch.fnmatch(file, "*.h5"):
                self.vxm_model_select.append(file)

        self.vxmModelLabel = Label(
            self.vxm_window,
            text="If using VoxelMorph, select a model to align the brain image and atlas:",
        )
        self.vxmModelListBox = Listbox(self.vxm_window, exportselection=0)
        self.vxmModelLabel.grid(row=0, column=4, columnspan=5, sticky=W + E + S)
        self.vxmModelListBox.grid(
            row=1, rowspan=4, column=4, columnspan=5, sticky=W + E + N
        )
        for item in self.vxm_model_select:
            self.vxmModelListBox.insert(END, item)

        if len(self.vxm_model_select) > 0:
            self.vxmModelListBox.bind(
                "<<ListboxSelect>>", lambda event: self.onSelectVxm(event)
            )

        self.templateEntryLabel = Label(self.vxm_window, text="Template file location")
        self.templateEntryLabel.grid(row=0, column=0, sticky=E + W)

        self.templateName_str = StringVar(self.vxm_window, value=self.templateName)
        self.templateEntryButton = Button(
            self.vxm_window, text="Browse...", command=lambda: self.OpenFile(5)
        )

        self.templateEntryButton.grid(row=0, column=2, sticky=E)
        self.templateEntryBox = Entry(
            self.vxm_window, textvariable=self.templateName_str, width=50
        )
        self.templateEntryBox.grid(row=0, column=1, padx=5, pady=5)

        self.flowEntryLabel = Label(self.vxm_window, text="Flow file location")
        self.flowEntryLabel.grid(row=1, column=0, sticky=E + W)

        self.flowName_str = StringVar(self.vxm_window, value=self.flowName)
        self.flowEntryButton = Button(
            self.vxm_window, text="Browse...", command=lambda: self.OpenFile(6)
        )

        self.flowEntryButton.grid(row=1, column=2, sticky=E)
        self.flowEntryBox = Entry(
            self.vxm_window, textvariable=self.flowName_str, width=50
        )
        self.flowEntryBox.grid(row=1, column=1, padx=5, pady=5)

        self.existTransformCheck = Checkbutton(
            self.vxm_window,
            text="Use existing transformation",
            variable=self.exist_transform,
            onvalue=True,
            offvalue=False,
        )
        self.existTransformCheck.grid(
            row=2, column=0, columnspan=2, padx=2, sticky=N + S + W
        )

        self.vxm_window.mainloop()

    def ImageDisplay(self, delta, folderName, reset):
        
        if glob.glob(os.path.join(folderName, "*.mat")) or glob.glob(
            os.path.join(folderName, "*.npy")
        ):
            convert_to_png(folderName)

        
        is_tif = False
        self.imgDisplayed = 1
        self.root.update()
        if reset == 1:
            self.j = -1
        self.j += delta
        file_list = []
        tif_list = []
        if glob.glob(os.path.join(folderName, "*_mask_segmented.png")):
            file_list = glob.glob(os.path.join(folderName, "*_mask_segmented.png"))
            file_list.sort(key=natural_sort_key)
        elif glob.glob(os.path.join(folderName, "*_mask.png")):
            file_list = glob.glob(os.path.join(folderName, "*_mask.png"))
            file_list.sort(key=natural_sort_key)
        elif glob.glob(os.path.join(folderName, "*.png")):
            file_list = glob.glob(os.path.join(folderName, "*.png"))
            file_list.sort(key=natural_sort_key)
        elif glob.glob(os.path.join(folderName, "*.tif")):
            is_tif = True
            tif_list = glob.glob(os.path.join(folderName, "*.tif"))
            tif_stack = imageio.mimread(tif_list[0])
            file_list = tif_stack
        self.picLen = len(file_list)
        if self.j > self.picLen - 1:
            self.j = 0
        if self.j <= -1:
            self.j = self.picLen - 1
        if delta != 0:
            if is_tif:
                image_orig = Image.fromarray(file_list[0])
                self.imageFileName = tif_list[0]
                self.imageFileName = os.path.basename(self.imageFileName)
            else:
                self.imageFileName = os.path.basename(file_list[self.j])
                image = os.path.join(folderName, file_list[self.j])
                image_orig = Image.open(image)
            image_resize = image_orig.resize((512, 512))
            image_disp = ImageTk.PhotoImage(image_resize)
            self.canvas.create_image(256, 256, image=image_disp)
            label = Label(image=image_disp)
            label.image = image_disp
            self.root.update()
        imageName = StringVar(self.root, value=self.imageFileName)
        imageNum = "Image {}/{}".format(self.j + 1, self.picLen)
        imageNumPrep = StringVar(self.root, value=imageNum)
        imageNameLabel = Label(self.root, textvariable=imageName)
        imageNameLabel.grid(row=7, column=0, columnspan=2, sticky=W)
        imageNumLabel = Label(self.root, textvariable=imageNumPrep)
        imageNumLabel.grid(row=7, column=2, columnspan=2, sticky=E)

    def onSelect(self, event):
        w = event.widget
        selected = int(w.curselection()[0])
        new_model = self.modelListBox.get(selected)
        self.model = new_model
        print("Model selected: {}".format(self.model))
        self.root.update()

    def onSelectVxm(self, event):
        w_vxm = event.widget
        selected_vxm = int(w_vxm.curselection()[0])
        new_vxm_model = self.vxmModelListBox.get(selected_vxm)
        self.vxm_model = new_vxm_model
        print("Model selected: {}".format(self.vxm_model))
        self.root.update()

    def forward(self, event):
        self.ImageDisplay(1, self.folderName, 0)

    def backward(self, event):
        self.ImageDisplay(-1, self.folderName, 0)

    def statusHandler(self, status_str):
        self.status = status_str
        self.status_str.set(self.status)
        self.root.update()

    def chooseLandmarks(self):
        left = self.landmark_left.get()
        right = self.landmark_right.get()
        bregma = self.landmark_bregma.get()
        lambd = self.landmark_lambda.get()
        top_left = self.landmark_top_left.get()
        top_centre = self.landmark_top_centre.get()
        top_right = self.landmark_top_right.get()
        bottom_left = self.landmark_bottom_left.get()
        bottom_right = self.landmark_bottom_right.get()

        if left:
            self.landmark_arr.append(0)
        if top_left:
            self.landmark_arr.append(1)
        if bottom_left:
            self.landmark_arr.append(2)
        if top_centre:
            self.landmark_arr.append(3)
        if bregma:
            self.landmark_arr.append(4)
        if lambd:
            self.landmark_arr.append(5)
        if right:
            self.landmark_arr.append(6)
        if top_right:
            self.landmark_arr.append(7)
        if bottom_right:
            self.landmark_arr.append(8)

    def EnterThread(self, command):
        if command == "predict_regions":
            threading.Thread(
                target=self.PredictRegions(
                    self.folderName,
                    self.picLen,
                    self.model,
                    self.saveFolderName,
                    int(self.mat_save.get()),
                    self.threshold,
                    False,
                    self.git_repo_base,
                    self.region_labels.get(),
                    self.olfactory_check.get(),
                    self.unet_select.get(),
                    self.plot_landmarks.get(),
                    self.align_once.get(),
                    self.region_labels.get(),
                )
            ).start()
        elif command == "predict_dlc":
            threading.Thread(
                target=self.PredictDLC(
                    self.config_path,
                    self.folderName,
                    self.saveFolderName,
                    False,
                    int(self.sensory_align.get()),
                    self.sensoryName,
                    os.path.join(
                        self.model_top_dir, "DongshengXiao_brain_bundary.hdf5"
                    ),
                    self.picLen,
                    int(self.mat_save.get()),
                    self.threshold,
                    True,
                    self.haveMasks,
                    self.git_repo_base,
                    self.region_labels.get(),
                    self.unet_select.get(),
                    self.dlc_select.get(),
                    self.atlas_to_brain_align.get(),
                    self.olfactory_check.get(),
                    self.plot_landmarks.get(),
                    self.align_once.get(),
                    self.original_label.get(),
                    self.vxm_select.get(),
                    self.exist_transform.get(),
                    os.path.join(self.model_top_dir, "voxelmorph", self.vxm_model),
                    self.templateName,
                    self.flowName,
                )
            ).start()
        elif command == "atlas_to_brain":
            threading.Thread(
                target=self.PredictDLC(
                    self.config_path,
                    self.folderName,
                    self.saveFolderName,
                    False,
                    0,
                    "",
                    os.path.join(
                        self.model_top_dir, "DongshengXiao_brain_bundary.hdf5"
                    ),
                    self.picLen,
                    True,
                    self.threshold,
                    True,
                    False,
                    self.git_repo_base,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    "",
                    "",
                    "",
                )
            ).start()
        elif command == "brain_to_atlas":
            threading.Thread(
                target=self.PredictDLC(
                    self.config_path,
                    self.folderName,
                    self.saveFolderName,
                    False,
                    0,
                    "",
                    os.path.join(
                        self.model_top_dir, "DongshengXiao_brain_bundary.hdf5"
                    ),
                    self.picLen,
                    True,
                    self.threshold,
                    True,
                    False,
                    self.git_repo_base,
                    False,
                    True,
                    True,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    "",
                    "",
                    "",
                )
            ).start()
        elif command == "atlas_to_brain_sensory":
            threading.Thread(
                target=self.PredictDLC(
                    self.config_path,
                    self.folderName,
                    self.saveFolderName,
                    False,
                    1,
                    self.sensoryName,
                    os.path.join(
                        self.model_top_dir, "DongshengXiao_brain_bundary.hdf5"
                    ),
                    self.picLen,
                    True,
                    self.threshold,
                    True,
                    False,
                    self.git_repo_base,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    "",
                    "",
                    "",
                )
            ).start()
        elif command == "MBFM_U_Net":
            threading.Thread(
                target=self.PredictRegions(
                    self.folderName,
                    self.picLen,
                    os.path.join(
                        self.model_top_dir,
                        "DongshengXiao_unet_motif_based_functional_atlas.hdf5",
                    ),
                    self.saveFolderName,
                    True,
                    self.threshold,
                    False,
                    self.git_repo_base,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                )
            ).start()
        elif command == "MBFM_brain_to_atlas_vxm":
            threading.Thread(
                target=self.PredictDLC(
                    self.config_path,
                    self.folderName,
                    self.saveFolderName,
                    False,
                    0,
                    "",
                    os.path.join(
                        self.model_top_dir, "DongshengXiao_brain_bundary.hdf5"
                    ),
                    self.picLen,
                    True,
                    self.threshold,
                    True,
                    False,
                    self.git_repo_base,
                    False,
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                    False,
                    True,
                    False,
                    os.path.join(
                        self.model_top_dir,
                        "voxelmorph",
                        "VoxelMorph_Motif_based_functional_map_model_transformed1000.h5",
                    ),
                    self.templateName,
                    self.flowName,
                )
            ).start()

    def PredictRegions(
        self,
        input_file,
        num_images,
        model,
        output,
        mat_save,
        threshold,
        mask_generate,
        git_repo_base,
        region_labels,
        olfactory_check,
        use_unet,
        plot_landmarks,
        align_once,
        original_label,
    ):
        self.statusHandler("Processing...")
        atlas_to_brain_align = True
        pts = []
        pts2 = []
        atlas_label_list = []
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
            pts,
            pts2,
            olfactory_check,
            use_unet,
            plot_landmarks,
            align_once,
            atlas_label_list,
            region_labels,
            original_label,
        )
        self.saveFolderName = output
        if mask_generate:
            self.folderName = os.path.join(self.saveFolderName, "output_mask")
            self.haveMasks = True
        else:
            self.folderName = os.path.join(self.saveFolderName, "output_overlay")
        self.statusHandler("Processing complete!")
        self.ImageDisplay(1, self.folderName, 1)

    def PredictDLC(
        self,
        config,
        input_file,
        output,
        atlas,
        sensory_match,
        sensory_path,
        model,
        num_images,
        mat_save,
        threshold,
        mask_generate,
        haveMasks,
        git_repo_base,
        region_labels,
        use_unet,
        use_dlc,
        atlas_to_brain_align,
        olfactory_check,
        plot_landmarks,
        align_once,
        original_label,
        use_voxelmorph,
        exist_transform,
        voxelmorph_model,
        template_path,
        flow_path,
    ):
        self.statusHandler("Processing...")
        self.chooseLandmarks()
        atlas_label_list = []
        coords_input_file = ""
        
        if mask_generate and not haveMasks and use_unet:
            pts = []
            pts2 = []
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
                pts,
                pts2,
                olfactory_check,
                use_unet,
                plot_landmarks,
                align_once,
                atlas_label_list,
                region_labels,
                original_label,
            )
        DLCPredict(
            config,
            input_file,
            output,
            atlas,
            sensory_match,
            sensory_path,
            mat_save,
            threshold,
            git_repo_base,
            region_labels,
            self.landmark_arr,
            use_unet,
            use_dlc,
            atlas_to_brain_align,
            model,
            olfactory_check,
            plot_landmarks,
            align_once,
            original_label,
            use_voxelmorph,
            exist_transform,
            voxelmorph_model,
            template_path,
            flow_path,
            coords_input_file,
        )
        saveFolderName = output
        if not atlas:
            self.folderName = os.path.join(saveFolderName, "output_overlay")
        elif atlas:
            self.folderName = os.path.join(saveFolderName, "dlc_output")
        config_project(
            input_file,
            saveFolderName,
            "test",
            config=config,
            atlas=atlas,
            sensory_match=sensory_match,
            mat_save=mat_save,
            threshold=threshold,
            model=model,
            region_labels=region_labels,
            use_unet=use_unet,
            use_dlc=use_dlc,
            atlas_to_brain_align=atlas_to_brain_align,
            olfactory_check=olfactory_check,
            plot_landmarks=plot_landmarks,
            align_once=align_once,
            atlas_label_list=atlas_label_list,
            original_label=original_label,
            use_voxelmorph=use_voxelmorph,
            exist_transform=exist_transform,
            voxelmorph_model=voxelmorph_model,
            template_path=template_path,
            flow_path=flow_path,
        )
        self.statusHandler("Processing complete!")
        self.ImageDisplay(1, self.folderName, 1)


def gui(git_find, config_file):
    Gui(git_find, config_file).root.mainloop()

```

## File: D:\MyCode\MesoNet\MesoNet\mesonet\gui_train.py

- Extension: .py
- Language: python
- Size: 19098 bytes
- Created: 2024-11-08 12:56:16
- Modified: 2024-11-08 12:56:16

### Code

```python


import fnmatch
import glob
import os
import numpy as np
import skimage.io as io
from tkinter import *  
from tkinter import filedialog

from PIL import Image, ImageTk, ImageDraw

from mesonet.train_model import trainModel
from mesonet.utils import config_project, find_git_repo
from mesonet.dlc_predict import DLCPrep, DLCLabel, DLCTrain, DLC_edit_bodyparts
from mesonet.mask_functions import inpaintMask


class GuiTrain:
    

    DEFAULT_PEN_SIZE = 20
    DEFAULT_COLOUR = "white"
    DEFAULT_MODEL_NAME = "my_unet.hdf5"
    DEFAULT_TASK = "MesoNet"
    DEFAULT_NAME = "Labeler"

    def __init__(self):
        self.old_y = None
        self.old_x = None
        self.image_resize = None
        self.imageFileName = None
        self.j = None
        self.picLen = None
        self.imgDisplayed = None
        self.root_train = Tk()
        self.root_train.resizable(False, False)
        self.Title = self.root_train.title("MesoNet Trainer")

        self.status = 'Please select a folder with brain images at "Input Folder".'
        self.status_str = StringVar(self.root_train, value=self.status)

        self.cwd = os.getcwd()
        self.logName = self.cwd
        self.git_repo_base = find_git_repo()
        self.folderName = self.cwd
        self.saveFolderName = self.cwd
        self.model_name = self.DEFAULT_MODEL_NAME
        self.dlc_folder = self.cwd
        self.task = self.DEFAULT_TASK
        self.name = self.DEFAULT_NAME
        self.bodyparts = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        self.config_path = ""
        self.steps_per_epoch = 300
        self.epochs = 60

        self.line_width = self.DEFAULT_PEN_SIZE
        self.color = self.DEFAULT_COLOUR
        self.cv_dim = 512
        self.canvas = Canvas(self.root_train, width=self.cv_dim, height=self.cv_dim)
        self.canvas.grid(row=5, column=0, columnspan=4, rowspan=9, sticky=N + S + W)
        self.mask = Image.new("L", (self.cv_dim, self.cv_dim))
        self.draw = ImageDraw.Draw(self.mask)

        
        self.fileEntryLabel = Label(self.root_train, text="Input folder")
        self.fileEntryLabel.grid(row=0, column=0, sticky=E + W)
        self.fileEntryButton = Button(
            self.root_train, text="Browse...", command=lambda: self.OpenFile(0)
        )
        self.folderName_str = StringVar(self.root_train, value=self.folderName)
        self.fileEntryButton.grid(row=0, column=2, sticky=E)
        self.fileEntryBox = Entry(
            self.root_train, textvariable=self.folderName_str, width=60
        )
        self.fileEntryBox.grid(row=0, column=1, padx=5, pady=5)

        self.fileSaveLabel = Label(self.root_train, text="Save folder")
        self.fileSaveLabel.grid(row=1, column=0, sticky=E + W)
        self.fileSaveButton = Button(
            self.root_train, text="Browse...", command=lambda: self.OpenFile(1)
        )
        self.saveFolderName_str = StringVar(self.root_train, value=self.saveFolderName)
        self.fileSaveButton.grid(row=1, column=2, sticky=E)
        self.fileSaveBox = Entry(
            self.root_train, textvariable=self.saveFolderName_str, width=60
        )
        self.fileSaveBox.grid(row=1, column=1, padx=5, pady=5)

        self.logSaveLabel = Label(self.root_train, text="Log folder")
        self.logSaveLabel.grid(row=2, column=0, sticky=E + W)
        self.logSaveButton = Button(
            self.root_train, text="Browse...", command=lambda: self.OpenFile(2)
        )
        self.logName_str = StringVar(self.root_train, value=self.logName)
        self.logSaveButton.grid(row=2, column=2, sticky=E)
        self.logSaveBox = Entry(
            self.root_train, textvariable=self.logName_str, width=60
        )
        self.logSaveBox.grid(row=2, column=1, padx=5, pady=5)

        self.line_width_str = StringVar(self.root_train, value=self.line_width)
        self.lineWidthLabel = Label(self.root_train, text="Brush size")
        self.lineWidthLabel.grid(row=2, column=4, sticky=E + W)
        self.lineWidthBox = Entry(
            self.root_train, textvariable=self.line_width_str, width=20
        )
        self.lineWidthBox.grid(row=2, column=5)

        self.model_name_str = StringVar(self.root_train, value=self.model_name)
        self.modelNameLabel = Label(self.root_train, text="Model name")
        self.modelNameLabel.grid(row=3, column=4, sticky=E + W)
        self.modelNameBox = Entry(
            self.root_train, textvariable=self.model_name_str, width=20
        )
        self.modelNameBox.grid(row=3, column=5)

        self.dlc_folder_str = StringVar(self.root_train, value=self.dlc_folder)
        self.dlcFolderLabel = Label(self.root_train, text="DLC Folder")
        self.dlcFolderLabel.grid(row=3, column=0, sticky=E + W)
        self.dlcFolderButton = Button(
            self.root_train, text="Browse...", command=lambda: self.OpenFile(3)
        )
        self.dlcFolderButton.grid(row=3, column=2, sticky=E)
        self.dlcFolderBox = Entry(
            self.root_train, textvariable=self.dlc_folder_str, width=60
        )
        self.dlcFolderBox.grid(row=3, column=1, padx=5, pady=5)

        
        self.saveButton = Button(
            self.root_train,
            text="Save current mask to file",
            command=lambda: self.mask_save(self.saveFolderName, self.j),
        )
        self.saveButton.grid(
            row=9, column=4, columnspan=2, padx=2, sticky=N + S + W + E
        )

        self.trainButton = Button(
            self.root_train,
            text="Train U-net model",
            command=lambda: self.trainModelGUI(
                self.saveFolderName,
                os.path.join(self.git_repo_base, "models", self.modelNameBox.get()),
                self.logName,
                self.git_repo_base,
                int(self.stepEpochsBox.get()),
                int(self.epochsBox.get()),
            ),
        )
        self.trainButton.grid(
            row=10, column=4, columnspan=2, padx=2, sticky=N + S + W + E
        )

        
        self.task_str = StringVar(self.root_train, value=self.task)
        self.taskLabel = Label(self.root_train, text="Task")
        self.taskLabel.grid(row=0, column=4, sticky=E + W)
        self.taskBox = Entry(self.root_train, textvariable=self.task_str, width=20)
        self.taskBox.grid(row=0, column=5)

        self.name_str = StringVar(self.root_train, value=self.name)
        self.nameLabel = Label(self.root_train, text="Name")
        self.nameLabel.grid(row=1, column=4, sticky=E + W)
        self.nameBox = Entry(self.root_train, textvariable=self.name_str, width=20)
        self.nameBox.grid(row=1, column=5)

        self.epochs_str = StringVar(self.root_train, value=self.epochs)
        self.epochsLabel = Label(self.root_train, text="U-Net epochs")
        self.epochsLabel.grid(row=4, column=4, sticky=E + W)
        self.epochsBox = Entry(self.root_train, textvariable=self.epochs_str, width=20)
        self.epochsBox.grid(row=4, column=5)

        self.step_epochs_str = StringVar(self.root_train, value=self.steps_per_epoch)
        self.stepEpochsLabel = Label(self.root_train, text="Steps per epoch")
        self.stepEpochsLabel.grid(row=5, column=4, sticky=E + W)
        self.stepEpochsBox = Entry(
            self.root_train, textvariable=self.step_epochs_str, width=20
        )
        self.stepEpochsBox.grid(row=5, column=5)

        self.displayiters = 100
        self.displayiters_str = StringVar(self.root_train, value=self.displayiters)
        self.displayitersLabel = Label(self.root_train, text="Display iters")
        self.displayitersLabel.grid(row=6, column=4, sticky=E + W)
        self.displayitersBox = Entry(
            self.root_train, textvariable=self.displayiters_str, width=20
        )
        self.displayitersBox.grid(row=6, column=5)

        self.saveiters = 1000
        self.saveiters_str = StringVar(self.root_train, value=self.saveiters)
        self.saveitersLabel = Label(self.root_train, text="Save iters")
        self.saveitersLabel.grid(row=7, column=4, sticky=E + W)
        self.saveitersBox = Entry(
            self.root_train, textvariable=self.saveiters_str, width=20
        )
        self.saveitersBox.grid(row=7, column=5)

        self.maxiters = 30000
        self.maxiters_str = StringVar(self.root_train, value=self.maxiters)
        self.maxitersLabel = Label(self.root_train, text="Max iters")
        self.maxitersLabel.grid(row=8, column=4, sticky=E + W)
        self.maxitersBox = Entry(
            self.root_train, textvariable=self.maxiters_str, width=20
        )
        self.maxitersBox.grid(row=8, column=5)

        
        self.dlcConfigButton = Button(
            self.root_train,
            text="Generate DLC config file",
            command=lambda: self.getDLCConfig(
                self.taskBox.get(), self.nameBox.get(), self.folderName, self.dlc_folder
            ),
        )
        self.dlcConfigButton.grid(row=11, column=4, columnspan=2, sticky=N + S + W + E)

        self.dlcLabelButton = Button(
            self.root_train,
            text="Label brain images\nwith landmarks",
            command=lambda: DLCLabel(self.config_path),
        )
        self.dlcLabelButton.grid(row=12, column=4, columnspan=2, sticky=N + S + W + E)

        self.dlcTrainButton = Button(
            self.root_train,
            text="Train DLC model",
            command=lambda: DLCTrain(
                self.config_path,
                self.displayitersBox.get(),
                self.saveitersBox.get(),
                self.maxitersBox.get(),
            ),
        )
        self.dlcTrainButton.grid(row=13, column=4, columnspan=2, sticky=N + S + W + E)

        
        
        self.nextButton = Button(
            self.root_train, text="->", command=lambda: self.forward()
        )
        self.nextButton.grid(row=14, column=2, columnspan=1)
        self.previousButton = Button(
            self.root_train, text="<-", command=lambda: self.backward()
        )
        self.previousButton.grid(row=14, column=0, columnspan=1)

        
        self.root_train.bind("<Right>", self.forward)
        self.root_train.bind("<Left>", self.backward)

        self.paint_setup()

    def OpenFile(self, openOrSave):
        if openOrSave == 0:
            newFolderName = filedialog.askdirectory(
                initialdir=self.cwd,
                title="Choose folder containing the brain images you want to analyze",
            )
            
            try:
                self.folderName_str.set(newFolderName)
                self.folderName = newFolderName
                self.ImageDisplay(1, self.folderName, 1)
                self.status = (
                    'Please select a folder to save your images to at "Save Folder".'
                )
                self.status_str.set(self.status)
                self.root_train.update()
            except:
                print("No image file selected!")
        elif openOrSave == 1:
            newSaveFolderName = filedialog.askdirectory(
                initialdir=self.cwd, title="Choose folder for saving files"
            )
            
            try:
                self.saveFolderName_str.set(newSaveFolderName)
                self.saveFolderName = newSaveFolderName
                self.status = "Save folder selected! Choose an option on the right to begin your analysis."
                self.status_str.set(self.status)
                self.root_train.update()
            except:
                print("No save file selected!")
                self.status = "No save file selected!"
                self.status_str.set(self.status)
                self.root_train.update()
        elif openOrSave == 2:
            newLogName = filedialog.askdirectory(
                initialdir=self.cwd,
                title="Choose folder for saving model training logs",
            )
            try:
                self.logName_str.set(newLogName)
                self.logName = newLogName
                self.root_train.update()
            except:
                print("No log folder selected!")
        elif openOrSave == 3:
            newDLCFolder = filedialog.askdirectory(
                initialdir=self.cwd, title="Choose folder for DLC project"
            )
            try:
                self.dlc_folder_str.set(newDLCFolder)
                self.dlc_folder = newDLCFolder
                self.root_train.update()
            except:
                print("No DLC folder selected!")

    def ImageDisplay(self, delta, folderName, reset):
        
        self.imgDisplayed = 1
        self.root_train.update()
        if reset == 1:
            self.j = -1
        self.j += delta
        if glob.glob(os.path.join(folderName, "*_mask_segmented.png")):
            file_list = glob.glob(os.path.join(folderName, "*_mask_segmented.png"))
        elif glob.glob(os.path.join(folderName, "*_mask.png")):
            file_list = glob.glob(os.path.join(folderName, "*_mask.png"))
        else:
            file_list = glob.glob(os.path.join(folderName, "*.png"))
        self.picLen = len(file_list)
        if self.j > self.picLen - 1:
            self.j = 0
        if self.j <= -1:
            self.j = self.picLen - 1
        if delta != 0:
            for file in file_list:
                if (
                    fnmatch.fnmatch(
                        file,
                        os.path.join(
                            folderName, "{}_mask_segmented.png".format(self.j)
                        ),
                    )
                    or fnmatch.fnmatch(
                        file, os.path.join(folderName, "{}.png".format(self.j))
                    )
                    or fnmatch.fnmatch(
                        file, os.path.join(folderName, "{}_mask.png".format(self.j))
                    )
                ):
                    self.imageFileName = os.path.basename(file)
                    image = os.path.join(folderName, file)
                    image_orig = Image.open(image)
                    self.image_resize = image_orig.resize((512, 512))
                    image_disp = ImageTk.PhotoImage(self.image_resize)
                    self.canvas.create_image(256, 256, image=image_disp)
                    label = Label(image=image_disp)
                    label.image = image_disp
                    self.root_train.update()
        imageName = StringVar(self.root_train, value=self.imageFileName)
        imageNum = "Image {}/{}".format(self.j + 1, self.picLen)
        imageNumPrep = StringVar(self.root_train, value=imageNum)
        imageNameLabel = Label(self.root_train, textvariable=imageName)
        imageNameLabel.grid(row=4, column=0, columnspan=1)
        imageNumLabel = Label(self.root_train, textvariable=imageNumPrep)
        imageNumLabel.grid(row=4, column=2, columnspan=1)

    def forward(self):
        self.ImageDisplay(1, self.folderName, 0)
        self.mask = Image.new("L", (self.cv_dim, self.cv_dim))
        self.draw = ImageDraw.Draw(self.mask)

    def backward(self):
        self.ImageDisplay(-1, self.folderName, 0)
        self.mask = Image.new("L", (self.cv_dim, self.cv_dim))
        self.draw = ImageDraw.Draw(self.mask)

    def paint_setup(self):
        self.old_x, self.old_y = None, None
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def paint(self, event):
        
        line_width = int(self.lineWidthBox.get())
        paint_color = self.color
        if self.old_x and self.old_y:
            self.canvas.create_oval(
                event.x - (line_width / 2),
                event.y - (line_width / 2),
                event.x + (line_width / 2),
                event.y + (line_width / 2),
                fill=paint_color,
                outline=paint_color,
            )
            self.draw.ellipse(
                (
                    event.x - (line_width / 2),
                    event.y - (line_width / 2),
                    event.x + (line_width / 2),
                    event.y + (line_width / 2),
                ),
                fill=paint_color,
                outline=paint_color,
            )
        self.old_x = event.x
        self.old_y = event.y

    def reset(self):
        self.old_x, self.old_y = None, None

    def mask_save(self, mask_folder, img_name):
        if not os.path.isdir(os.path.join(mask_folder, "image")):
            os.mkdir(os.path.join(mask_folder, "image"))
        if not os.path.isdir(os.path.join(mask_folder, "label")):
            os.mkdir(os.path.join(mask_folder, "label"))

        self.image_resize.save(
            os.path.join(mask_folder, "image", "{}.png".format(img_name))
        )
        mask_cv2 = np.array(self.mask)
        mask_cv2 = inpaintMask(mask_cv2)
        io.imsave(
            os.path.join(mask_folder, "label", "{}.png".format(img_name)), mask_cv2
        )
        

    def trainModelGUI(
        self,
        mask_folder,
        model_name,
        log_folder,
        git_repo_base,
        steps_per_epoch,
        epochs,
    ):
        trainModel(
            mask_folder, model_name, log_folder, git_repo_base, steps_per_epoch, epochs
        )
        config_project(mask_folder, log_folder, "train", model_name=model_name)

    def getDLCConfig(self, project_name, your_name, img_path, output_dir_base):
        config_path = DLCPrep(project_name, your_name, img_path, output_dir_base)
        print(config_path)
        self.config_path = config_path
        DLC_edit_bodyparts(self.config_path, self.bodyparts)


def gui():
    GuiTrain().root_train.mainloop()

```

## File: D:\MyCode\MesoNet\MesoNet\mesonet\img_augment.py

- Extension: .py
- Language: python
- Size: 4688 bytes
- Created: 2024-11-08 12:56:16
- Modified: 2024-11-08 12:56:16

### Code

```python

import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

import glob
import os
import cv2
import pandas as pd
import skimage.io as io


def img_augment_run(input_path, output_path, coords_input, data_gen_args):
    flatten = lambda l: [obj for sublist in l for obj in sublist]
    img_list = glob.glob(os.path.join(input_path, "*.png"))
    print(img_list)

    coords = pd.read_csv(coords_input, header=[0, 1, 2], index_col=[0])
    coords_aug = coords.copy()

    for img_num, img_name in enumerate(img_list):
        img = cv2.imread(img_name)
        coord_row_x = coords.iloc[img_num, 0::2]
        coord_row_y = coords.iloc[img_num, 1::2]
        coords_list = [
            Keypoint(float(x), float(y)) for [x, y] in zip(coord_row_x, coord_row_y)
        ]
        keypoints = KeypointsOnImage(coords_list, shape=img.shape)
        seq = iaa.Sequential(
            [
                iaa.Multiply(
                    (
                        0.5 - data_gen_args["brightness_range"],
                        0.5 + data_gen_args["brightness_range"],
                    )
                ),
                iaa.Affine(
                    rotate=(
                        -1 * data_gen_args["rotation_range"],
                        data_gen_args["rotation_range"],
                    ),
                    scale=(
                        1 - data_gen_args["zoom_range"],
                        1 + data_gen_args["zoom_range"],
                    ),
                    shear=(
                        -1 * data_gen_args["shear_range"],
                        data_gen_args["shear_range"],
                    ),
                    translate_percent={
                        "x": (
                            -1 * data_gen_args["width_shift_range"],
                            data_gen_args["width_shift_range"],
                        ),
                        "y": (
                            -1 * data_gen_args["height_shift_range"],
                            data_gen_args["height_shift_range"],
                        ),
                    },
                ),
            ]
        )

        
        image_aug, kps_aug = seq(image=img, keypoints=keypoints)
        coords_aug.iloc[img_num, :] = flatten(
            [[kp.x, kp.y] for kp in kps_aug.keypoints]
        )
        idx_name = coords_aug.index[img_num]
        idx_basename = os.path.basename(idx_name)
        coords_aug.rename(
            index={
                idx_name: idx_name.replace(
                    idx_basename, "{}_aug.png".format(idx_basename.split(".")[0])
                )
            },
            inplace=True,
        )
        print(
            idx_name.replace(
                idx_basename, "{}_aug.png".format(idx_basename.split(".")[0])
            )
        )

        io.imsave(os.path.join(output_path, os.path.basename(img_name)), img)
        io.imsave(
            os.path.join(
                output_path,
                "{}_aug.png".format(os.path.basename(img_name).split(".")[0]),
            ),
            image_aug,
        )

    
    
    coords_aug.sort_index(inplace=True)
    coords_aug = coords_aug.append(coords)
    coords_aug.to_csv(
        os.path.join(
            output_path, "{}.csv".format(os.path.basename(coords_input).split(".")[0])
        )
    )
    coords_aug.to_hdf(
        os.path.join(
            output_path, "{}.h5".format(os.path.basename(coords_input).split(".")[0])
        ),
        "df_with_missing",
        format="table",
        mode="w",
    )


def img_augment(
        input_path,
        output_path,
        coords_input,
        brightness_range=0.3,
        rotation_range=0.3,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
):
    data_gen_args = dict(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        brightness_range=brightness_range,
        zoom_range=zoom_range,
        shear_range=0.05,
    )
    img_augment_run(input_path, output_path, coords_input, data_gen_args)

```

## File: D:\MyCode\MesoNet\MesoNet\mesonet\mask_functions.py

- Extension: .py
- Language: python
- Size: 45315 bytes
- Created: 2024-11-08 12:56:16
- Modified: 2024-11-09 11:27:36

### Code

```python

from mesonet.utils import natural_sort_key
import numpy as np
import scipy.io as sio
import os
import glob
import skimage.io as io
import skimage.transform as trans
from skimage.util import img_as_ubyte
import cv2
import imageio
import imutils
import scipy
import pylab
from PIL import Image
import pandas as pd
from keras import backend as k
from polylabel import polylabel


Background = [0, 0, 0]

Region = [255, 255, 255]

COLOR_DICT = np.array([Background, Region])
NUM_COLORS = 9


def testGenerator(
    test_path,
    output_mask_path,
    num_image=60,
    target_size=(512, 512),
    flag_multi_class=False,
    as_gray=True,
    atlas_to_brain_align=True,
):
    
    suff = "png"
    img_list = glob.glob(os.path.join(test_path, "*png"))
    img_list.sort(key=natural_sort_key)
    tif_list = glob.glob(os.path.join(test_path, "*tif"))
    if tif_list:
        tif_stack = imageio.mimread(os.path.join(test_path, tif_list[0]))
        num_image = len(tif_stack)
    for i in range(num_image):
        if len(tif_list) > 0:
            print("TIF detected")
            img = tif_stack[i]
            img = np.uint8(img)
            img = cv2.resize(img, target_size)
        elif len(tif_list) == 0:
            if atlas_to_brain_align:
                img = io.imread(os.path.join(test_path, img_list[i]))
            else:
                try:
                    img = io.imread(
                        os.path.join(test_path, "{}_brain_warp.{}".format(i, suff))
                    )
                except:
                    img = io.imread(os.path.join(test_path, img_list[i]))
            img = trans.resize(img, target_size)
        img = img_as_ubyte(img)
        io.imsave(os.path.join(output_mask_path, "{}.{}".format(i, suff)), img)
        img = io.imread(
            os.path.join(output_mask_path, "{}.{}".format(i, suff)), as_gray=as_gray
        )
        if img.dtype == "uint8":
            img = img / 255
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


def labelVisualize(num_class, color_dict, img):
    
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


def saveResult(save_path, npyfile, flag_multi_class=False, num_class=2):
    
    for i, item in enumerate(npyfile):
        img = (
            labelVisualize(num_class, COLOR_DICT, item)
            if flag_multi_class
            else item[:, :, 0]
        )

        
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        
        img_normalized = (img - img.min()) / (img.max() - img.min()) * 255

        
        img_normalized = img_normalized.astype(np.uint8)

        
        img_pil = Image.fromarray(img_normalized)

        
        img_pil.save(os.path.join(save_path, "{}.png".format(i)))


def returnResult(save_path, npyfile):
    
    img = npyfile[0][:, :, 0]
    return img


def atlas_to_mask(
    atlas_path,
    mask_input_path,
    mask_warped_path,
    mask_output_path,
    n,
    use_unet,
    use_voxelmorph,
    atlas_to_brain_align,
    git_repo_base,
    olfactory_check,
    olfactory_bulbs_to_use,
    atlas_label
):
    
    atlas = cv2.imread(atlas_path, cv2.IMREAD_GRAYSCALE)
    mask_warped = cv2.imread(mask_warped_path, cv2.IMREAD_GRAYSCALE)
    if use_unet:
        mask_input = cv2.imread(mask_input_path, cv2.IMREAD_GRAYSCALE)
        mask_input_orig = mask_input
        if olfactory_check and not use_voxelmorph:
            cnts_for_olfactory = cv2.findContours(
                mask_input.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cnts_for_olfactory = imutils.grab_contours(cnts_for_olfactory)
            if len(cnts_for_olfactory) == 3:
                olfactory_bulbs = sorted(
                    cnts_for_olfactory, key=cv2.contourArea, reverse=True
                )[1:3]
            else:
                olfactory_bulbs = sorted(
                    cnts_for_olfactory, key=cv2.contourArea, reverse=True
                )[2:4]
        io.imsave(os.path.join(mask_output_path, "{}_mask.png".format(n)), mask_input)
        
        if atlas_to_brain_align:
            
            if use_voxelmorph and olfactory_check:
                olfactory_bulbs_to_add = olfactory_bulbs_to_use
                mask_input = cv2.bitwise_and(atlas, mask_warped)
                mask_input_orig = cv2.bitwise_and(mask_input, mask_warped)
            else:
                if olfactory_check:
                    olfactory_bulbs_to_add = olfactory_bulbs
                mask_input_orig = cv2.bitwise_and(mask_input, mask_warped)
                mask_input = cv2.bitwise_and(atlas, mask_input)
                mask_input = cv2.bitwise_and(mask_input, mask_warped)
            if len(atlas_label) > 0:
                atlas_label[np.where(mask_input == 0)] = 1000
            if olfactory_check:
                for bulb in olfactory_bulbs_to_add:
                    cv2.fillPoly(mask_input, pts=[bulb], color=[255, 255, 255])
                    cv2.fillPoly(mask_input_orig, pts=[bulb], color=[255, 255, 255])
                if len(atlas_label) > 0:
                    try:
                        cv2.fillPoly(
                            atlas_label, pts=[olfactory_bulbs_to_add[0]], color=[300]
                        )
                        cv2.fillPoly(
                            atlas_label, pts=[olfactory_bulbs_to_add[1]], color=[400]
                        )
                        atlas_label[np.where(atlas_label == 300)] = 300
                        atlas_label[np.where(atlas_label == 400)] = 400
                    except:
                        print('No olfactory bulb found!')
                        
                        
                        mask_input = cv2.imread(mask_input_path, cv2.IMREAD_GRAYSCALE)
                        mask_input_orig = mask_input
                        mask_input_orig = cv2.bitwise_and(mask_input, mask_warped)
                        mask_input = cv2.bitwise_and(atlas, mask_input)
                        mask_input = cv2.bitwise_and(mask_input, mask_warped)
        else:
            
            mask_input = cv2.bitwise_and(atlas, mask_warped)
            mask_input_orig = cv2.bitwise_and(mask_input, mask_warped)
            if olfactory_check and len(olfactory_bulbs_to_use) > 0:
                for bulb in olfactory_bulbs_to_use:
                    cv2.fillPoly(mask_input, pts=[bulb], color=[255, 255, 255])
                    cv2.fillPoly(mask_input_orig, pts=[bulb], color=[255, 255, 255])
                if len(atlas_label) > 0:
                    try:
                        cv2.fillPoly(
                            atlas_label, pts=[olfactory_bulbs_to_use[0]], color=[300]
                        )
                        cv2.fillPoly(
                            atlas_label, pts=[olfactory_bulbs_to_use[1]], color=[400]
                        )
                        atlas_label[np.where(atlas_label == 300)] = 300
                        atlas_label[np.where(atlas_label == 400)] = 400
                    except:
                        print('No olfactory bulb found!')
                        
                        
                        mask_input = cv2.imread(mask_input_path, cv2.IMREAD_GRAYSCALE)
                        mask_input_orig = mask_input
                        mask_input = cv2.bitwise_and(atlas, mask_warped)
                        mask_input_orig = cv2.bitwise_and(mask_input, mask_warped)
        io.imsave(
            os.path.join(mask_output_path, "{}_mask_no_atlas.png".format(n)),
            mask_input_orig
        )
    else:
        mask_input = cv2.bitwise_and(atlas, mask_warped)
        if len(atlas_label) > 0:
            atlas_label[np.where(mask_input == 0)] = 1000
        if olfactory_check and not atlas_to_brain_align:
            olfactory_path = os.path.join(git_repo_base, "atlases")
            olfactory_left = cv2.imread(
                os.path.join(olfactory_path, "02.png"), cv2.IMREAD_GRAYSCALE
            )
            olfactory_right = cv2.imread(
                os.path.join(olfactory_path, "01.png"), cv2.IMREAD_GRAYSCALE
            )
            cnts_for_olfactory_left, hierarchy = cv2.findContours(
                olfactory_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )[-2:]
            olfactory_left_cnt = min(cnts_for_olfactory_left, key=cv2.contourArea)
            cnts_for_olfactory_right, hierarchy = cv2.findContours(
                olfactory_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )[-2:]
            olfactory_right_cnt = min(cnts_for_olfactory_right, key=cv2.contourArea)
            cv2.fillPoly(mask_input, pts=[olfactory_left_cnt], color=[255, 255, 255])
            cv2.fillPoly(mask_input, pts=[olfactory_right_cnt], color=[255, 255, 255])
    io.imsave(os.path.join(mask_output_path, "{}.png".format(n)), mask_input)
    return atlas_label


def inpaintMask(mask):
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    for cnt in cnts:
        cv2.fillPoly(mask, pts=[cnt], color=[255, 255, 255])
    return mask


def applyMask(
    image_path,
    mask_path,
    save_path,
    segmented_save_path,
    mat_save,
    threshold,
    git_repo_base,
    bregma_list,
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
    olfactory_bulbs_to_use_list,
    region_labels=True,
    original_label=False,
):
    

    tif_list = glob.glob(os.path.join(image_path, "*tif"))
    if atlas_to_brain_align:
        if use_dlc and align_once:
            image_name_arr = glob.glob(os.path.join(mask_path, "*_brain_warp.png"))
        else:
            image_name_arr = glob.glob(os.path.join(image_path, "*.png"))
        image_name_arr.sort(key=natural_sort_key)
        if tif_list:
            tif_stack = imageio.mimread(os.path.join(image_path, tif_list[0]))
            image_name_arr = tif_stack
    else:
        
        image_name_arr = glob.glob(os.path.join(mask_path, "*_brain_warp.png"))
        image_name_arr.sort(key=natural_sort_key)

    region_bgr_lower = (220, 220, 220)  
    region_bgr_upper = (255, 255, 255)
    base_c_max = []
    count = 0
    
    mat_files = glob.glob(os.path.join(git_repo_base, "atlases/mat_contour_base/*.mat"))
    mat_files.sort(key=natural_sort_key)

    
    cm = pylab.get_cmap("viridis")
    colors = [cm(1.0 * i / NUM_COLORS)[0:3] for i in range(NUM_COLORS)]
    colors = [tuple(color_idx * 255 for color_idx in color_t) for color_t in colors]
    for file in mat_files:
        mat = scipy.io.loadmat(
            os.path.join(git_repo_base, "atlases/mat_contour_base/", file)
        )
        mat = mat["vect"]
        ret, thresh = cv2.threshold(mat, 5, 255, cv2.THRESH_BINARY)
        base_c = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        base_c = imutils.grab_contours(base_c)
        base_c_max.append(max(base_c, key=cv2.contourArea))

    for i, item in enumerate(image_name_arr):
        label_num = 0
        if not atlas_to_brain_align:
            atlas_path = os.path.join(mask_path, "{}_atlas.png".format(str(i)))
            mask_input_path = os.path.join(mask_path, "{}.png".format(i))
            mask_warped_path = os.path.join(
                mask_path, "{}_mask_warped.png".format(str(i))
            )
            if olfactory_check:
                olfactory_bulbs_to_use = olfactory_bulbs_to_use_list[i]
            else:
                olfactory_bulbs_to_use = []
            atlas_to_mask(
                atlas_path,
                mask_input_path,
                mask_warped_path,
                mask_path,
                i,
                use_unet,
                use_voxelmorph,
                atlas_to_brain_align,
                git_repo_base,
                olfactory_check,
                olfactory_bulbs_to_use,
                []
            )
        new_data = []
        if len(tif_list) != 0 and atlas_to_brain_align:
            img = item
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.imread(item)
        if atlas_to_brain_align:
            img = cv2.resize(img, (512, 512))
        if use_dlc:
            bregma_x, bregma_y = bregma_list[i]
        else:
            bregma_x, bregma_y = [round(img.shape[0]/2), round(img.shape[1]/2)]
            original_label = True
        if use_voxelmorph and i == 1:
            mask = cv2.imread(os.path.join(mask_path, "{}.png".format(0)))
        else:
            mask = cv2.imread(os.path.join(mask_path, "{}.png".format(i)))
        mask = cv2.resize(mask, (img.shape[0], img.shape[1]))
        
        mask_color = cv2.inRange(mask, region_bgr_lower, region_bgr_upper)
        io.imsave(os.path.join(save_path, "{}_mask_binary.png".format(i)), mask_color)
        
        
        kernel = np.ones((3, 3), np.uint8)  
        mask_color = np.uint8(mask_color)
        thresh_atlas, atlas_bw = cv2.threshold(mask_color, 128, 255, 0)
        
        
        

        if not atlas_to_brain_align:
            watershed_run_rule = True
        else:
            if len(tif_list) == 0:
                watershed_run_rule = True
            else:
                watershed_run_rule = i == 0
        if align_once:
            watershed_run_rule = i == 0

        labels_from_region = []

        if watershed_run_rule:
            orig_list = []
            orig_list_labels = []
            orig_list_labels_left = []
            orig_list_labels_right = []
            
            
            unique_regions = [
                -275,
                -268,
                -255,
                -249,
                -164,
                -150,
                -143,
                -136,
                -129,
                -98,
                -78,
                -71,
                -64,
                -57,
                -50,
                -43,
                -36,
                -29,
                -21,
                -15,
                0,
                15,
                21,
                29,
                36,
                43,
                50,
                57,
                64,
                71,
                78,
                98,
                129,
                136,
                143,
                150,
                164,
                249,
                255,
                268,
                275,
                300,
                400,
            ]
            cnts_orig = []
            
            if atlas_to_brain_align and not original_label:
                np.savetxt(
                    "atlas_label_list_{}.csv".format(i),
                    atlas_label_list[i],
                    delimiter=",",
                )
                for region_idx in unique_regions:
                    if region_idx in [300, 400]:
                        
                        region = cv2.inRange(
                            atlas_label_list[i], region_idx - 5, region_idx + 5
                        )
                        cnt_for_idx, hierarchy = cv2.findContours(
                            region.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
                        )[-2:]
                        if len(cnt_for_idx) >= 1:
                            cnt_for_idx = cnt_for_idx[0]
                    else:
                        region = cv2.inRange(
                            atlas_label_list[i], region_idx, region_idx
                        )
                        cnt_for_idx = cv2.findContours(
                            region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                        )
                        cnt_for_idx = imutils.grab_contours(cnt_for_idx)
                        if len(cnt_for_idx) >= 1:
                            cnt_for_idx = max(cnt_for_idx, key=cv2.contourArea)
                    if len(cnt_for_idx) >= 1:
                        cnts_orig.append(cnt_for_idx)
                        labels_from_region.append(region_idx)
            else:
                cnts_orig = cv2.findContours(
                    atlas_bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                cnts_orig = imutils.grab_contours(cnts_orig)
            if not use_dlc:
                cnts_orig, hierarchy = cv2.findContours(
                    atlas_bw.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
                )[-2:]
            labels_cnts = []
            for (num_label, cnt_orig) in enumerate(cnts_orig):
                labels_cnts.append(cnt_orig)
                try:
                    cv2.drawContours(img, cnt_orig, -1, (255, 0, 0), 1)
                except:
                    print("Could not draw contour!")
                
                if atlas_to_brain_align:
                    c_orig_as_list = cnt_orig.tolist()
                    c_orig_as_list = [[c_val[0] for c_val in c_orig_as_list]]
                else:
                    c_orig_as_list = cnt_orig.tolist()
                    c_orig_as_list = [[c_val[0] for c_val in c_orig_as_list]]
                orig_polylabel = polylabel(c_orig_as_list)
                orig_x, orig_y = int(orig_polylabel[0]), int(orig_polylabel[1])

                if not original_label and atlas_to_brain_align:
                    label_to_use = unique_regions.index(labels_from_region[num_label])
                    (text_width, text_height) = cv2.getTextSize(
                        str(label_to_use), cv2.FONT_HERSHEY_SIMPLEX, 0.4, thickness=1
                    )[0]
                    label_jitter = 0
                    label_color = (0, 0, 255)
                    cv2.rectangle(
                        img,
                        (orig_x + label_jitter, orig_y + label_jitter),
                        (
                            orig_x + label_jitter + text_width,
                            orig_y + label_jitter - text_height,
                        ),
                        (255, 255, 255),
                        cv2.FILLED,
                    )
                    cv2.putText(
                        img,
                        str(label_to_use),
                        (int(orig_x + label_jitter), int(orig_y + label_jitter)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        label_color,
                        1,
                    )
                    label_num += 1
                orig_list.append((orig_x, orig_y))
                orig_list_labels.append(
                    (orig_x - bregma_x, orig_y - bregma_y, orig_x, orig_y, num_label)
                )
                if (orig_x - bregma_x) < 0:
                    orig_list_labels_left.append(
                        (
                            orig_x - bregma_x,
                            orig_y - bregma_y,
                            orig_x,
                            orig_y,
                            num_label,
                        )
                    )
                elif (orig_x - bregma_x) > 0:
                    orig_list_labels_right.append(
                        (
                            orig_x - bregma_x,
                            orig_y - bregma_y,
                            orig_x,
                            orig_y,
                            num_label,
                        )
                    )
                orig_list.sort()
            orig_list_labels_sorted_left = sorted(
                orig_list_labels_left, key=lambda t: t[0], reverse=True
            )
            orig_list_labels_sorted_right = sorted(
                orig_list_labels_right, key=lambda t: t[0]
            )
            flatten = lambda l: [obj for sublist in l for obj in sublist]
            orig_list_labels_sorted = flatten(
                [orig_list_labels_sorted_left, orig_list_labels_sorted_right]
            )
            vertical_check = np.asarray([val[0] for val in orig_list_labels_sorted])
            for (orig_coord_val, orig_coord) in enumerate(orig_list_labels_sorted):
                vertical_close = np.where((abs(vertical_check - orig_coord[0]) <= 5))
                vertical_close_slice = vertical_close[0]
                vertical_matches = np.asarray(orig_list_labels_sorted)[
                    vertical_close_slice
                ]
                if len(vertical_close_slice) > 1:
                    vertical_match_sorted = sorted(vertical_matches, key=lambda t: t[1])
                    orig_list_labels_sorted_np = np.asarray(orig_list_labels_sorted)
                    orig_list_labels_sorted_np[
                        vertical_close_slice
                    ] = vertical_match_sorted
                    orig_list_labels_sorted = orig_list_labels_sorted_np.tolist()
            img = np.uint8(img)
        else:
            for num_label, cnt_orig in enumerate(cnts_orig):  
                try:
                    cv2.drawContours(img, cnt_orig, -1, (255, 0, 0), 1)
                except:
                    print("Could not draw contour!")
        if not atlas_to_brain_align and use_unet:
            cortex_mask = cv2.imread(os.path.join(mask_path, "{}_mask.png".format(i)))
            cortex_mask = cv2.cvtColor(cortex_mask, cv2.COLOR_RGB2GRAY)
            thresh, cortex_mask_thresh = cv2.threshold(cortex_mask, 128, 255, 0)
            cortex_cnt = cv2.findContours(
                cortex_mask_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cortex_cnt = imutils.grab_contours(cortex_cnt)
        labels_x = []
        labels_y = []
        areas = []
        sorted_labels_arr = []
        label_jitter = 0
        mask = np.zeros(mask_color.shape, dtype="uint8")
        cnts = cnts_orig
        print("LEN CNTS: {}".format(len(cnts)))
        print("LEN LABELS: {}".format(len(orig_list_labels_sorted)))
        if original_label or not atlas_to_brain_align:
            labels_from_region = [0] * len(orig_list_labels_sorted)
        for (z, cnt), (coord_idx, coord), label_from_region in zip(
            enumerate(cnts), enumerate(orig_list_labels_sorted), labels_from_region
        ):
            if atlas_to_brain_align and not original_label:
                coord_label_num = unique_regions.index(labels_from_region[coord_idx])
            else:
                coord_label_num = coord_idx
            
            if len(cnts) > 1:
                z = 0
            c_x, c_y = int(coord[2]), int(coord[3])
            c = cnt
            if not atlas_to_brain_align and use_unet:
                cnt_loc_label = (
                    "inside"
                    if [1.0]
                    in [
                        list(
                            set(
                                [
                                    cv2.pointPolygonTest(
                                        cortex_sub_cnt,
                                        (
                                            c_coord.tolist()[0][0],
                                            c_coord.tolist()[0][1],
                                        ),
                                        False,
                                    )
                                    for c_coord in c
                                ]
                            )
                        )
                        for cortex_sub_cnt in cortex_cnt
                    ]
                    else "outside"
                )
            else:
                cnt_loc_label = ""
            rel_x = c_x - bregma_x
            rel_y = c_y - bregma_y

            pt_inside_cnt = [
                coord_check
                for coord_check in orig_list_labels_sorted
                if cv2.pointPolygonTest(
                    c, (int(coord_check[2]), int(coord_check[3])), False
                )
                == 1
            ]
            if original_label:
                try:
                    pt_inside_cnt_idx = orig_list_labels_sorted.index(pt_inside_cnt[0])
                    label_for_mat = pt_inside_cnt_idx
                except:
                    label_for_mat = coord_label_num
                    print(
                        "WARNING: label {} was not found in region. Order of labels may be incorrect!".format(str(coord_idx))
                    )
            else:
                label_for_mat = coord_label_num
            c_rel_centre = [rel_x, rel_y]
            if not os.path.isdir(
                os.path.join(segmented_save_path, "mat_contour_centre")
            ):
                os.mkdir(os.path.join(segmented_save_path, "mat_contour_centre"))

            
            if mat_save:
                mat_save = True
            else:
                mat_save = False
            
            
            
            sorted_labels_arr.append(coord_label_num)
            labels_x.append(int(c_x))
            labels_y.append(int(c_y))
            areas.append(cv2.contourArea(c))
            shape_list = []
            label_color = (0, 0, 255)
            for n_bc, bc in enumerate(base_c_max):
                shape_compare = cv2.matchShapes(c, bc, 1, 0.0)
                shape_list.append(shape_compare)
            if (not region_labels and original_label) or (
                not region_labels and not atlas_to_brain_align
            ):
                (text_width, text_height) = cv2.getTextSize(
                    str(coord_label_num), cv2.FONT_HERSHEY_SIMPLEX, 0.4, thickness=1
                )[0]
                cv2.rectangle(
                    img,
                    (c_x + label_jitter, c_y + label_jitter),
                    (c_x + label_jitter + text_width, c_y + label_jitter - text_height),
                    (255, 255, 255),
                    cv2.FILLED,
                )
                cv2.putText(
                    img,
                    str(coord_label_num),
                    (int(c_x + label_jitter), int(c_y + label_jitter)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    label_color,
                    1,
                )
                label_num += 1

            if mat_save:
                
                
                c_total = np.zeros_like(mask)
                c_centre = np.zeros_like(mask)
                
                
                cv2.fillPoly(c_total, pts=[c], color=(255, 255, 255))
                
                if c_x < mask.shape[0] and c_y < mask.shape[0]:
                    c_centre[c_x, c_y] = 255
                if not os.path.isdir(os.path.join(segmented_save_path, "mat_contour")):
                    os.mkdir(os.path.join(segmented_save_path, "mat_contour"))
                if not os.path.isdir(
                    os.path.join(segmented_save_path, "mat_contour_centre")
                ):
                    os.mkdir(os.path.join(segmented_save_path, "mat_contour_centre"))
                sio.savemat(
                    os.path.join(
                        segmented_save_path,
                        "mat_contour/roi_{}_{}_{}_{}.mat".format(
                            cnt_loc_label, i, label_for_mat, z
                        ),
                    ),
                    {
                        "roi_{}_{}_{}_{}".format(
                            cnt_loc_label, i, label_for_mat, z
                        ): c_total
                    },
                    appendmat=False,
                )
                sio.savemat(
                    os.path.join(
                        segmented_save_path,
                        "mat_contour_centre/roi_centre_{}_{}_{}_{}.mat".format(
                            cnt_loc_label, i, label_for_mat, z
                        ),
                    ),
                    {
                        "roi_centre_{}_{}_{}_{}".format(
                            cnt_loc_label, i, label_for_mat, z
                        ): c_centre
                    },
                    appendmat=False,
                )
                sio.savemat(
                    os.path.join(
                        segmented_save_path,
                        "mat_contour_centre/rel_roi_centre_{}_{}_{}_{}.mat".format(
                            cnt_loc_label, i, label_for_mat, z
                        ),
                    ),
                    {
                        "rel_roi_centre_{}_{}_{}_{}".format(
                            cnt_loc_label, i, label_for_mat, z
                        ): c_rel_centre
                    },
                    appendmat=False,
                )
            count += 1
        if align_once:
            idx_to_use = 0
        else:
            idx_to_use = i
        if plot_landmarks:
            for pt, color in zip(dlc_pts[idx_to_use], colors):
                cv2.circle(img, (int(pt[0]), int(pt[1])), 10, color, -1)
            for pt, color in zip(atlas_pts[idx_to_use], colors):
                cv2.circle(img, (int(pt[0]), int(pt[1])), 5, color, -1)
        io.imsave(
            os.path.join(segmented_save_path, "{}_mask_segmented.png".format(i)), img
        )
        img_edited = Image.open(os.path.join(save_path, "{}_mask_binary.png".format(i)))
        
        img_rgba = img_edited.convert("RGBA")
        data = img_rgba.getdata()
        for pixel in data:
            if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
                new_data.append((pixel[0], pixel[1], pixel[2], 0))
            else:
                new_data.append(pixel)
        img_rgba.putdata(new_data)
        img_rgba.save(os.path.join(save_path, "{}_mask_transparent.png".format(i)))
        img_transparent = cv2.imread(
            os.path.join(save_path, "{}_mask_transparent.png".format(i))
        )
        img_trans_for_mat = np.uint8(img_transparent)
        if mat_save:
            sio.savemat(
                os.path.join(
                    segmented_save_path, "mat_contour/transparent_{}".format(i)
                ),
                {"transparent_{}".format(i): img_trans_for_mat},
            )
        masked_img = cv2.bitwise_and(img, img_transparent, mask=mask_color)
        if plot_landmarks:
            for pt, color in zip(dlc_pts[idx_to_use], colors):
                cv2.circle(masked_img, (int(pt[0]), int(pt[1])), 10, color, -1)
            for pt, color in zip(atlas_pts[idx_to_use], colors):
                cv2.circle(masked_img, (int(pt[0]), int(pt[1])), 5, color, -1)
        io.imsave(os.path.join(save_path, "{}_overlay.png".format(i)), masked_img)
        print("Mask {} saved!".format(i))
        d = {
            "sorted_label": sorted_labels_arr,
            "x": labels_x,
            "y": labels_y,
            "area": areas,
        }
        df = pd.DataFrame(data=d)
        if not os.path.isdir(os.path.join(segmented_save_path, "region_labels")):
            os.mkdir(os.path.join(segmented_save_path, "region_labels"))
        df.to_csv(
            os.path.join(
                segmented_save_path, "region_labels", "{}_region_labels.csv".format(i)
            )
        )
    print(
        "Analysis complete! Check the outputs in the folders of {}.".format(save_path)
    )
    k.clear_session()
    if dlc_pts:
        os.chdir("../..")
    else:
        os.chdir(os.path.join(save_path, '..'))

```

## File: D:\MyCode\MesoNet\MesoNet\mesonet\model.py

- Extension: .py
- Language: python
- Size: 4077 bytes
- Created: 2024-11-08 12:56:16
- Modified: 2024-11-08 12:56:16

### Code

```python






from keras.models import *
from keras.layers import *
from keras.optimizers import *


def unet(pretrained_weights=None, input_size=(512, 512, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(inputs)
    conv1 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool1)
    conv2 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool2)
    conv3 = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool3)
    conv4 = Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(
        1024, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool4)
    conv5 = Conv2D(
        1024, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(
        512, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge6)
    conv6 = Conv2D(
        512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv6)

    up7 = Conv2D(
        256, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge7)
    conv7 = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv7)

    up8 = Conv2D(
        128, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge8)
    conv8 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv8)

    up9 = Conv2D(
        64, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge9)
    conv9 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv9)
    conv9 = Conv2D(
        2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv9)
    conv10 = Conv2D(1, 1, activation="sigmoid")(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(
        optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=["accuracy"]
    )

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

```

## File: D:\MyCode\MesoNet\MesoNet\mesonet\predict_regions.py

- Extension: .py
- Language: python
- Size: 6795 bytes
- Created: 2024-11-08 12:56:16
- Modified: 2024-11-08 12:56:16

### Code

```python

from mesonet.mask_functions import *
import os
from mesonet.utils import parse_yaml
from keras.models import load_model


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
    
    
    
    output_mask_path = os.path.join(output, "output_mask")
    
    output_overlay_path = os.path.join(output, "output_overlay")
    if not os.path.isdir(output_mask_path):
        os.mkdir(output_mask_path)
    if not os.path.isdir(output_overlay_path):
        os.mkdir(output_overlay_path)
    
    print(model)
    if not mask_generate:
        model_path = os.path.join(git_repo_base, "models", model)
        model_to_use = load_model(model_path)
    else:
        model_to_use = load_model(model)
    
    print(input_file)
    test_gen = testGenerator(
        input_file,
        output_mask_path,
        num_images,
        atlas_to_brain_align=atlas_to_brain_align,
    )
    
    results = model_to_use.predict(test_gen, steps=num_images, verbose=1)
    
    saveResult(output_mask_path, results)
    if not mask_generate:
        plot_landmarks = False
        use_dlc = False
        use_voxelmorph = False
        
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

```

## File: D:\MyCode\MesoNet\MesoNet\mesonet\train_model.py

- Extension: .py
- Language: python
- Size: 4448 bytes
- Created: 2024-11-08 12:56:16
- Modified: 2024-11-08 12:56:16

### Code

```python

from mesonet.model import *
from mesonet.data import *
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
from mesonet.utils import parse_yaml
from mesonet.dlc_predict import DLC_edit_bodyparts


def trainModel(
    input_file,
    model_name,
    log_folder,
    git_repo_base,
    steps_per_epoch,
    epochs,
    rotation_range=0.3,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest",
):
    
    data_gen_args = dict(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode,
    )
    train_gen = trainGenerator(
        2, input_file, "image", "label", data_gen_args, save_to_dir=None
    )
    model_path = os.path.join(git_repo_base, "models", model_name)
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = unet()
    model_checkpoint = ModelCheckpoint(
        model_name, monitor="loss", verbose=1, save_best_only=True
    )
    history_callback = model.fit_generator(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[model_checkpoint],
    )
    loss_history = history_callback.history["loss"]
    try:
        acc_history = history_callback.history["acc"]
        np_acc_hist = np.array(acc_history)
        log_acc_df = pd.DataFrame(np_acc_hist, columns=["acc"])
        log_acc_df.to_csv(os.path.join(log_folder, "acc_history.csv"))
    except:
        print("Cannot find acc history!")
    np_loss_hist = np.array(loss_history)
    log_loss_df = pd.DataFrame(np_loss_hist, columns=["loss"])
    log_loss_df.to_csv(os.path.join(log_folder, "loss_history.csv"))
    model.save(model_path)


def train_model(config_file):
    
    cfg = parse_yaml(config_file)
    input_file = cfg["input_file"]
    model_name = cfg["model_name"]
    log_folder = cfg["log_folder"]
    git_repo_base = cfg["git_repo_base"]
    steps_per_epoch = cfg["steps_per_epoch"]
    epochs = cfg["epochs"]
    bodyparts = cfg["bodyparts"]
    DLC_edit_bodyparts(config_file, bodyparts)
    rotation_range = cfg["rotation_range"]
    width_shift_range = cfg["width_shift_range"]
    height_shift_range = cfg["height_shift_range"]
    shear_range = cfg["shear_range"]
    zoom_range = cfg["zoom_range"]
    horizontal_flip = cfg["horizontal_flip"]
    fill_mode = cfg["fill_mode"]
    trainModel(
        input_file,
        model_name,
        log_folder,
        git_repo_base,
        steps_per_epoch,
        epochs,
        rotation_range,
        width_shift_range,
        height_shift_range,
        shear_range,
        zoom_range,
        horizontal_flip,
        fill_mode,
    )

```

## File: D:\MyCode\MesoNet\MesoNet\mesonet\utils.py

- Extension: .py
- Language: python
- Size: 16731 bytes
- Created: 2024-11-08 12:56:16
- Modified: 2024-11-08 12:56:16

### Code

```python

import yaml
import glob
import re
import os
from os.path import join
from sys import platform
import scipy.io as sio
import cv2
import numpy as np
import pathlib


def config_project(
    input_dir,
    output_dir,
    mode,
    model_name="unet.hdf5",
    config="dlc/config.yaml",
    atlas=False,
    sensory_match=False,
    sensory_path="sensory",
    mask_generate=True,
    mat_save=True,
    use_unet=True,
    use_dlc=True,
    atlas_to_brain_align=True,
    olfactory_check=True,
    plot_landmarks=True,
    align_once=False,
    original_label=False,
    use_voxelmorph=True,
    exist_transform=False,
    voxelmorph_model="motif_model_atlas.h5",
    template_path="templates",
    flow_path="",
    coords_input_file="",
    atlas_label_list=[],
    threshold=0.0001,
    model="models/unet_bundary.hdf5",
    region_labels=False,
    steps_per_epoch=300,
    epochs=60,
    rotation_range=0.3,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest",
):
    

    
    git_repo_base = find_git_repo()
    print(git_repo_base)
    config = join(git_repo_base, config)
    model = join(git_repo_base, model)
    filename = "mesonet_config.yaml"
    data = dict()

    if mode == "test":
        filename = "mesonet_test_config.yaml"
        num_images = len(glob.glob(os.path.join(input_dir, "*.png")))
        data = dict(
            config=config,
            input_file=input_dir,
            output=output_dir,
            atlas=atlas,
            sensory_match=sensory_match,
            sensory_path=sensory_path,
            mat_save=mat_save,
            threshold=threshold,
            mask_generate=mask_generate,
            num_images=num_images,
            model=model,
            git_repo_base=git_repo_base,
            region_labels=region_labels,
            landmark_arr=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            use_unet=use_unet,
            use_dlc=use_dlc,
            atlas_to_brain_align=atlas_to_brain_align,
            olfactory_check=olfactory_check,
            plot_landmarks=plot_landmarks,
            align_once=align_once,
            atlas_label_list=atlas_label_list,
            original_label=original_label,
            use_voxelmorph=use_voxelmorph,
            exist_transform=exist_transform,
            voxelmorph_model=voxelmorph_model,
            template_path=template_path,
            flow_path=flow_path,
            coords_input_file=coords_input_file,
        )
    elif mode == "train":
        filename = "mesonet_train_config.yaml"
        data = dict(
            input_file=input_dir,
            model_name=model_name,
            log_folder=output_dir,
            git_repo_base=git_repo_base,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            bodyparts=["A", "B", "C", "D", "E", "F", "G", "H", "I"],
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            fill_mode=fill_mode,
        )

    if glob.glob(os.path.join(input_dir, "*.mat")) or glob.glob(
        os.path.join(input_dir, "*.npy")
    ):
        convert_to_png(input_dir)

    with open(os.path.join(output_dir, filename), "w") as outfile:
        yaml.dump(data, outfile)

    config_file = os.path.join(output_dir, filename)
    return config_file


def parse_yaml(config_file):
    
    with open(config_file, "r") as stream:
        try:
            d = yaml.safe_load(stream)
            return d
        except yaml.YAMLError as exc:
            print(exc)


def natural_sort_key(s):
    
    _nsre = re.compile("([0-9]+)")
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)
    ]


def find_git_repo():
    
    
    git_repo_base = ""
    try:
        git_repo_base = os.path.join(os.getenv("MESONET_GIT"), "mesonet")
        print(
            "MesoNet git repo found at {}, skipping directory check...".format(
                git_repo_base
            )
        )
    except:
        
        
        

        
        
        
        in_colab = False
        try:
            import google.colab
            in_colab = True
        except:
            in_colab = False

        root_folder = "C:\\"
        git_repo_marker = "mesonet.txt"
        if platform == "linux" or platform == "linux2":
            
            root_folder = "/home"
        elif platform == "darwin":
            
            root_folder = "/Users"
        elif platform == "win32":
            
            
            drive_letter = pathlib.Path.home().drive
            root_folder = "{}\\".format(drive_letter)
        elif in_colab:
            root_folder = "/content"
        for root, dirs, files in os.walk(root_folder):
            if git_repo_marker in files:
                git_repo_base = root
                break
    return git_repo_base


def convert_to_png(input_folder):
    
    if glob.glob(os.path.join(input_folder, "*.mat")):
        input_file = glob.glob(os.path.join(input_folder, "*.mat"))[0]
        base_name = os.path.basename(input_file).split(".")[0]
        img_path = os.path.join(os.path.split(input_file)[0], base_name + ".png")
        print(img_path)
        mat = sio.loadmat(input_file)
        mat_shape = mat[list(mat.keys())[3]]
        if len(mat_shape.shape) > 2:
            for idx_arr in range(0, mat_shape.shape[2]):
                mat_layer = mat_shape[:, :, idx_arr]
                base_name_multi_idx = str(idx_arr) + "_" + base_name
                img_path_multi_idx = os.path.join(
                    os.path.split(input_file)[0], base_name_multi_idx + ".png"
                )
                cv2.imwrite(img_path_multi_idx, mat_layer)
        else:
            mat = mat[
                str(list({k: v for (k, v) in mat.items() if "__" not in k}.keys())[0])
            ]
            mat = mat * 255
            cv2.imwrite(img_path, mat)
        print(".mat written to .png!")
    elif glob.glob(os.path.join(input_folder, "*.npy")):
        input_file = glob.glob(os.path.join(input_folder, "*.npy"))[0]
        base_name = os.path.basename(input_file).split(".")[0]
        img_path = os.path.join(os.path.split(input_file)[0], base_name + ".png")
        npy = np.load(input_file)
        if npy.ndim == 3:
            for idx_arr, arr in enumerate(npy):
                base_name_multi_idx = str(idx_arr) + "_" + base_name
                img_path_multi_idx = os.path.join(
                    os.path.split(input_file)[0], base_name_multi_idx + ".png"
                )
                cv2.imwrite(img_path_multi_idx, arr * 255)
        else:
            npy = npy * 255
            cv2.imwrite(img_path, npy)




















```

## File: D:\MyCode\MesoNet\MesoNet\mesonet\voxelmorph_align.py

- Extension: .py
- Language: python
- Size: 6014 bytes
- Created: 2024-11-08 12:56:16
- Modified: 2024-11-08 12:56:16

### Code

```python


from mesonet.mask_functions import *
import numpy as np
import voxelmorph as vxm
from skimage.color import rgb2gray


def vxm_data_generator(x_data, template, batch_size=1):
    

    
    if batch_size == 1:
        x_data = rgb2gray(x_data)
        template = rgb2gray(template)
        x_data = np.expand_dims(x_data, axis=0)
        template = np.expand_dims(template, axis=0)
    vol_shape = x_data.shape[1:]  
    ndims = len(vol_shape)

    
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        
        
        idx1 = np.random.randint(0, template.shape[0], size=batch_size)
        moving_images = template[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]
        inputs = [moving_images, fixed_images]
        outputs = [fixed_images, zero_phi]

        yield inputs, outputs


def init_vxm_model(img_path, model_path):
    
    
    nb_features = [
        [32, 32, 32, 32],  
        [32, 32, 32, 32, 32, 16],  
    ]

    
    inshape = img_path.shape[0:2]
    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
    losses = [vxm.losses.MSE().loss, vxm.losses.Grad("l2").loss]
    lambda_param = 0.05
    loss_weights = [1, lambda_param]
    vxm_model.compile(optimizer="Adam", loss=losses, loss_weights=loss_weights)
    vxm_model.load_weights(model_path)
    return vxm_model


def vxm_transform(x_data, flow_path):
    flow_data = np.load(flow_path, allow_pickle=True)
    
    x_data = np.expand_dims(x_data, axis=0)
    x_data = x_data[..., np.newaxis]

    vol_size = x_data.shape[1:-1]

    results = vxm.networks.Transform(
        vol_size, interp_method="linear", nb_feats=x_data.shape[-1]
    ).predict([x_data, flow_data])
    output_img = results[0, :, :, 0]
    return output_img


def voxelmorph_align(model_path, img_path, template, exist_transform, flow_path):
    
    if not exist_transform:
        vxm_model = init_vxm_model(img_path, model_path)
        val_generator = vxm_data_generator(img_path, template)
        val_input, _ = next(val_generator)

        
        results = vxm_model.predict(val_input)
        
        output_img = [img[0, :, :, 0] for img in results][0]
        
        flow_img = results[1]
    else:
        print("using existing transform")
        output_img = vxm_transform(img_path, flow_path)
        
        flow_img = ""

    print("Results saved!")
    return output_img, flow_img

```

## File: D:\MyCode\MesoNet\MesoNet\mesonet\__init__.py

- Extension: .py
- Language: python
- Size: 482 bytes
- Created: 2024-11-08 12:56:16
- Modified: 2024-11-08 12:56:16

### Code

```python


from mesonet.utils import *
from mesonet.dlc_predict import predict_dlc
from mesonet.predict_regions import predict_regions
from mesonet.train_model import train_model
from mesonet.gui_start import gui_start
from mesonet.img_augment import img_augment

```

