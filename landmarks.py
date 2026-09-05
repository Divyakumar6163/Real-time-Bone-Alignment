import math
import os
import torch
import numpy as np
from utils.landmark_utils import mask2contour, denoise, get_points
import cv2 as cv
import matplotlib.pyplot as plt
from inference import inference
from numpy.polynomial import Polynomial
from tqdm import tqdm

'''
The methods for landmarks locating
Run this file to test the alignment. the test image and mask 
'''


def locate_anatomical_axis(mask):

    cut_out = 0.10

    contour = mask2contour(mask)

    pixels = np.where(contour > 0)

    if len(pixels[0]) == 0:
        raise ValueError(
            "No bone pixels found while calculating anatomical axis."
        )

    x, y = pixels

    if len(x) < 20:
        raise ValueError(
            f"Too few bone pixels: {len(x)}"
        )

    # Sort by vertical coordinate
    idx = np.argsort(x)

    x_sorted = x[idx]
    y_sorted = y[idx]

    num_cut = int(len(x) * cut_out)

    if len(x) <= 2 * num_cut:
        num_cut = max(
            1,
            len(x) // 10
        )

    x_filtered = x_sorted[
        num_cut:-num_cut
    ]

    y_filtered = y_sorted[
        num_cut:-num_cut
    ]

    # Least-squares line
    line = Polynomial.fit(
        x_filtered,
        y_filtered,
        1
    )

    y_fit = line(x)

    return (
        x,
        y_fit
    )
def largest_component(mask):

    binary = (
        mask > 0
    ).astype(np.uint8)

    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    if num_labels <= 1:
        return binary * 255

    largest = 1 + np.argmax(
        stats[1:, cv.CC_STAT_AREA]
    )

    output = np.zeros_like(binary)

    output[
        labels == largest
    ] = 255

    return output


def locate_femur_head(mask):
    '''
    locate the center of femur head by fit a circle to femur head area
    :param mask: the rgb mask of unilateral long-leg x-ray
    :return: the center of femur head x_c, y_c, and radius of the circle
    '''
    pixels = np.where(mask > 0)
    # find the pixel bound of the femur
    max_x, max_y = np.max(pixels[0]), np.max(pixels[1])
    min_x, min_y = np.min(pixels[0]), np.min(pixels[1])
    # narrow down the region of femur head
    region_x = (min_x - 10, int(min_x + 0.15 * (max_x - min_x)))
    region_y = (int(min_y + 0.35 * (max_y - min_y)), max_y)
    # get the pixels in region of femur head
    region_mask = mask[region_x[0]:region_x[1] + 1, region_y[0]:region_y[1] + 1].copy()
    region_mask[:, :2] = 0
    # plot the region of interest (femur head)
    # plt.imshow(region_mask)
    # plt.show()
    dist_map = cv.distanceTransform(region_mask, cv.DIST_L2, cv.DIST_MASK_PRECISE)
    _, radius, _, center = cv.minMaxLoc(dist_map)
    y_c, x_c = center
    y_c = y_c + region_y[0]
    x_c = x_c + region_x[0]
    return (x_c, y_c), int(radius)


def locate_knee_center(mask):
    '''
    locate the center of the knee
    :param mask: the femur mask. should be a binary mask
    :return: the coordinates of the center of the knee
    '''
    # contour = mask2contour(mask)
    contour = np.where(mask > 0)
    contour_points = set()
    for i in range(len(contour[0])):
        contour_points.add((contour[0][i], contour[1][i]))

    femur_aa = locate_anatomical_axis(mask)
    aa_points = set()
    for i in range(len(femur_aa[0])):
        aa_points.add((femur_aa[0][i], int(femur_aa[1][i])))
        aa_points.add((femur_aa[0][i], round(femur_aa[1][i])))

    center = (0, 0)
    for i in list(aa_points.intersection(contour_points)):
        if i[0] > center[0]:
            center = i
    return center


def locate_tibia_center(mask):
    '''
    locate the center of the tibia, and the center of the ankle
    :param mask: the tibia mask. should be a binary mask
    :return: the coordinates of the center of the tibia and the center of the ankle
    '''
    # contour = mask2contour(mask)
    contour = np.where(mask > 0)
    contour_points = set()
    for i in range(len(contour[0])):
        contour_points.add((contour[0][i], contour[1][i]))

    tibia_aa = locate_anatomical_axis(mask)
    aa_points = set()
    for i in range(len(tibia_aa[0])):
        aa_points.add((tibia_aa[0][i], int(tibia_aa[1][i])))

    tibia_center = (1024, 1024)
    ankle_center = (0, 0)
    for i in list(aa_points.intersection(contour_points)):
        if i[0] < tibia_center[0]:
            tibia_center = i
        if i[0] > ankle_center[0]:
            ankle_center = i
    return tibia_center, ankle_center


def calculate_hka(
    femur_head,
    knee_center,
    tibia_center,
    ankle_center
):

    hip = np.array(
        femur_head,
        dtype=float
    )

    knee = np.array(
        knee_center,
        dtype=float
    )

    tibia = np.array(
        tibia_center,
        dtype=float
    )

    ankle = np.array(
        ankle_center,
        dtype=float
    )

    # Femoral mechanical axis:
    # knee -> hip

    femur_vector = hip - knee

    # Tibial mechanical axis:
    # knee/tibia -> ankle

    tibia_vector = ankle - tibia

    norm_femur = np.linalg.norm(
        femur_vector
    )

    norm_tibia = np.linalg.norm(
        tibia_vector
    )

    if norm_femur < 1e-6:
        raise ValueError(
            "Femoral mechanical axis has zero length."
        )

    if norm_tibia < 1e-6:
        raise ValueError(
            "Tibial mechanical axis has zero length."
        )

    cosine = np.dot(
        femur_vector,
        tibia_vector
    ) / (
        norm_femur *
        norm_tibia
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    angle = math.degrees(
        math.acos(cosine)
    )

    # Convert 180° anatomical angle
    # into deviation from a straight mechanical axis.

    hka_deviation = 180.0 - angle

    # Determine varus/valgus sign
    cross = (
        femur_vector[0] * tibia_vector[1]
        -
        femur_vector[1] * tibia_vector[0]
    )

    if cross > 0:
        hka_deviation = -hka_deviation

    return hka_deviation

def locate_mechanical_axis(mask):

    # ---------------------------------------------------------
    # Femur
    # ---------------------------------------------------------

    femur_mask = mask[:, :, 1].copy()

    # Femur region
    femur_mask[700:, :] = 0

    femur_mask = denoise(
        femur_mask,
        kernel=10
    )

    femur_mask = largest_component(
        femur_mask
    )

    # ---------------------------------------------------------
    # Tibia
    # ---------------------------------------------------------

    tibia_mask = mask[:, :, 0].copy()

    # Tibia region
    tibia_mask[:300, :] = 0

    tibia_mask = denoise(
        tibia_mask,
        kernel=10
    )

    tibia_mask = largest_component(
        tibia_mask
    )

    # ---------------------------------------------------------
    # Safety checks
    # ---------------------------------------------------------

    femur_pixels = np.sum(
        femur_mask > 0
    )

    tibia_pixels = np.sum(
        tibia_mask > 0
    )

    print(
        "    Femur pixels:",
        femur_pixels
    )

    print(
        "    Tibia pixels:",
        tibia_pixels
    )

    if femur_pixels < 100:
        raise ValueError(
            "Femur segmentation is too small."
        )

    if tibia_pixels < 100:
        raise ValueError(
            "Tibia segmentation is too small."
        )

    # ---------------------------------------------------------
    # Landmarks
    # ---------------------------------------------------------

    femur_head, r = locate_femur_head(
        femur_mask
    )

    knee_center = locate_knee_center(
        femur_mask
    )

    tibia_center, ankle_center = locate_tibia_center(
        tibia_mask
    )

    # ---------------------------------------------------------
    # ANATOMICAL AXES
    # ---------------------------------------------------------

    femur_aa = locate_anatomical_axis(
        femur_mask
    )

    tibia_aa = locate_anatomical_axis(
        tibia_mask
    )

    # ---------------------------------------------------------
    # HKA
    # ---------------------------------------------------------

    hka = calculate_hka(
        femur_head,
        knee_center,
        tibia_center,
        ankle_center
    )

    return (
        femur_head,
        r,
        knee_center,
        tibia_center,
        ankle_center,
        hka,
        femur_aa,
        tibia_aa
    )
def draw_mechanical_axis(img, mask, img_name=None, save_dir=None):
    '''
    Visualize segmentation, mechanical axes,
    anatomical axes, landmarks and HKA.

    Mechanical axis  = YELLOW solid line
    Anatomical axis  = CYAN dashed line
    '''

    (
        femur_head,
        r,
        knee_center,
        tibia_center,
        ankle_center,
        hka,
        femur_aa,
        tibia_aa
    ) = locate_mechanical_axis(mask)

    mask[:, :, 2] = 0

    dpi = 80

    height, width, depth = img.shape

    figsize = (
        width / float(dpi),
        height / float(dpi)
    )

    plt.figure(
        figsize=figsize
    )

    plt.axis("off")

    # ---------------------------------------------------------
    # X-ray image
    # ---------------------------------------------------------

    plt.imshow(
        img
    )

    # ---------------------------------------------------------
    # Segmentation mask
    # ---------------------------------------------------------

    plt.imshow(
        mask,
        cmap='jet',
        alpha=0.2
    )

    # =========================================================
    # FEMORAL HEAD
    # =========================================================

    circle = plt.Circle(
        (
            femur_head[1],
            femur_head[0]
        ),
        r,
        color='y',
        fill=False,
        linewidth=1.5
    )

    plt.gca().add_patch(
        circle
    )

    plt.plot(
        femur_head[1],
        femur_head[0],
        ",",
        color="y"
    )

    # =========================================================
    # LANDMARKS
    # =========================================================

    # Knee center
    plt.plot(
        knee_center[1],
        knee_center[0],
        "o",
        color="y",
        markersize=3
    )

    # Tibia center
    plt.plot(
        tibia_center[1],
        tibia_center[0],
        "o",
        color="y",
        markersize=3
    )

    # Ankle center
    plt.plot(
        ankle_center[1],
        ankle_center[0],
        "o",
        color="y",
        markersize=3
    )

    # =========================================================
    # MECHANICAL AXES
    # =========================================================

    # Femoral mechanical axis
    plt.plot(
        [
            femur_head[1],
            knee_center[1]
        ],
        [
            femur_head[0],
            knee_center[0]
        ],
        "-",
        color="yellow",
        linewidth=2.0,
        label="Femoral Mechanical Axis"
    )

    # Tibial mechanical axis
    plt.plot(
        [
            tibia_center[1],
            ankle_center[1]
        ],
        [
            tibia_center[0],
            ankle_center[0]
        ],
        "-",
        color="yellow",
        linewidth=2.0,
        label="Tibial Mechanical Axis"
    )

    # =========================================================
    # ANATOMICAL AXES
    # =========================================================

    # Femoral anatomical axis
    plt.plot(
        femur_aa[1],
        femur_aa[0],
        "--",
        color="cyan",
        linewidth=2.0,
        label="Femoral Anatomical Axis"
    )

    # Tibial anatomical axis
    plt.plot(
        tibia_aa[1],
        tibia_aa[0],
        "--",
        color="cyan",
        linewidth=2.0,
        label="Tibial Anatomical Axis"
    )

    # =========================================================
    # HKA TEXT
    # =========================================================

    plt.text(
        3,
        160,
        'HKA = {:.2f}'.format(hka),
        color='yellow',
        size=12
    )

    # =========================================================
    # SAVE
    # =========================================================

    plt.savefig(
        save_dir + img_name,
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0.0
    )

    plt.close('all')

if __name__ == '__main__':

    img_path = "./data/inference/images/"
    mask_path = "./data/inference/results/"
    save_path = "./data/inference/alignment/"

    os.makedirs(
        save_path,
        exist_ok=True
    )

    model_path = "exp_1_128x1024_dc_b=8_model.pth"

    model = torch.load(
        model_path,
        map_location="cuda",
        weights_only=False
    )

    model = model.cuda()

    inference(
        img_path,
        model,
        save_path=mask_path
    )

    img_names = os.listdir(
        img_path
    )

    for name in tqdm(img_names):

        try:

            img = cv.imread(
                os.path.join(
                    img_path,
                    name
                )
            )

            if img is None:
                print(
                    "Could not read:",
                    name
                )
                continue

            img = cv.cvtColor(
                img,
                cv.COLOR_BGR2RGB
            )

            mask = cv.imread(
                os.path.join(
                    mask_path,
                    name.rsplit(".", 1)[0] + ".png"
                )
            )

            if mask is None:
                print(
                    "Could not read mask:",
                    name
                )
                continue

            mask = cv.cvtColor(
                mask,
                cv.COLOR_BGR2RGB
            )

            draw_mechanical_axis(
                img,
                mask,
                img_name=name,
                save_dir=save_path
            )

        except Exception as e:

            print(
                f"\nFAILED: {name}"
            )

            print(
                "Reason:",
                e
            )