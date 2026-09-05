import torch
import PIL.Image as Image
from torchvision import transforms
import cv2 as cv
from utils.prediction2rgbmask import pred2mask
import os
import numpy as np


img_path = "./data/inference/images"
save_path = "./data/inference/results"
model_path = "exp_1_128x1024_dc_b=8_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_single_leg(img):
    """
    Preprocess a SINGLE-LEG full-length X-ray.

    Target:
        height = 1024
        width  = 128

    The original project preprocesses bilateral X-rays into
    128 x 1024 unilateral images before training/inference.
    """

    # Convert to numpy grayscale
    img = np.array(img)

    # Make sure uint8
    if img.dtype != np.uint8:
        img = cv.normalize(
            img,
            None,
            0,
            255,
            cv.NORM_MINMAX
        ).astype(np.uint8)

    original_h, original_w = img.shape

    print(
        f"    Original image: "
        f"{original_w} x {original_h}"
    )

    # ---------------------------------------------------------
    # 1. Resize HEIGHT to 1024
    # ---------------------------------------------------------

    target_h = 1024

    scale = target_h / original_h

    new_w = int(round(original_w * scale))

    img = cv.resize(
        img,
        (new_w, target_h),
        interpolation=cv.INTER_AREA
    )

    # ---------------------------------------------------------
    # 2. Make width exactly 128
    # ---------------------------------------------------------

    target_w = 128

    if new_w > target_w:

        # Center crop
        start = (new_w - target_w) // 2

        img = img[
            :,
            start:start + target_w
        ]

    else:

        # Pad left/right
        total_pad = target_w - new_w

        left = total_pad // 2
        right = total_pad - left

        img = cv.copyMakeBorder(
            img,
            0,
            0,
            left,
            right,
            cv.BORDER_CONSTANT,
            value=0
        )

    # ---------------------------------------------------------
    # 3. CLAHE
    # ---------------------------------------------------------

    clahe = cv.createCLAHE(
        clipLimit=1.0,
        tileGridSize=(8, 8)
    )

    img = clahe.apply(img)

    print(
        f"    Network input: "
        f"{img.shape[1]} x {img.shape[0]}"
    )

    return Image.fromarray(img)


def inference(img_path, model, save_path):

    model.eval()
    model.to(device)

    os.makedirs(save_path, exist_ok=True)

    image_names = [
        x for x in os.listdir(img_path)
        if x.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ]

    with torch.no_grad():

        for name in image_names:

            img_dir = os.path.join(
                img_path,
                name
            )

            print("\nProcessing:", name)

            # -------------------------------------------------
            # Read image
            # -------------------------------------------------

            img = Image.open(
                img_dir
            ).convert("L")

            # -------------------------------------------------
            # IMPORTANT:
            # Apply training-style preprocessing
            # -------------------------------------------------

            img_processed = preprocess_single_leg(img)

            # -------------------------------------------------
            # Convert to tensor
            # -------------------------------------------------

            img_tensor = transforms.ToTensor()(
                img_processed
            )

            img_tensor = img_tensor.unsqueeze(0)

            img_tensor = img_tensor.to(device)

            # -------------------------------------------------
            # Network
            # -------------------------------------------------

            pred = model(img_tensor)

            # -------------------------------------------------
            # Convert prediction to RGB mask
            # -------------------------------------------------

            mask = pred2mask(pred)

            mask = (
                mask.cpu()
                .numpy()
                * 255
            ).astype(np.uint8)

            # RGB -> BGR
            mask = cv.cvtColor(
                mask,
                cv.COLOR_RGB2BGR
            )

            # -------------------------------------------------
            # Save
            # -------------------------------------------------

            output_path = os.path.join(
                save_path,
                name.rsplit(".", 1)[0] + ".png"
            )

            cv.imwrite(
                output_path,
                mask
            )

            print(
                "    Saved:",
                output_path
            )


if __name__ == "__main__":

    print("Device:", device)

    print(
        "Loading pretrained model..."
    )

    model = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    print(
        "Model:",
        type(model)
    )

    inference(
        img_path,
        model,
        save_path
    )

    print("\nInference completed.")