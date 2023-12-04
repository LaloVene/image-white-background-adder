from pathlib import Path
from rembg import remove
import cv2
from cv2 import dnn_superres
import json


# Replace transparent pixels with white
def replace(img):
    img = img.copy()
    # make mask of where the transparent bits are
    trans_mask = img[:, :, 3] <= 10
    border_mask = (img[:, :, 3] > 10) & (img[:, :, 3] <= 40)

    # replace areas of transparency with white and not transparent
    img[trans_mask] = [255, 255, 255, 255]

    # Apply Gaussian blur to the border to smooth it
    border = cv2.GaussianBlur(img, (0, 0), 3)
    img[border_mask] = border[border_mask]

    # new image without alpha channel...
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


# Set Input / Output Paths
INPUT_PATH = "/mnt/c/Users/laloh/Documents/Mercado Libre/Backup/images/"
OUTPUT_PATH = Path(INPUT_PATH + "/images_no_bg_low")

# Create Output Path if it doesn't exist
if not OUTPUT_PATH.is_dir():
    OUTPUT_PATH.mkdir()

with open("./product-images-empty.json", "r") as json_file:
    products = json.load(json_file)

# Loop through all images in Input Path
counter = 0
for file in Path(INPUT_PATH).glob("*.jpg"):
    file_name = file.stem.split("_image")[0].strip().replace("_", "/")
    try:
        if len(products[file_name]) > 1:
            continue
    except KeyError:
        continue
    input_path = str(file)
    output_path = str(OUTPUT_PATH / (file.stem + ".png"))

    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    # Create an SR object
    sr = dnn_superres.DnnSuperResImpl_create()

    # Set the desired model and scale to get correct pre- and post-processing
    sr.readModel("./FSRCNN-small_x4.pb")
    sr.setModel("fsrcnn", 4)

    # Upscale the image
    image = sr.upsample(image)

    # remove background
    image = remove(image)
    image = replace(image)

    # save new image
    cv2.imwrite(output_path, image)
    print(file.stem + ".png " + image.shape.__str__())
