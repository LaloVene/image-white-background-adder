from pathlib import Path
from rembg import remove
import cv2


# Replace transparent pixels with white
def replace(img):
    img = img.copy()
    # make mask of where the transparent bits are
    trans_mask = img[:, :, 3] <= 10
    border_mask = (img[:, :, 3] > 10) & (img[:, :, 3] <= 40)

    # replace areas of transparency with white and not transparent
    img[trans_mask] = [255, 255, 255, 255]

    # Apply Gaussian blur to the border to smooth it
    border = cv2.GaussianBlur(img, (0, 0), 10)
    img[border_mask] = border[border_mask]

    # new image without alpha channel...
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


# Set Input / Output Paths
INPUT_PATH = ""
OUTPUT_PATH = Path(INPUT_PATH + "/images_no_bg")

# Create Output Path if it doesn't exist
if not OUTPUT_PATH.is_dir():
    OUTPUT_PATH.mkdir()

# Loop through all images in Input Path
for file in Path(INPUT_PATH).glob("*.[jpg][jpeg][png]"):
    input_path = str(file)
    output_path = str(OUTPUT_PATH / (file.stem + ".png"))

    # remove background
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    image = remove(image)
    image = replace(image)

    # save new image
    cv2.imwrite(output_path, image)
    print(file.stem + ".png")
