from tqdm import tqdm
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
from datasets import Crop

cx = 89
cy = 121
x1 = cy - 64
x2 = cy + 64
y1 = cx - 64
y2 = cx + 64
transform=transforms.Compose(
    [
        Crop(x1, x2, y1, y2),
        transforms.Resize((64,64))
    ]
)
image_dir = Path("/lustre/scratch/client/guardpro/trungdt21/research/face_gen/truncated-diffusion-probabilistic-models/data/celeba/img_align_celeba")
image_list = list(image_dir.rglob("*.jpg"))
outdir = Path("/lustre/scratch/client/guardpro/trungdt21/research/face_gen/truncated-diffusion-probabilistic-models/data/celeba/img_align_celeba_cropped")
outdir.mkdir(exist_ok=True, parents=True)
for image_path in tqdm(image_list):
    img = Image.open(image_path.as_posix())
    cropped = transform(img)
    assert cropped.size == (64, 64), f"crop size: {cropped.size}"
    outpath = outdir / image_path.relative_to(image_dir).with_suffix(".png")
    cropped.save(outpath.as_posix())
