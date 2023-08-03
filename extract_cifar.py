from torchvision.datasets import CIFAR10
import torchvision.utils as tvu

dataset = CIFAR10(
    "/lustre/scratch/client/guardpro/trungdt21/research/face_gen/truncated-diffusion-probabilistic-models/out/datasets/cifar10",
    train=True,
    download=False,
)
for i, (image,_) in enumerate(dataset):
    image.save(f"/lustre/scratch/client/guardpro/trungdt21/research/face_gen/truncated-diffusion-probabilistic-models/out/datasets/cifar10/images/{i}.png")