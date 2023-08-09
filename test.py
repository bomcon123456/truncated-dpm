import typer
from natsort import natsorted
import torch
import numpy as np
import yaml
import os
from pathlib import Path
from torch_utils.ops import conv2d_gradfix

from runners.diffusion import Diffusion
from main import dict2namespace

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
    config_name: str = typer.Argument(..., help="config name"),
    image_path: Path = typer.Argument(..., help="Image path", exists=True),
    out_path: Path = typer.Argument(..., help="Output path"),
    ckpt_path: Path = typer.Option(None, help="override ckpt path", exists=True),
    truncated_step: int = typer.Option(None, "--truncated-step", "-s", help="truncated step"),
    seed: int = typer.Option(0, help="seed"),
    eta: float = typer.Option(0.0, help="eta"),
):
    with open(os.path.join("configs", config_name), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    print(new_config)
    args = dict2namespace(dict(eta=eta, seed=seed))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    new_config.device = device
    
    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    conv2d_gradfix.enabled = True

    if image_path.is_dir():
        image_paths = natsorted(list(image_path.rglob("*.[pj][np]g")))
    else:
        image_paths = [image_path]

    new_config.denoise.image_paths = image_paths
    new_config.denoise.out_path = out_path
    if ckpt_path is not None and ckpt_path.exists():
        new_config.denoise.ckpt_path = ckpt_path
    if truncated_step is not None:
        new_config.diffusion.truncated_timestep = truncated_step

    runner = Diffusion(args, new_config)
    runner.denoise()

if __name__ == "__main__":
    app()