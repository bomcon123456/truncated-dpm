import typer


from runners.diffusion import Diffusion
from main import dict2namespace

app = typer.Typer()

@app.command()
def main(
    config_name: str = typer.Argument(..., help="config name"),
    image_path: Path = typer.Argument(..., help="Image path", exists=True),
    out_path: Path = typer.Argument(..., help="Output path"),
    ckpt_path: Path = typer.Option(None, help="override ckpt path", exists=True),
):
    with open(os.path.join("configs", config_name), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    args = {}
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device
    
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True
    conv2d_gradfix.enabled = True

    if image_path.is_dir():
        image_paths = image_path.rglob("*.[pj][np]g")
    else:
        image_paths = [image_path]

    new_config.denoise.image_paths = image_paths
    new_config.denoise.out_path = out_path
    if ckpt_path is not None and ckpt_path.exists():
        new_config.denoise.ckpt_path = ckpt_path

    runner = Diffusion(args, config)
    runner.denoise()

if __name__ == "__main__":
    app()