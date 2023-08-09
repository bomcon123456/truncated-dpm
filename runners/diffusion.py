import os
import logging
import time
import glob
import sys
import copy
from pathlib import Path

sys.path.append(
    "/lustre/scratch/client/guardpro/trungdt21/research/face_gen/truncated-diffusion-probabilistic-models/stylegan2-ada-pytorch"
)
sys.path.append(
    "/lustre/scratch/client/guardpro/trungdt21/research/face_gen/truncated-diffusion-probabilistic-models/"
)

import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import PIL

from models.diffusion import Model, Discriminator
from models.ema import EMAHelper
from functions import get_optimizer, get_g_optimizer, get_d_optimizer
from functions.losses import loss_registry
from functions.denoising import generalized_steps
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

# stylegan module
from train import setup_GAN_kwargs

import torchvision.utils as tvu
import matplotlib.pyplot as plt

import torchvision.transforms.functional as tvF
from torch_utils.ops import conv2d_gradfix


# stylegan2 lib
import dnnlib

plt.rcParams["savefig.bbox"] = "tight"

# -------------------------------------------------------------------------------------------------------------------
# from stylegan2 training_loop: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/training_loop.py
# -------------------------------------------------------------------------------------------------------------------


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], "L").save(fname)
    if C == 3:
        PIL.Image.fromarray(img, "RGB").save(fname)


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


def q_sample(x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0).to(x_0.device)
    alphas_t = extract(alphas_bar_sqrt, t, x_0)
    alphas_1_m_t = extract(one_minus_alphas_bar_sqrt, t, x_0)
    return alphas_t * x_0 + alphas_1_m_t * noise


def create_gan(img_size):
    G_kwargs = dnnlib.EasyDict(
        class_name="training.networks.Generator",
        z_dim=512,
        w_dim=512,
        mapping_kwargs=dnnlib.EasyDict(),
        synthesis_kwargs=dnnlib.EasyDict(),
    )
    res = img_size
    map = 2
    # fmaps = 1 if res >= 512 else 0.5
    fmaps = 1
    gan_lrate = 0.025
    G_kwargs.synthesis_kwargs.channel_base = int(fmaps * 32768)
    G_kwargs.synthesis_kwargs.channel_max = 512
    G_kwargs.mapping_kwargs.num_layers = map
    G_kwargs.synthesis_kwargs.num_fp16_res = 4  # enable mixed-precision training
    G_kwargs.synthesis_kwargs.conv_clamp = (
        256  # clamp activations to avoid float16 overflow
    )
    G_opt_kwargs = dnnlib.EasyDict(
        class_name="torch.optim.Adam", lr=gan_lrate, betas=[0, 0.99], eps=1e-8
    )
    common_kwargs = dict(c_dim=0, img_resolution=res, img_channels=3)
    G = (
        dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs)
        .train()
        .requires_grad_(True)
        .to("cuda:0")
    )  # subclass of torch.nn.Module
    return G


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        self.truncated_timestep = config.diffusion.truncated_timestep + 1

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        self.Dreg_interval = config.discriminator.Dreg_interval
        self.r1_gamma = config.discriminator.gamma

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            drop_last=True,
        )
        model = Model(config)

        discriminator = Discriminator(
            c_dim=0,
            img_resolution=config.data.image_size,
            img_channels=config.data.channels,
            channel_base=config.discriminator.channel_base,
        )

        # generator = create_gan(img_size=config.data.image_size)
        generator_kwargs = setup_GAN_kwargs(
            gpus=1,
            # Dataset.
            label_dim=0,
            # Base config.
            cfg=config.gan.cfg,  # Base config: 'auto' (default), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar'
            gamma=None,  # Override R1 gamma: <float>
            nobench=None,  # Disable cuDNN benchmarking: <bool>, default = False
        )
        common_kwargs = dict(
            c_dim=generator_kwargs.label_dim,
            img_resolution=config.data.image_size,
            img_channels=config.data.channels,
        )
        generator = (
            dnnlib.util.construct_class_by_name(**generator_kwargs.G_kwargs, **common_kwargs)
            .train()
            .requires_grad_(False)
            .to(self.device)
        )  # subclass of torch.nn.Module
        generator_ema = copy.deepcopy(generator).eval()

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        discriminator = discriminator.to(self.device)
        discriminator = torch.nn.DataParallel(discriminator)

        optimizer = get_optimizer(self.config, model.parameters())
        optimizer_d = get_d_optimizer(self.config, discriminator.parameters())
        optimizer_g = get_g_optimizer(self.config, generator.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[2]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[2])
            start_epoch = states[-3]
            step = states[-2]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[-1])

        label = torch.zeros([1, generator.c_dim], device=self.device)
        for epoch in range(start_epoch, self.config.training.n_epochs):
            epoch_start_time = data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.truncated_timestep, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.truncated_timestep - t - 1], dim=0)[:n]
                t_max = torch.tensor([self.truncated_timestep]).to(self.device)
                loss = loss_registry[config.model.type](model, x, t, e, b)

                optimizer.zero_grad()
                loss.backward()
                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                # implicit term
                # z_si = torch.randn_like(x).to(self.device)
                discriminator.requires_grad_(False)
                generator.requires_grad_(True)
                z_si = torch.randn([x.shape[0], generator.z_dim], device=self.device)
                x_gen_prime_implicit = generator(z_si, label)

                x_fake_logits = discriminator(x_gen_prime_implicit, c=0)
                loss_T = torch.nn.functional.softplus(-x_fake_logits).mean()

                tb_logger.add_scalar("implicit loss", loss_T, global_step=step)

                tb_logger.add_scalar("loss", loss, global_step=step)
                logging.info(
                    f"Epoch: {epoch}, step: {step}, loss: {loss.item()}, implicit loss: {loss_T.item()}, data time: {data_time / (i+1)}"
                )
                optimizer_g.zero_grad()
                loss_T.backward()
                optimizer_g.step()

                # Update G_EMA
                ema_nimg = generator_kwargs.ema_kimg * 1000
                ema_beta = 0.5 ** (config.training.batch_size / max(ema_nimg, 1e-8))
                for p_ema, p in zip(generator_ema.parameters(), generator.parameters()):
                    p_ema.copy_(p.lerp(p_ema, ema_beta))
                for b_ema, b in zip(generator_ema.buffers(), generator.buffers()):
                    b_ema.copy_(b)

                # update discriminator
                discriminator.requires_grad_(True)
                generator.requires_grad_(False)
                optimizer_d.zero_grad()

                do_Dr1 = ((step % self.Dreg_interval) == 0)
                z_si = torch.randn([x.shape[0], generator.z_dim], device=self.device)
                # z_si = torch.randn_like(x).to(self.device)
                x_t_implicit = (
                    q_sample(
                        x, self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t_max
                    )
                    .detach()
                    .requires_grad_(do_Dr1)
                )

                x_t_gen_implicit = generator(z_si, label)

                gen_logits = discriminator(x_t_gen_implicit, c=0)
                loss_Dgen = torch.nn.functional.softplus(gen_logits).mean()
                loss_Dgen.backward()
                tb_logger.add_scalar("DLoss/Dgen", loss_Dgen, global_step=step)

                real_logits = discriminator(x_t_implicit, c=0)
                loss_Dreal = torch.nn.functional.softplus(-real_logits).mean()
                if not do_Dr1:
                    loss_Dreal.backward()
                tb_logger.add_scalar("DLoss/Dreal", loss_Dreal, global_step=step)


                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function(
                        "r1_grads"
                    ), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(
                            outputs=[real_logits.sum()],
                            inputs=[x_t_implicit],
                            create_graph=True,
                            only_inputs=True,
                        )[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = (r1_penalty * (self.r1_gamma / 2)).mean()
                    (loss_Dreal + loss_Dr1.mul(self.Dreg_interval)).backward()
                    tb_logger.add_scalar("DLoss/r1_penalty", r1_penalty.mean(), global_step=step)
                    tb_logger.add_scalar("DLoss/reg", loss_Dr1, global_step=step)

                # d_loss = loss_Dreal + loss_Dgen + loss_Dr1

                # d_loss.backward()
                optimizer_d.step()

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        generator.state_dict(),
                        generator_ema.state_dict(),
                        optimizer.state_dict(),
                        optimizer_d.state_dict(),
                        optimizer_g.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

            image_save_folder = Path(self.config.imagelog_path)
            tvu.save_image(
                x_t_implicit.detach().cpu(), (image_save_folder / f"xreal_{epoch}.jpg"), normalize=True
            )
            with torch.no_grad():
                x_t_gen_implicit_ema = generator_ema(z_si, label)
                tvu.save_image(
                    x_t_gen_implicit_ema.detach().cpu(),
                    (image_save_folder / f"xgen_{epoch}.jpg").as_posix(),
                    normalize=True,
                )

            logging.info(
                f"Epoch: {epoch}, epoch training time: {time.time() - epoch_start_time}"
            )

    def sample(self):
        model = Model(self.config)

        generator_kwargs = setup_GAN_kwargs(
            gpus=1,
            # Dataset.
            label_dim=0,
            # Base config.
            cfg=self.config.gan.cfg,  # Base config: 'auto' (default), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar'
            gamma=None,  # Override R1 gamma: <float>
            nobench=None,  # Disable cuDNN benchmarking: <bool>, default = False
        )
        common_kwargs = dict(
            c_dim=generator_kwargs.label_dim,
            img_resolution=self.config.data.image_size,
            img_channels=self.config.data.channels,
        )
        generator = (
            dnnlib.util.construct_class_by_name(**generator_kwargs.G_kwargs, **common_kwargs)
            .train()
            .requires_grad_(False)
            .to(self.device)
        )  # subclass of torch.nn.Module

        states = torch.load(
            os.path.join(
                self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
            ),
            map_location=self.config.device,
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)
        generator.load_state_dict(states[2], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
        else:
            ema_helper = None

        model.eval()
        generator.eval()

        if self.args.fid:
            self.sample_fid(model, generator)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model, generator):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size
        label = torch.zeros([1, generator.c_dim], device=self.device)

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn([n, generator.z_dim], device=self.device)
                # x = torch.randn(
                #     n,
                #     config.data.channels,
                #     config.data.image_size,
                #     config.data.image_size,
                #     device=self.device,
                # )

                x = generator(x, label)
                # flipped_x = torch.flip(x, [3])

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                # flipped_x = self.sample_image(flipped_x, model)
                # flipped_x = inverse_data_transform(config, flipped_x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    # tvu.save_image(
                    #     flipped_x[i], os.path.join(self.args.image_folder, f"{img_id}_flipped.png")
                    # )
                    img_id += 1

    def sample_image(self, x, model, last=True):
        seq = range(0, self.truncated_timestep)
        xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
        x = xs
        if last:
            x = x[0][-1]
        return x

    def generate_truncated(self):
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            drop_last=False,
        )
        outpath = Path(config.generate_truncated.outpath)
        outpath.mkdir(exist_ok=True, parents=True)
        t_max = torch.tensor([self.truncated_timestep])
        for i, (x, y) in enumerate(tqdm.tqdm(train_loader)):
            x_noised = q_sample(
                x, self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t_max
            )
            # x_noised = inverse_data_transform(config, x_noised)
            for j in range(x.shape[0]):
                tvu.save_image(
                    x_noised[j], (outpath / f"{str(i*config.training.batch_size + j).zfill(8)}.png")
                )
        
    def denoise(self):
        model = Model(self.config)

        states = torch.load(
            self.config.ckpt_path,
            map_location=self.config.device,
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
        else:
            ema_helper = None

        model.eval()
        test_transform = transforms.Compose(
            [transforms.Resize(self.config.data.image_size), transforms.ToTensor()]
        )
        outpath = self.config.denoise.out_path
        outpath.mkdir(exist_ok=True, parents=True)
        with torch.no_grad():
            for image_path in tqdm.tqdm(
                config.denoise.image_paths
            ):
                image = PIL.Image.open(image_path)
                image = test_transform(image)
                denoised_image = self.sample_image(model, image)
                denoised_image = inverse_data_transform(config, denoised_image)
                tvu.save_image(
                    denoised_image, (outpath / f"{image_path.stem}.png")
                )
                

if __name__ == "__main__":
    G = create_gan()
    print(G)
