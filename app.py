from pyexpat import model
import lightning as L
import gradio as gr
from lightning.app.components.serve import ServeGradio

import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


class ModelSampler:
    def __init__(self, model, sampler) -> None:
        self.model = model
        self.sampler = sampler


class ModelDemo(ServeGradio):

    inputs = gr.inputs.Textbox()
    outputs = gr.outputs.Image(type="pil")

    ddim_steps=50
    plms = True
    laion400m = False
    fixed_code = False
    ddim_eta=0.0
    n_iter = 2
    H, W, C = 512, 512, 4
    f = 8
    n_samples=3
    n_rows =0
    scale = 7.5

    config="configs/stable-diffusion/v1-inference.yaml"
    ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"
    seed=42
    precision = "autocast"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    def predict(self, prompt:str)->Image.Image:
        batch_size = self.n_samples
        n_rows = self.n_rows if self.n_rows > 0 else batch_size
        assert prompt is not None
        data = [batch_size * [prompt]]

        start_code = None
        if self.fixed_code:
            start_code = torch.randn([self.n_samples, self.C, self.H // self.f, self.W // self.f], device=self.device)

        precision_scope = autocast if self.precision=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in trange(self.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if self.scale != 1.0:
                                uc = self.model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)
                            shape = [self.C, self.H // self.f, self.W // self.f]
                            samples_ddim, _ = self.sampler.sample(S=self.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=self.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=self.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=self.ddim_eta,
                                                            x_T=start_code)

                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            all_samples.append(x_samples_ddim)

                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    return Image.fromarray(grid.astype(np.uint8))


    def build_model(self):
        seed_everything(self.seed)

        config = OmegaConf.load(f"{self.config}")
        model = load_model_from_config(config, f"{self.ckpt}")
        model = model.to(self.device)

        if self.plms:
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)
        return ModelSampler(model, sampler)

class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.demo = ModelDemo()

    def run(self):
        self.demo.run()
    
    def configure_layout(self):
        return {"name": "Model demo", "content": self.demo}

if __name__=="__main__":
    app = L.LightningApp(RootFlow())
