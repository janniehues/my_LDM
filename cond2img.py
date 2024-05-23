import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
import argparse,os
from tqdm import trange
from taming.data.imagenet import retrieve

"""
And go. Quality, sampling speed and diversity are best controlled via the `scale`, 
`ddim_steps` and `ddim_eta` variables. As a rule of thumb, higher values of `scale`
produce better samples at the cost of a reduced output diversity. Furthermore, 
increasing `ddim_steps` generally also gives higher quality samples, but returns
are diminishing for values > 250. Fast sampling (i e. low values of `ddim_steps`)
while retaining good quality can be achieved by using `ddim_eta = 0.0`.
"""
def load_model_from_config(config, ckpt, verbose=True):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
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

#def load_model_from_config(config, ckpt):
#    print(f"Loading model from {ckpt}")
#    pl_sd = torch.load(ckpt)#, map_location="cpu")
#    sd = pl_sd["state_dict"]
#    model = instantiate_from_config(config.model)
#    m, u = model.load_state_dict(sd, strict=False)
#    model.cuda()
#    model.eval()
#    return model


#def get_model():
#    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")  
#    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
#    return model

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/cond2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="skip grid generation",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

#    parser.add_argument(
#        "--H",
#        type=int,
#        default=256,
#        help="image height, in pixel space",
#    )
#
#    parser.add_argument(
#        "--W",
#        type=int,
#        default=256,
#        help="image width, in pixel space",
#    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="path to model config file",
    )
    
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to model ckpt file",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--classes",
        nargs="*",
        type=int,
        default=None,
        help="Classes to sample for, separated by spaces",
    )


    parser.add_argument(
        "--f_down",
        type=int,
        default=None,
        help="Scaling factor of encoder stage",
    )

    parser.add_argument(
        "--z",
        nargs="*",
        type=int,
        default=None,
        help="Embedded dimensions",
    )
    
    opt = parser.parse_args()
    

    config = OmegaConf.load(opt.cfg)
    if not type(config)==dict:
        config_dict = OmegaConf.to_container(config)
    else:
        config_dict=config

    config_cond_stage=config_dict['model']['params']['cond_stage_config']['params']
    n_classes = retrieve(config_cond_stage, "n_classes", default=1000)
    n_classes=n_classes-1
    image_size = config_dict['model']['params']['unet_config']['params']['image_size']
    embed_dim = config_dict['model']['params']['first_stage_config']['params']['embed_dim']
    #load model and put on gpu
    model = load_model_from_config(config,opt.ckpt)  # TODO: check path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        print("Uing PLMSSampler")
        sampler = PLMSSampler(model)
    else:
        print("Using DDIMSampler")
        sampler = DDIMSampler(model)

    
    import numpy as np 
    from PIL import Image
    from einops import rearrange
    from torchvision.utils import make_grid

    #TODO add assert statement for shape in runcard and here
    shape = [embed_dim, image_size, image_size]

    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(opt.n_samples*[n_classes]).to(model.device)}
                )

            for class_label in opt.classes:
                all_samples = list()
                
                print(f"Rendering {opt.n_samples} examples of class '{class_label}' \
                    in {opt.ddim_steps} steps and using unconditional_guidance_scale={opt.scale:.2f}.")
                xc = torch.tensor(opt.n_samples*[class_label])
                cond = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                
                # make outdir if necessary
                outdir=os.path.join(opt.outdir,str(class_label))
                os.makedirs(outdir, exist_ok=True)
                sample_path = os.path.join(outdir, "samples")
                os.makedirs(sample_path, exist_ok=True)
                base_count = len(os.listdir(sample_path))

                for n in trange(opt.n_iter, desc="Sampling"):
                    #make dirs if necessary
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=cond,
                                                     batch_size=opt.n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                        base_count += 1
                    all_samples.append(x_samples_ddim)

                # display as grid
                if opt.skip_grid:
                    pass
                else:
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=opt.n_samples)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()   
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outdir, f'grid_{class_label}.png'))
