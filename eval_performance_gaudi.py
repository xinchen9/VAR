import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
from torch.profiler import profile, record_function, ProfilerActivity

setattr(torch.nn.Linear, 'reset_parameters', lambda self: None) 
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

from models import VQVAE, build_vae_var

depths = {30}
hf_home = "/mnt/disk4/xinchen/models/var"
vae_ckpt = 'vae_ch160v4096z32.pth'
vae_ckpt = hf_home + '/' + vae_ckpt

patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


for model_depth in depths:
    print(f"model depth = {model_depth}")
    if 'vae' not in globals() or 'var' not in globals():
        vae, var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
            device=device, patch_nums=patch_nums,
            num_classes=1000, depth=model_depth, shared_aln=False,
        )
    var_ckpt = f'var_d{model_depth}.pth'
    var_ckpt = hf_home + '/' + var_ckpt 

    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')  

    ############################# 2. Sample with classifier-free guidance

# set args
    seed = 0 #@param {type:"number"}
    torch.manual_seed(seed)
    num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
    cfg = 4 #@param {type:"slider", min:1, max:10, step:0.1}
    class_labels = (980, 980, 437, 437, 22, 22, 562, 562)  #@param {type:"raw"}
    more_smooth = False # True for more smooth output

    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    # sample
    B = len(class_labels)
    activities = [ProfilerActivity.CUDA]
    sort_by_keyword = "self_" + device + "_time_total"
    label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("model_inference"):
            with torch.inference_mode():
                with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                    recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)

    chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
    chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
    chw = PImage.fromarray(chw.astype(np.uint8))
    img_name = f'image_{model_depth}.png'
    chw.save(img_name)
    key_averages_filename=f'key_averages_{model_depth}.txt'
    with open(key_averages_filename, "w") as f:
        f.write(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))
    print(f"Key averages saved to '{key_averages_filename}'")
    f.close()
    
    print(f"finish save image at {model_depth}")

    # import pdb
    # pdb.set_trace()
