from safetensors.torch import load_file
import random
from datetime import datetime
import torch
from diffusers import StableDiffusionXLPipeline


def load_safetensors():
    f = load_file("C:\\noobai\\unet\\diffusion_pytorch_model.safetensors")
    print(f.keys())


def dump_civitai_to_diffusers_models():
    pipe = StableDiffusionXLPipeline.from_single_file("C:\\noobai\\unet\\diffusion_pytorch_model.safetensors")
    pipe.save_pretrained("C:\\noobai\\dump")


def diffusers_text_to_img():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "C:\\noobai\\dump", use_safetensors=True, torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.unet.set_default_attn_processor()
    pipe.vae.set_default_attn_processor()
    seed = random.randint(0, 2 ** 32)
    generator = torch.Generator("cuda").manual_seed(seed)
    lora_weight = load_file('./mari_NoobXL.safetensors')
    pipe.load_lora_weights(lora_weight, 'mari_NoobXL')
    pipe.fuse_lora()
    image = pipe("mari_(blue_archive),kani_biimu,ningen_mame,ciloranko", generator=generator,
                 num_inference_steps=28, width=768, height=1344).images[0]
    now_str = datetime.now().strftime('%Y%m%d%H%M%S')
    image.save('noob_%s.png' % now_str)


def main():
    diffusers_text_to_img()


if __name__ == '__main__':
    main()
