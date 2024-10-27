import os
import hashlib
import collections
import safetensors
import torch
import enum
from omegaconf import OmegaConf, ListConfig
from sgm import DiffusionEngine
import sd_models_xl
import sd_disable_initialization

model_dir = "Stable-diffusion"
model_path = "C:/c_git_project/sd-webui-aki/models/Stable-diffusion"
abs_ckpt_dir = "C:/c_git_project/sd-webui-aki/models/Stable-diffusion"
checkpoint_config = 'C:/c_git_project/sd-webui-aki/repositories/generative-models/configs/inference/sd_xl_base.yaml'

checkpoints_list = {}
checkpoint_aliases = {}
checkpoint_alisases = checkpoint_aliases  # for compatibility with old name
checkpoints_loaded = collections.OrderedDict()


def load_model(ckpt_file_name: str):
    checkpoint_info = CheckpointInfo(ckpt_file_name)
    state_dict = get_checkpoint_state_dict(checkpoint_info)
    sd_config = OmegaConf.load(checkpoint_config)

    clip_is_included_into_sd = True

    de_params = {**sd_config['model'].get("params", {})}
    try:
        with sd_disable_initialization.DisableInitialization(disable_clip=clip_is_included_into_sd):
            with sd_disable_initialization.InitializeOnMeta():
                sd_model = DiffusionEngine(**de_params)
    except Exception as e:
        return
    sd_model.used_config = checkpoint_config
    load_model_weights(sd_model, checkpoint_info, state_dict)
    return sd_model


class ModelType(enum.Enum):
    SD1 = 1
    SD2 = 2
    SDXL = 3
    SSD = 4
    SD3 = 5


def set_model_type(model, state_dict):
    model.is_sd1 = False
    model.is_sd2 = False
    model.is_sdxl = False
    model.is_ssd = False
    model.is_sd3 = False

    if "model.diffusion_model.x_embedder.proj.weight" in state_dict:
        model.is_sd3 = True
        model.model_type = ModelType.SD3
    elif hasattr(model, 'conditioner'):
        model.is_sdxl = True

        if 'model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight' not in state_dict.keys():
            model.is_ssd = True
            model.model_type = ModelType.SSD
        else:
            model.model_type = ModelType.SDXL
    elif hasattr(model.cond_stage_model, 'model'):
        model.is_sd2 = True
        model.model_type = ModelType.SD2
    else:
        model.is_sd1 = True
        model.model_type = ModelType.SD1


def set_model_fields(model):
    if not hasattr(model, 'latent_channels'):
        model.latent_channels = 4


class CheckpointInfo:
    def __init__(self, filename):
        self.filename = filename
        abspath = os.path.abspath(filename)
        self.is_safetensors = os.path.splitext(filename)[1].lower() == ".safetensors"

        if abs_ckpt_dir and abspath.startswith(abs_ckpt_dir):
            name = abspath.replace(abs_ckpt_dir, '')
        elif abspath.startswith(model_path):
            name = abspath.replace(model_path, '')
        else:
            name = os.path.basename(filename)

        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        def read_metadata():
            metadata = read_metadata_from_safetensors(filename)
            self.modelspec_thumbnail = metadata.pop('modelspec.thumbnail', None)

            return metadata

        self.metadata = read_metadata()
        self.name = name
        self.name_for_extra = os.path.splitext(os.path.basename(filename))[0]
        self.model_name = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]
        self.hash = model_hash(filename)

        self.sha256 = hashes_sha256(self.filename + f"checkpoint/{name}")
        self.shorthash = self.sha256[0:10] if self.sha256 else None

        self.title = name if self.shorthash is None else f'{name} [{self.shorthash}]'
        self.short_title = self.name_for_extra if self.shorthash is None else f'{self.name_for_extra} [{self.shorthash}]'

        self.ids = [self.hash, self.model_name, self.title, name, self.name_for_extra, f'{name} [{self.hash}]']
        if self.shorthash:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]',
                         f'{self.name_for_extra} [{self.shorthash}]']

    def register(self):
        checkpoints_list[self.title] = self
        for id in self.ids:
            checkpoint_aliases[id] = self

    def calculate_shorthash(self):
        self.sha256 = hashes_sha256(self.filename + f"checkpoint/{self.name}")
        if self.sha256 is None:
            return

        shorthash = self.sha256[0:10]
        if self.shorthash == self.sha256[0:10]:
            return self.shorthash

        self.shorthash = shorthash

        if self.shorthash not in self.ids:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]',
                         f'{self.name_for_extra} [{self.shorthash}]']

        old_title = self.title
        self.title = f'{self.name} [{self.shorthash}]'
        self.short_title = f'{self.name_for_extra} [{self.shorthash}]'

        replace_key(checkpoints_list, old_title, self.title, self)
        self.register()

        return self.shorthash


def replace_key(d, key, new_key, value):
    keys = list(d.keys())

    d[new_key] = value

    if key not in keys:
        return d

    index = keys.index(key)
    keys[index] = new_key

    new_d = {k: d[k] for k in keys}

    d.clear()
    d.update(new_d)
    return d


def hashes_sha256(text: str):
    h = hashlib.sha256()
    h.update(text.encode())
    return h.hexdigest()


def model_hash(filename):
    """old hash that only looks at a small part of the file and is prone to collisions"""

    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'


def read_metadata_from_safetensors(filename):
    import json

    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"

        res = {}

        try:
            json_data = json_start + file.read(metadata_len - 2)
            json_obj = json.loads(json_data)
            for k, v in json_obj.get("__metadata__", {}).items():
                res[k] = v
                if isinstance(v, str) and v[0:1] == '{':
                    try:
                        res[k] = json.loads(v)
                    except Exception:
                        pass
        except Exception:
            pass

        return res


def read_state_dict(checkpoint_file):
    return safetensors.torch.load_file(checkpoint_file)


def get_checkpoint_state_dict(checkpoint_info: CheckpointInfo):
    sd_model_hash = checkpoint_info.calculate_shorthash()

    if checkpoint_info in checkpoints_loaded:
        # use checkpoint cache
        print(f"Loading weights [{sd_model_hash}] from cache")
        # move to end as latest
        checkpoints_loaded.move_to_end(checkpoint_info)
        return checkpoints_loaded[checkpoint_info]

    print(f"Loading weights [{sd_model_hash}] from {checkpoint_info.filename}")
    res = read_state_dict(checkpoint_info.filename)
    return res


def apply_alpha_schedule_override(sd_model, p=None):
    """
    Applies an override to the alpha schedule of the model according to settings.
    - downcasts the alpha schedule to half precision
    - rescales the alpha schedule to have zero terminal SNR
    """

    if not hasattr(sd_model, 'alphas_cumprod') or not hasattr(sd_model, 'alphas_cumprod_original'):
        return

    sd_model.alphas_cumprod = sd_model.alphas_cumprod_original.to(torch.device("cuda"))


def load_model_weights(model, checkpoint_info: CheckpointInfo, state_dict):
    sd_model_hash = checkpoint_info.calculate_shorthash()

    set_model_type(model, state_dict)
    set_model_fields(model)

    if model.is_sdxl:
        sd_models_xl.extend_sdxl(model)

    model.load_state_dict(state_dict, strict=False)
    del state_dict

    # Set is_sdxl_inpaint flag.
    # Checks Unet structure to detect inpaint model. The inpaint model's
    # checkpoint state_dict does not contain the key
    # 'diffusion_model.input_blocks.0.0.weight'.
    diffusion_model_input = model.model.state_dict().get(
        'diffusion_model.input_blocks.0.0.weight'
    )
    model.is_sdxl_inpaint = (
            model.is_sdxl and
            diffusion_model_input is not None and
            diffusion_model_input.shape[1] == 9
    )
    no_half = False

    if no_half:
        pass
    else:
        vae = model.first_stage_model
        depth_model = getattr(model, 'depth_model', None)
        alphas_cumprod = model.alphas_cumprod
        model.alphas_cumprod = None
        model.half()
        model.alphas_cumprod = alphas_cumprod
        model.alphas_cumprod_original = alphas_cumprod
        model.first_stage_model = vae
        if depth_model:
            model.depth_model = depth_model

    apply_alpha_schedule_override(model)

    for module in model.modules():
        if hasattr(module, 'fp16_weight'):
            del module.fp16_weight
        if hasattr(module, 'fp16_bias'):
            del module.fp16_bias

    model.first_stage_model.to(torch.float16)

    model.sd_model_hash = sd_model_hash
    model.sd_model_checkpoint = checkpoint_info.filename
    model.sd_checkpoint_info = checkpoint_info
