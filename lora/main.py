import networks
import sd_models

ckpt_sd = sd_models.load_model("C:/c_git_project/sd-webui-aki/models/Stable-diffusion/noobaiXLNAIXL_epsilonPred05Version.safetensors")
networks.assign_network_names_to_compvis_modules(ckpt_sd)
print(ckpt_sd.keys())

