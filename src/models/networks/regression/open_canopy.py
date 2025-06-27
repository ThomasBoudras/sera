import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import pkgutil
import src.models
from src import train_utils as utils
import importlib

log = utils.get_logger(__name__)

class OpenCanopyModel(nn.Module):
    """Segmentation Network using timm models as backbone."""

    def __init__(
        self,
        ckpt_path,
        repo_model,
        model_class,

    ):
        super(OpenCanopyModel, self).__init__()
        old_src_modules = {name: mod for name, mod in sys.modules.items() if name.startswith("src.") or name == "src"}

        for module_name in list(old_src_modules.keys()):
            del sys.modules[module_name]
        sys.path.insert(0, repo_model)
         # Instancier le modèle
        from src.models.components.timmNet import timmNet  

        self.net = timmNet(**model_class)

        print("net " , list(self.net.state_dict().keys())[:10])
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        new_state_dict = {}
        for key in checkpoint["state_dict"]:
            new_key = key.replace("net.", "").replace("stages.", "stages_")   # Remplacement du préfixe
            new_state_dict[new_key] = checkpoint["state_dict"][key]
        print("ckpt ", list(new_state_dict.keys())[:10])


        keys_ckpt = new_state_dict.keys()
        keys_model = self.net.state_dict()
        load_keys = list(set(keys_ckpt) & set(keys_model))
        unload_keys = list(set(keys_model) - set(load_keys))
        print(f"List of weights of the model unloaded : {unload_keys}")
        
        self.net.load_state_dict(new_state_dict, strict=False)

        new_src_modules = {name: mod for name, mod in sys.modules.items() if name.startswith("src.") or name == "src"}
        for module_name in list(new_src_modules.keys()):
            del sys.modules[module_name]
        
        sys.path.remove(repo_model)
        for module_name in old_src_modules.keys():
            sys.modules[module_name] = importlib.import_module(module_name)

        print("Modèle chargé avec succès.")

    
    def forward(self, x, meta_data):
        return self.net(x)["out"]


