import re
import torch
from omegaconf import DictConfig
import hydra
from pathlib import Path

def load_and_rename_weights(model: torch.nn.Module,
                            checkpoint_path: str,
                            rename_rules: dict,
                            verbose: bool = True) -> torch.nn.Module:
    """
    Load model weights from a checkpoint file and apply renaming rules
    to parameter names before loading them into the model.

    Args:
        model (torch.nn.Module): PyTorch model instance.
        checkpoint_path (str): Path to the .pt or .pth file containing state_dict.
        rename_rules (dict): Dictionary of renaming rules where each key is
                             a regex pattern to match and the value is the
                             replacement pattern.
        verbose (bool): If True, prints matched and renamed parameter names.

    Returns:
        model (torch.nn.Module): Model with loaded and renamed weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)

    new_state_dict = {}
    for old_name, weight in state_dict.items():
        new_name = old_name
        for pattern, repl in rename_rules.items():
            if re.search(pattern, new_name):
                renamed = re.sub(pattern, repl, new_name)
                if verbose:
                    print(f"Renaming '{new_name}' -> '{renamed}' using pattern '{pattern}' -> '{repl}'")
                new_name = renamed
        new_state_dict[new_name] = weight

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if verbose:
        if missing:
            print("Missing keys in the model:")
            for k in missing:
                print(f"  - {k}")
        if unexpected:
            print("Unexpected keys in checkpoint (not used):")
            for k in unexpected:
                print(f"  - {k}")
    return model

@hydra.main(config_path="../../configs/preprocessing", config_name="rename_weights")
def main(cfg: DictConfig):
    """
    Main entry point for the script using Hydra for configuration.

    """
    # Instantiate model
    model = hydra.utils.instantiate(cfg.model)

    checkpoint_path = Path(cfg.checkpoint_path).resolve()
    output_path = checkpoint_path.parent / "renamed_" + checkpoint_path.stem + ".pth"

    # Load and rename weights
    model = load_and_rename_weights(
        model,
        checkpoint_path=checkpoint_path,
        rename_rules=dict(cfg.rename_rules),
        verbose=cfg.verbose
    )

    # Switch to evaluation mode
    model.eval()

    # Save or return model as needed
    torch.save(model.state_dict(), output_path)
    if cfg.verbose:
        print(f"Renamed state_dict saved to {output_path}")


if __name__ == "__main__":
    main()


def load_state_dict(path, model_name="model"):
    """Load a state dict from a path.

    Args:
        path (str): The path to the state dict.
        model_name (str, optional): The name of the model. Defaults to "model".

    Returns:
        Dict[str, Any]: The state dict.
    """

    state_dict = torch.load(path, map_location="cpu")
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]
    if "model" in state_dict.keys():
        state_dict = state_dict["model"]

    # remove *.{model_name}. from the keys
    if model_name is not None:
        new_state_dict = {}
        for key in state_dict.keys():
            if model_name in key:
                new_key = re.sub(rf"^.*{model_name}\.", "", key)
                new_state_dict[new_key] = state_dict[key]

        return new_state_dict
    else:
        return state_dict