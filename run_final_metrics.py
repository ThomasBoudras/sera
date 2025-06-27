import hydra
from omegaconf import DictConfig

from omegaconf import OmegaConf
import hydra
from omegaconf import DictConfig

def from_float_to_str(number: int) -> str:
    number_str = str(number)
    formatted_str = number_str.replace('.', '-')
    return formatted_str

def mul(factor, number) :
    return factor * number

# Register the resolver
OmegaConf.register_new_resolver("from_float_to_str", from_float_to_str)
OmegaConf.register_new_resolver("mul", mul)


@hydra.main(version_base="1.3", config_path="configs/final_metrics", config_name="change_detections_threshold_test.yaml")
def main(config: DictConfig):

    from src.results.computes_metrics import compute_metrics
    from src.utils.utils import print_config
    print_config(config, fields={"get_images", "get_metrics", "get_plots"})
    return compute_metrics(config)


if __name__ == "__main__":
    main()
