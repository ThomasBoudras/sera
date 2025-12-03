import geefetch
import geefetch.utils
import geefetch.utils.gee
import hydra
from omegaconf import DictConfig
from retry import retry
from src.preprocessing.download.download_s1_s2_utils import download_s1_s2
from pathlib import Path
from omegaconf import OmegaConf




@hydra.main(version_base=None, config_path="../../../configs/preprocessing/download", config_name="dwd_bbox_chantilly_timeseries")
@retry(exceptions=Exception, delay=10, tries=100)
def main(cfg: DictConfig) -> None:
    geefetch.utils.gee.auth(cfg.gee_project)
    print(OmegaConf.to_yaml(cfg))
    for date in cfg.dates:
        date = date.replace("/","")
        download_s1_s2(
            data_dir=Path(cfg.data_dir).resolve(),
            resolution=cfg.resolution,
            tile_shape=cfg.tile_shape,
            max_tile_size=cfg.max_tile_size,
            cloudless_portion=cfg.cloudless_portion,
            cloud_prb=cfg.cloud_prb,
            country=cfg.country,
            composite_method_s1=cfg.composite_method_s1,
            composite_method_s2=cfg.composite_method_s2,
            bounds=(cfg.bbox["left"], cfg.bbox["bottom"], cfg.bbox["right"], cfg.bbox["top"]),
            reference_date=date,
            duration=cfg.duration,
            s1_orbit=cfg.s1_orbit
        )


if __name__ == "__main__":
    main()

