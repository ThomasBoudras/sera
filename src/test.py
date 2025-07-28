import hydra
from omegaconf import DictConfig
from lightning import seed_everything
from src import global_utils as utils

# Get logger    
log = utils.get_logger(__name__)

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def test(config: DictConfig) :

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule.instance._target_}>")
    datamodule = hydra.utils.instantiate(config.datamodule.instance)

    # Init lightning module
    log.info(f"Instantiating module <{config.module.instance._target_}>")
    module = hydra.utils.instantiate(config.module.instance)

    # Init lightning callbacks
    callbacks = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger)
        
    # Test the model
    if config.get("ckpt_path"):
        ckpt_path = config.ckpt_path
        log.info(f"Starting testing with {ckpt_path}!")
        trainer.test(model=module, datamodule=datamodule, ckpt_path=ckpt_path)
    else :
        raise Exception("Give a checkpoint to test")
    
    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        module=module,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]

    return None
