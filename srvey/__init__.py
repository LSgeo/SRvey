""" SRvey 

Here I collect functions such as the Session handler, and set "run-time" 
options like seeds, deterministic mode, etc.
"""

from pathlib import Path
from datetime import datetime
import logging

from comet_ml import Experiment
import numpy as np
import torch
import time

import srvey.cfg as cfg

# Config


class Session:
    """Create folder structure, create comet experiment"""

    def __init__(self, debug=False):
        self.t0 = time.perf_counter()
        np.seterr(all="raise")
        torch.manual_seed(cfg.manual_seed)
        torch.backends.cudnn.benchmark = not cfg.reproducibile_mode
        # torch.use_deterministic_algorithms(cfg.reproducibile_mode)
        self.device = torch.device("cuda")

        self.experiment = Experiment(disabled=debug)

        root = Path()
        self.session_id = (
            "debug" if debug else "Session_" + datetime.now().strftime("%y%m%d-%H%M")
        )
        self.session_dir = (
            root
            / "experiments"
            / (
                "debug"
                if debug
                else f"{cfg.pretrained_model_id or self.experiment.get_key()}"
            )
        )
        if cfg.pretrained_model_id:
            self.session_dir = (
                self.session_dir / f"continued_in_{self.experiment.get_key()}"
            )

        self.model_out_path = self.session_dir / "models"
        self.model_out_path.mkdir(exist_ok=True, parents=True)

        self.__init_comet()
        self.__init_log()
        d = torch.cuda.get_device_properties(torch.cuda.device)
        self.log_train.info(
            f"| GPU: {d.name} "
            f"| GPU RAM: {d.total_memory / 1024**3:.1f} GB "
            f"| SM's: {d.multi_processor_count} |"
        )

    def __init_comet(self):
        self.experiment.add_tags(cfg.tags)
        self.experiment.log_parameters(
            {
                "Batch size train": cfg.trn_batch_size,
                "Batch size validation": cfg.val_batch_size,
                "Max LR": cfg.max_lr,
                "LR scheduler": cfg.scheduler_spec["name"],
                "Number of epochs": cfg.num_epochs,
                "Preview indices": cfg.preview_indices,
                "AMP enabled": cfg.use_amp,
                "Reproducible mode": cfg.reproducibile_mode,
                "Seed": cfg.manual_seed,
                "Validation frequency": cfg.val_freq,
                "Preview frequency": cfg.preview_freq,
                "Checkpoint frequency": cfg.checkpoint_freq,
                "Number of DataLoader workers": cfg.num_workers,
                # "Load discriminator weights": cfg.load_d_weights,
                # "Iterations per epoch": iters_per_epoch,
            }
        )
        self.experiment.log_parameters(cfg.dataset_config, prefix="Noddy")
        self.experiment.log_parameters(cfg.encoder_spec, prefix="SwinIR")
        self.experiment.log_parameters(cfg.imnet_spec, prefix="MLP")
        self.experiment.log_parameters(cfg.lte_spec, prefix="LTE")

    def __init_log(self):
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
            datefmt="%m-%d %H:%M",
            filename=f"{self.session_dir / 'session.log'}",
            filemode="w",
        )

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s %(name)-6s: %(levelname)-8s %(message)s"
        )
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)

        self.log = logging.getLogger("Session")
        self.log_train = logging.getLogger("Train")
        self.log.info("Initialised logging")

    def begin_epoch(self, epoch):
        """Hook for beginning an epoch"""
        self.log.info(f"Beginning epoch {epoch}")
        self.experiment.set_epoch(epoch)
        self.epoch = epoch

    def end(self):
        self.experiment.end()
        self.log.info(f"Finished in {time.perf_counter() - self.t0} seconds.")
