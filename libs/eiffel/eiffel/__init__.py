import os
from logging import getLogger
from pathlib import Path

from flwr.common.logger import logger as flwr_logger
from omegaconf import OmegaConf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


log = getLogger(__name__)
flwr_logger.removeHandler(flwr_logger.handlers[0])


def get_git_root():
    """OmegaConf resolver to get the git directory."""
    for p in (Path.cwd(), *Path.cwd().parents):
        if (p / ".git").exists():
            return str(p)
    log.warning("No git directory found.")
    return str(Path.cwd())


OmegaConf.register_new_resolver("gitdir", get_git_root)
OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)
