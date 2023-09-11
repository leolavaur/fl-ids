import os
from logging import getLogger
from pathlib import Path

from omegaconf import OmegaConf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


log = getLogger(__name__)


def gitdir():
    """OmegaConf resolver to get the git directory."""
    for p in (Path.cwd(), *Path.cwd().parents):
        if (p / ".git").exists():
            return str(p)
    log.warning("No git directory found.")
    return str(Path.cwd())


OmegaConf.register_new_resolver("gitdir", gitdir)
