from logging import getLogger
from pathlib import Path

from omegaconf import OmegaConf

logger = getLogger(__name__)


def gitdir():
    for p in (Path.cwd(), *Path.cwd().parents):
        if (p / ".git").exists():
            return str(p)
    logger.warning("No git directory found.")
    return str(Path.cwd())


OmegaConf.register_new_resolver("gitdir", gitdir)
