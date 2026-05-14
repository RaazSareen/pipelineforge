"""
PipelineForge
A modular, production-grade Data Science engineering framework.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pipelineforge")
except PackageNotFoundError:
    __version__ = "unknown"
