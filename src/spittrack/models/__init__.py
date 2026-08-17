"""Action-recognition model implementations."""

from spittrack.models.resnet50 import ResNet50Classifier
from spittrack.models.videomae import VideoMAEClassifier

__all__ = ["ResNet50Classifier", "VideoMAEClassifier"]

