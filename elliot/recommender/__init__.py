"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from .base_recommender_model import BaseRecommenderModel
from .generic import ProxyRecommender
from .knn import ItemKNN, UserKNN, AttributeItemKNN, AttributeUserKNN
from .graph_based import RP3beta
from .autoencoers import EASER
from .unpersonalized import MostPop, Random

