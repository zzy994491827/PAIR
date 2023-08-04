#-*-coding:utf-8 -*-
# --------------------------------------------------------
# Pytorch THH

# --------------------------------------------------------


import os
import logging
import torch

ROOT_PATH = os.path.join(os.environ['HOME'], 'VisualSearch')
MIN_WORD_COUNT = 5

TEXT_ENCODINGS = ['bow', 'bow_nsw', 'gru']
DEFAULT_TEXT_ENCODING = 'bow'
DEFAULT_LANG = 'en'

logger = logging.getLogger(__file__) 
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)

torch.multiprocessing.set_sharing_strategy('file_system') 

class No:
    pass
