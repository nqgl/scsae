import torch.utils
import torch.utils.data
from nqgl.sc_sae.from_hsae_re.stored_acts_buffer import ActsConfig
import torch


class StoredActsDataSet(torch.utils.data.IterableDataset):
    def __init__(self, ac: ActsConfig):
        self.ac = ac

    def __iter__(self, batch_size=1024): ...
