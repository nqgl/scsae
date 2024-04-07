from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

import datasets

SAVE_DIR = Path.home() / "workspace"
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir()


@dataclass
class SplitConfig:
    splitname: str
    start: int
    end: int
    split_key: str = "train"
    documents_per_chunk: int = 2**18  # 128 * 1m // 2 = 32 mtok in the default case

    def get_splitstr(self):
        return f"{self.split_key}[0.{self.start}:0.{self.end}]"

    def get_split_key(self):
        return f"{self.split_key}[{self.start}%:{self.end}%]"


@dataclass
class DataConfig:
    set_bos: bool = True
    dataset: str = "alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2"
    model_name: str = "gpt2"
    trainsplit: Tuple[int, int] = (0, 50)
    testsplit: Tuple[int, int] = (50, 80)
    valsplit: Tuple[int, int] = (90, 100)
    seq_mul: int = 2
    seq_len: int = 128

    def __post_init__(self):
        self.validate_splits()

    def get_train_data(self): ...

    def validate_splits(self):
        return
        raise NotImplementedError

    def idstr(self):
        return f"{self.dataset.replace('/', '_')}_{self.seq_len}_{self.seq_mul}"

    def get_shuffled_chunks_dir(self):
        return SAVE_DIR / "data" / "shuffled_chunks" / self.idstr()

    def _store_chunks(self, split, splitname):
        splitstr = f"train[{split[0]}%:{split[1]}%]"

        # train_tokens = load_data(
        #     model,
        #     dataset=cfg.data_cfg.dataset,
        #     name=cfg.data_cfg.model_name,
        #     front_only=False,
        #     seq_len=128,
        #     seq_mul=cfg.data_cfg.seq_mul,
        #     set_bos=cfg.data_cfg.set_bos,
        # )  # .cuda()

        # splitname = name + split
        # splitname = splitname.replace("%", "_percent_")
        # splitname = (
        #     splitname if not select_first_n else splitname + f"_first_{select_first_n}"
        # )
        # if front_only:
        #     splitname += "_front_only"
        # if seq_len != 128:
        #     splitname += f"_seq_len_{seq_len}"
        # last_filename = (
        #     splitname + "_reshaped.pt" if set_bos else splitname + "_reshaped_no_bos.pt"
        # )

        # reshaped_name = dataset.split("/")[-1] + last_filename
        # dataset_reshaped_path = SAVE_DIR / "data" / reshaped_name
        # # if dataset exists loading_data_first_time=False
        # loading_data_first_time = not dataset_reshaped_path.exists()
