from dataclasses import dataclass, field
from typing import Tuple
from pathlib import Path

import datasets

SAVE_DIR = Path.home() / "workspace"
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir()
import einops


@dataclass
class SplitConfig:
    splitname: str
    start: int
    end: int
    split_key: str = "train"
    documents_per_chunk: int = 2**18  # 128 * 1m // 2 = 32 mtok in the default case
    tokens_from_split: int = None

    def get_splitstr(self):
        return f"{self.split_key}[0.{self.start}:0.{self.end}]"

    def get_split_key(self):
        return f"{self.split_key}[{self.start}%:{self.end}%]"


@dataclass
class DataConfig:
    dataset: str = "alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2"
    model_name: str = "gpt2"
    trainsplit: SplitConfig = field(
        default_factory=lambda: SplitConfig(
            splitname="train",
            start=0,
            end=50,
            tokens_from_split=450_000_000,
        )
    )
    testsplit: SplitConfig = field(
        default_factory=lambda: SplitConfig(
            splitname="test",
            start=80,
            end=90,
        )
    )
    valsplit: SplitConfig = field(
        default_factory=lambda: SplitConfig(
            splitname="val",
            start=90,
            end=100,
        )
    )

    set_bos: bool = True
    seq_mul: int = 2
    seq_len: int = 128

    def __post_init__(self):
        self.validate_splits()

    def get_train_data(self): ...

    def validate_splits(self):
        return
        raise NotImplementedError

    def idstr(self):
        return f"{self.dataset.replace('/', '_')}_{self.seq_len}_{self.seq_mul}_{self.set_bos}_{self.model_name}"

    def get_shuffled_chunks_dir(self):
        return SAVE_DIR / "data" / "shuffled_chunks" / self.idstr()

    def _get_split_path(self, split: SplitConfig):
        return self.get_shuffled_chunks_dir() / split.get_splitstr()

    def _get_tokens_split_path(self, split: SplitConfig):
        return self._get_split_path(split) / "tokens"

    def _get_acts_split_path(self, split: SplitConfig):
        return self._get_split_path(split) / "acts"

    def _store_split_token_chunks(
        self, split: SplitConfig, splitname, tokens_per_chunk
    ):
        splitstr = f"train[{split[0]}%:{split[1]}%]"
        tok_dir = self._get_tokens_split_path(split)
        tok_dir.mkdir(exist_ok=False, parents=True)
        train_tokens = datasets.load_dataset(
            dataset=self.dataset,
            split=split.get_split_key(),
            cache_dir=SAVE_DIR / "cache",
        )
        all_tokens_reshaped = einops.rearrange(
            all_tokens,
            "batch (x seq_len) -> (batch x) seq_len",
            x=seq_mul,
            seq_len=seq_len,
        )

        # loading_data_first_time = not dataset_reshaped_path.exists()
