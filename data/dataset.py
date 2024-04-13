from dataclasses import dataclass, field
from typing import Tuple
from transformer_lens import HookedTransformer, utils
from nqgl.sc_sae.data.tabletensor import Piler
import datasets
import torch
from unpythonic import box


from data.locations import DATA_DIRS

import einops
import tqdm


@dataclass
class SplitConfig:
    splitname: str
    start: int
    end: int
    split_key: str = "train"
    documents_per_chunk: int = 2**18  # 128 * 1m // 2 = 32 mtok in the default case
    tokens_from_split: int = None
    tokens_per_pile: int = 2**24
    acts_per_pile: int = 2**18
    approx_tokens_per_percent: int = 30_000_000 * 256 // 100

    def __post_init__(self):
        if (
            self.tokens_from_split is not None
            and self.tokens_from_split > self._approx_num_tokens_in_dataset_split
        ):
            raise ValueError(
                f"tokens_from_split {self.tokens_from_split} is greater than approx_num_tokens_in_split {self._approx_num_tokens_in_dataset_split}"
            )

    @property
    def split_dir_id(self):
        return f"{self.split_key}[{self.start}_p:{self.end}_p]"

    def get_split_key(self):
        return f"{self.split_key}[{self.start}%:{self.end}%]"

    @property
    def _approx_num_tokens_in_dataset_split(self):
        return self.approx_tokens_per_percent * (self.end - self.start)

    @property
    def approx_num_tokens(self):
        return self.tokens_from_split or self._approx_num_tokens_in_dataset_split


@dataclass
class ActsDataConfig:
    d_data: int = 768
    excl_first: bool = False
    layer_num: int = 6
    site: str = "resid_pre"

    @property
    def hook_site(self):
        return utils.get_act_name(self.site, self.layer_num)


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

    acts_config: ActsDataConfig = field(default_factory=ActsDataConfig)

    def __post_init__(self):
        self.validate_splits()

    def get_train_data(self): ...

    def validate_splits(self):  # TODO
        return
        raise NotImplementedError

    def idstr(self):
        return f"{self.dataset.replace('/', '_')}_{self.seq_len}_{self.seq_mul}_{self.set_bos}_{self.model_name}"

    def _get_tokens_split_path(self, split: SplitConfig):
        return DATA_DIRS._CHUNKS_DIR / self.idstr() / split.split_dir_id / "tokens"

    def _get_acts_split_path(self, split: SplitConfig):
        return DATA_DIRS._CHUNKS_DIR / self.idstr() / split.split_dir_id / "acts"

    def _tokens_piles_path(self, split: SplitConfig):
        return self._get_tokens_split_path(split) / "piles"

    def _acts_piles_path(self, split: SplitConfig):
        return self._get_acts_split_path(split) / "piles"

    def tokens_piler(self, split: SplitConfig, write=False) -> Piler:
        return Piler(
            self._tokens_piles_path(split),
            dtype=torch.int64,
            fixed_shape=[self.seq_len],
            num_piles=(
                split._approx_num_tokens_in_dataset_split // split.tokens_per_pile
                if write
                else None
            ),
        )

    def acts_piler(
        self, split: SplitConfig, write=False, target_gb_per_pile=2
    ) -> Piler:
        bytes_per_pile = target_gb_per_pile * 2**30
        dtype_bytes = 2  # hardcoded assumption of float16
        b_per_act = self.acts_config.d_data * dtype_bytes
        total_b = split.approx_num_tokens * b_per_act
        num_piles = (total_b + bytes_per_pile - 1) // bytes_per_pile
        return Piler(
            self._acts_piles_path(split),
            dtype=torch.float16,
            fixed_shape=[self.acts_config.d_data],
            num_piles=(num_piles if write else None),
        )

        # loading_data_first_time = not dataset_reshaped_path.exists()

    def train_data_batch_generator(self, model, batch_size):
        return ActsData(self, model).acts_generator(
            self.trainsplit, batch_size=batch_size
        )


class TokensData:
    def __init__(self, cfg: DataConfig, model: HookedTransformer):
        self.cfg = cfg
        self.model = model

    def _store_split(self, split: SplitConfig):
        tqdm.tqdm.write(f"Storing tokens for {split.splitname}")
        dataset = datasets.load_dataset(
            self.cfg.dataset,
            split=split.get_split_key(),
            cache_dir=DATA_DIRS.CACHE_DIR,
        )
        dataset.set_format(type="torch", columns=["input_ids"])

        all_documents = einops.rearrange(
            dataset["input_ids"],
            "batch (x seq_len) -> (batch x) seq_len",
            x=self.cfg.seq_mul,
            seq_len=self.cfg.seq_len,
        )
        if self.cfg.set_bos:
            all_documents[:, 0] = self.model.tokenizer.bos_token_id
        num_tok = all_documents.numel()
        piler = self.cfg.tokens_piler(split, write=True)
        tqdm.tqdm.write("Distributing tokens to piles")
        doc_dist_batch_size = all_documents.shape[0] // 100
        for i in tqdm.tqdm(
            range(
                0,
                all_documents.shape[0] // doc_dist_batch_size * doc_dist_batch_size,
                doc_dist_batch_size,
            )
        ):
            piler.distribute(all_documents[i : i + doc_dist_batch_size])
        piler.shuffle_piles()

    def get_tokens_from_split(self, split: SplitConfig, num_tokens=None):
        if not self.cfg._tokens_piles_path(split).exists():
            self._store_split(split)
        piler = self.cfg.tokens_piler(split)

        num_piles = (
            piler.num_piles
            if num_tokens is None
            else (num_tokens + split.tokens_per_pile - 1) // split.tokens_per_pile
        )
        assert (
            num_piles <= piler.num_piles
        ), f"{num_tokens}, {split.tokens_per_pile}, {piler.num_piles}"
        tokens = piler[0:num_piles]
        assert (
            num_tokens is None
            or abs(tokens.numel() - num_tokens) < split.tokens_per_pile
        ), f"{tokens.shape} from piler vs {num_tokens} requested\
                this is expected if tokens per split is small, otherwise a bug.\
                    \n piles requested: {num_piles}, available: {piler.num_piles}"
        return tokens


class ActsData:
    def __init__(self, cfg: DataConfig, model: HookedTransformer):
        self.cfg = cfg
        self.model = model

    def _store_split(self, split: SplitConfig):
        tokens = TokensData(self.cfg, self.model).get_tokens_from_split(
            split, num_tokens=split.tokens_from_split
        )
        acts_piler = self.cfg.acts_piler(split, write=True)

        tqdm.tqdm.write(f"Storing acts for {split.splitname}")

        def tokens_generator():
            meta_batch_size = 2048 * 4
            for i in tqdm.tqdm(
                range(
                    0, len(tokens) // meta_batch_size * meta_batch_size, meta_batch_size
                )
            ):
                yield tokens[i : i + meta_batch_size]

        for acts in self.acts_generator_from_tokens_generator(tokens_generator()):
            acts_piler.distribute(acts)
        acts_piler.shuffle_piles()

    def acts_generator_from_tokens_generator(self, tokens_generator):
        for tokens in tokens_generator:
            acts = self.to_acts(tokens)
            yield acts

    def to_acts(self, tokens):
        acts_list = []
        llm_batch_size = 1024 * 2
        assert tokens.shape[0] % llm_batch_size == 0
        with torch.autocast(device_type="cuda"):
            with torch.inference_mode():

                def hook_fn(acts, hook):
                    acts_list.append(acts)

                for i in range(
                    0,
                    tokens.shape[0] // llm_batch_size * llm_batch_size,
                    llm_batch_size,
                ):
                    self.model.run_with_hooks(
                        tokens[i : i + llm_batch_size],
                        stop_at_layer=self.cfg.acts_config.layer_num + 1,
                        fwd_hooks=[(self.cfg.acts_config.hook_site, hook_fn)],
                    )
        acts = torch.cat(acts_list, dim=0)
        if self.cfg.acts_config.excl_first:
            acts = acts[:, 1:]
        acts = einops.rearrange(
            acts,
            "batch seq d_data -> (batch seq) d_data",
        )
        return acts.half()

    def acts_generator(self, split: SplitConfig, batch_size):
        if not self.cfg._acts_piles_path(split).exists():
            self._store_split(split)
        piler = self.cfg.acts_piler(split)
        progress = tqdm.trange(split.approx_num_tokens // batch_size)
        for p in range(piler.num_piles):
            pile = piler[p]
            for i in range(0, len(pile) // batch_size * batch_size, batch_size):
                yield pile[i : i + batch_size].cuda()
                progress.update()
