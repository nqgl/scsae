from nqgl.sae.training.setup_utils import DTYPES
from transformer_lens import HookedTransformer, utils
import einops
import torch


import time
from dataclasses import dataclass


@dataclass
class DataConfig:
    split: str
    model_name: str
    layer_num: int
    skip: bool
    set_bos: bool
    dataset: str = "alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2"
    seq_mul: int = 2


@dataclass
class BufferConfig:
    layer: int
    site: str
    flatten_heads: bool
    device: str
    d_data: int
    batch_size: int
    buffer_mult: int
    buffer_refresh_ratio: float
    buffer_dtype: str
    excl_first: bool
    buffer_autocast_dtype: str = None
    seq_len: int = 128
    act_name: str = None
    buffer_batches: int = None
    buffer_size: int = None
    model_batch_size: int = None
    buffer_batch_divisor: int = 1

    def __post_init__(self):
        self.post_init_cfg()

    def post_init_cfg(self):
        self.buffer_autocast_dtype = self.buffer_autocast_dtype or self.buffer_dtype
        self.model_batch_size = (
            self.model_batch_size
            or self.batch_size // self.seq_len * 16 // self.buffer_batch_divisor
        )
        self.buffer_size = self.batch_size * self.buffer_mult
        self.buffer_batches = self.buffer_size // self.seq_len
        self.act_name = utils.get_act_name(self.site, self.layer)
        return self


#     # dont_shuffle


class Buffer:
    """
    This defines a data buffer, to store a bunch of MLP acts that can be used to train the autoencoder.
    It'll automatically run the model to generate more when it gets halfway empty.
    """

    def __init__(self, cfg: BufferConfig, tokens, model: HookedTransformer):
        self.buffer = torch.zeros(
            (cfg.buffer_size, cfg.d_data),
            dtype=DTYPES[cfg.buffer_dtype],
            requires_grad=False,
            device=cfg.device,
        )
        self.cfg: BufferConfig = cfg
        self.token_pointer = 0
        self.first = True
        self.all_tokens = tokens
        self.model: HookedTransformer = model
        self.time_shuffling = 0
        self.dont_shuffle = False
        self.perm = (
            torch.arange(self.buffer.shape[0])
            if self.dont_shuffle
            else torch.randperm(self.buffer.shape[0])
        )
        self.prevperm = None
        self.perm_i = 0
        self.refresh()
        assert not (self.buffer == 0).all(dim=-1).any()

    def end_refresh(self):
        self.perm_i += int(self.buffer.shape[0] * self.cfg.buffer_refresh_ratio)
        if (
            self.perm_i
            + (
                int(self.buffer.shape[0] * self.cfg.buffer_refresh_ratio)
                + self.cfg.batch_size
            )
            >= self.buffer.shape[0]
        ):
            print("Resetting the perm")
            if not self.dont_shuffle:
                self.perm = torch.randperm(self.buffer.shape[0])
            self.perm_i = 0

    def getperm(self, i, size):
        # assert i + size  <= int(self.buffer.shape[0] * self.cfg.buffer_refresh_ratio), (
        #     i,
        #     size,
        #     int(self.buffer.shape[0] * self.cfg.buffer_refresh_ratio),
        #     self.perm.shape,
        #     self.first,
        # )
        return self.perm[self.perm_i + i : self.perm_i + i + size]

    def nextperm(self, size):
        perm = self.getperm(self.pointer, size)
        self.pointer += size
        return perm

    @torch.inference_mode()
    def refresh(self):
        """
        Refreshes the buffer by populating it with new activations, then shuffles it.

        Note: This method assumes that the necessary attributes and configurations are already set.
        """
        t0 = time.time()
        self.pointer = 0
        with torch.autocast("cuda"):
            if self.first:
                num_batches = self.cfg.buffer_size // (
                    self.cfg.seq_len - self.cfg.excl_first
                )
            else:
                num_batches = (
                    int(
                        self.cfg.buffer_size
                        // (self.cfg.seq_len - self.cfg.excl_first)
                        * self.cfg.buffer_refresh_ratio
                    )
                    + 1
                )

            for batch_i in range(0, num_batches, self.cfg.model_batch_size):
                tokens = self.all_tokens[
                    self.token_pointer : self.token_pointer + self.cfg.model_batch_size
                ]
                # _, cache = self.model.run_with_cache(
                #     tokens, stop_at_layer=self.cfg.layer + 1
                # )

                l = []

                def hook_fn(acts, hook):
                    l.append(acts)

                self.model.run_with_hooks(
                    tokens,
                    stop_at_layer=self.cfg.layer + 1,
                    fwd_hooks=[(self.cfg.act_name, hook_fn)],
                )
                assert len(l) == 1

                if self.cfg.flatten_heads:
                    acts = einops.rearrange(
                        l[0],
                        "batch seq_pos n_head d_head -> (batch seq_pos) (n_head d_head)",
                    )
                else:
                    if self.cfg.excl_first:
                        acts_no_re = l[0][:, 1:]
                    else:
                        acts_no_re = l[0]
                    # assert torch.all(l[0][:, 0, :] - l[0][0, 0, :] < 1e-5), (
                    #     l[0][:, 0, :] - l[0][0, 0, :]
                    # ).max()
                    acts = einops.rearrange(
                        acts_no_re,
                        "batch seq_pos d_act -> (batch seq_pos) d_act",
                    )
                assert acts.shape[-1] == self.cfg.d_data
                if not self.first:
                    self.buffer[self.nextperm(acts.shape[0])] = acts.to(
                        self.buffer.dtype
                    )
                else:
                    if batch_i == num_batches - 1:
                        self.buffer[self.pointer : self.pointer + acts.shape[0]] = acts[
                            : self.buffer.shape[0] - self.pointer
                        ].to(self.buffer.dtype)
                    else:
                        self.buffer[self.pointer : self.pointer + acts.shape[0]] = (
                            acts.to(self.buffer.dtype)
                        )
                    self.pointer += acts.shape[0]
                self.token_pointer += self.cfg.model_batch_size
            assert self.pointer + self.perm_i > int(
                self.buffer.shape[0] * self.cfg.buffer_refresh_ratio
            ), f"Pointer: {self.pointer}, buffer shape: {self.buffer.shape[0]}, buffer refresh ratio: {self.cfg.buffer_refresh_ratio}"
        self.pointer = 0
        self.end_refresh()
        self.first = False
        assert (
            self.token_pointer < self.all_tokens.shape[0]
        ), f"Ran out of tokens! token pointer: {self.token_pointer}, all tokens: {self.all_tokens.shape[0]}"
        self.time_shuffling += time.time() - t0
        # torch.cuda.empty_cache()

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.nextperm(self.cfg.batch_size)]
        if (
            self.pointer
            > int(self.buffer.shape[0] * self.cfg.buffer_refresh_ratio)
            - self.cfg.batch_size
        ):
            print("Refreshing the buffer!")
            self.refresh()

        return out

    @torch.no_grad()
    def next_tokens(self):
        out = self.buffer[self.pointer : self.pointer + self.cfg.batch_size]
        return out

    @torch.no_grad()
    def freshen_buffer(self, fresh_factor=1, half_first=True):
        """
        Refreshes the buffer by moving the pointer and calling the refresh method.
        Warning: this burns data

        Args:
            fresh_factor (int): The factor by which the buffer should be refreshed.
            half_first (bool): Whether to refresh half of the buffer first.

        Returns:
            None
        """
        if half_first:
            n = (0.5 * self.cfg.buffer_size) // self.cfg.batch_size
            self.pointer += n * self.cfg.batch_size
            self.refresh()
        n = (
            (self.cfg.buffer_refresh_ratio) * self.cfg.buffer_size
        ) // self.cfg.batch_size
        for _ in range(1 + int(fresh_factor / (self.cfg.buffer_refresh_ratio))):
            self.pointer += (n + 1) * self.cfg.batch_size
            self.refresh()

    @torch.no_grad()
    def skip_first_tokens_ratio(self, skip_percent):
        """
        Fast-forwards through skip_percent proportion of the data
        """
        self.token_pointer += int(self.all_tokens.shape[0] * skip_percent)
        self.first = True
        self.refresh()
