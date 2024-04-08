from nqgl.sae.training.setup_utils import load_data, get_model
from nqgl.sae.hsae.hsae import HierarchicalAutoEncoderConfig, HierarchicalAutoEncoder
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import torch
import pathlib
import os

fp32 = True
DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

DEVICE = os.environ.get("DEVICE", "cuda")


@dataclass
class ActsConfig:
    start_percent: Optional[int]
    end_percent: Optional[int]
    dataset: str = "alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2"
    model_name: str = "gpt2-small"
    layer_num: int = 6
    site_name: str = "resid_pre"
    dtype: str = "fp32"
    storage_dtype: str = "fp16"
    front_only: bool = False
    buffer_refresh_ratio: float = 0.5
    d_data: int = 768
    max_chunk_size_mb: int = 512
    exclude_first_acts: bool = True
    set_bos: bool = True

    def chunk_name(self, chunk_num: int):
        return self.folder_name() + f"/chunk{chunk_num}.pt"

    def folder_name(self):
        return (
            f"saved_acts/{self.dataset.replace('/', '_')}/{self.model_name}"
            + f"/layer{self.layer_num}/{self.site_name}"
            + f"/{self.start_percent}-{self.end_percent}/{self.dtype}-{self.storage_dtype}"
            + f"excl_bos_{self.exclude_first_acts}"
            + "set_bos"
            if self.set_bos
            else ""
        )

    def path(self, chunk_num: Optional[int] = None):
        p = Path.home() / "workspace" / self.folder_name()
        if chunk_num is not None:
            p /= f"chunk{chunk_num}.pt"
        return p

    def cfg_path(self):
        return self.path() / "config.json"

    def save_chunk(self, chunk, chunk_num: int):
        folder = self.path()
        folder.mkdir(parents=True, exist_ok=True)
        if self.cfg_path().exists():
            assert self.cfg_path().read_text() == str(self)
        else:
            self.cfg_path().write_text(str(self))
        path = self.path(chunk_num)
        assert not path.exists()
        torch.save(chunk, path)

    def read_chunk(self, chunk, cast=None, read_device=DEVICE) -> torch.Tensor:
        assert self.cfg_path().exists()
        assert self.cfg_path().read_text() == str(self)
        tensor = torch.load(self.path(chunk), map_location=read_device)
        assert tensor.dtype == DTYPES[self.storage_dtype], (
            tensor.dtype,
            self.storage_dtype,
        )
        if cast is not None:
            return tensor.to(cast)
        return tensor

    def get_tensor_len(self):
        bytes_per_element = 4 if self.storage_dtype == "fp32" else 2
        bytes_per_act = self.d_data * bytes_per_element
        max_chunk_bytes = 1024**2 * self.max_chunk_size_mb
        return max_chunk_bytes // bytes_per_act

    def get_tensor_to_fill(self, batch_size: int, device=DEVICE):
        tlen = self.get_tensor_len() // batch_size * batch_size
        return torch.zeros(
            tlen, self.d_data, dtype=DTYPES[self.storage_dtype], device=device
        )

    def num_chunks(self):
        import os
        import glob

        l = glob.glob(str(self.path() / "*.pt"))
        return len(l)

    def read_as_iter(self, batch_size, stop_after=None, cast=None):
        chunk_num = 0
        next_chunk = self.read_chunk(chunk_num)
        tqdm_iter = tqdm.trange(
            (stop_after or self.num_chunks()) * (len(next_chunk) // batch_size)
        )
        while True:
            for i in range(0, len(next_chunk), batch_size):
                tqdm_iter.update(1)
                yield next_chunk[i : i + batch_size]
            # del next_chunk
            chunk_num += 1
            if stop_after is not None and chunk_num >= stop_after:
                break
            try:
                print("loading chunk", chunk_num)
                next_chunk = self.read_chunk(chunk_num, cast=cast)
                print("loaded chunk", chunk_num, next_chunk.shape)
            except FileNotFoundError:
                break

    @torch.no_grad()
    def read_as_iter_no_bos(self, batch_size):
        chunk_num = 0
        bos_example = None
        next_chunk = self.read_chunk(chunk_num)
        tqdm_iter = tqdm.trange(5000 * len(next_chunk) // batch_size)
        while True:
            if bos_example is None:
                norms = next_chunk.norm(dim=-1)
                i = norms.argmax()
                mm = norms[i]
                if 3100 > mm > 3000:
                    bos_example = next_chunk[i]
                    assert bos_example.ndim == 1
            cheq_okay = (
                next_chunk[:, :200] - bos_example[:200].unsqueeze(0) > 1e-5
            ).any(dim=-1)
            next_chunk = next_chunk[cheq_okay]
            for i in range(0, len(next_chunk), batch_size):
                tqdm_iter.update(1)
                if i + batch_size > len(next_chunk):
                    break
                yield next_chunk[i : i + batch_size]
            chunk_num += 1
            try:
                # print("loading chunk", chunk_num)
                next_chunk = self.read_chunk(chunk_num)
                # print("loaded chunk", chunk_num, next_chunk.shape)
            except FileNotFoundError:
                break


import tqdm

fp32 = True
device = os.environ.get("DEVICE", "cuda")
DATA_DTYPE = "fp16"
print(
    f"""
fp32:{fp32}
device:{device}
DATA_DTYPE:{DATA_DTYPE}
"""
)


def store_acts(ac: ActsConfig, batch_size=1024, buffer_mult=2048):
    hcfg = HierarchicalAutoEncoderConfig(
        site=ac.site_name,
        d_data=ac.d_data,
        model_name=ac.model_name,
        layer=ac.layer_num,
        # gram_shmidt_trail=512,
        batch_size=batch_size,
        buffer_mult=buffer_mult,
        buffer_refresh_ratio=ac.buffer_refresh_ratio,
        flatten_heads=False,
        buffer_dtype=ac.dtype,
        enc_dtype=ac.dtype,
        device=DEVICE,
        # buffer_batch_divisor=4,
    )
    from nqgl.sc_sae.from_hsae_re.buffer2_no_cast import Buffer, BufferConfig

    buf_cfg = BufferConfig(
        layer=hcfg.layer,
        site=hcfg.site,
        flatten_heads=hcfg.flatten_heads,
        device=hcfg.device,
        d_data=hcfg.d_data,
        batch_size=hcfg.batch_size,
        buffer_mult=hcfg.buffer_mult,
        buffer_refresh_ratio=hcfg.buffer_refresh_ratio,
        buffer_dtype=hcfg.buffer_dtype,
        excl_first=ac.exclude_first_acts,
        buffer_autocast_dtype=DATA_DTYPE,
    )
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(ac.model_name).to(DEVICE)
    all_tokens = load_data(
        model,
        dataset=ac.dataset,
        # dataset="stas/openwebtext-10k",
        name=hcfg.model_name,
        split=f"train[{ac.start_percent}%:{ac.end_percent}%]",
        front_only=False,
        seq_mul=2,
        set_bos=ac.set_bos,
    )  # .cuda()
    # if ac.exclude_first_acts:
    #     from data.buffer2 import Buffer
    # else:
    #     from nqgl.sae.training.buffer import Buffer
    buffer = Buffer(buf_cfg, all_tokens, model)
    buffer.freshen_buffer(4, True)
    # chunk_num = 0
    num_chunks = all_tokens.numel() // ac.get_tensor_len()
    overflow_lost = all_tokens.numel() % ac.get_tensor_len()
    print(f"num_chunks={num_chunks}, overflow_lost={overflow_lost}")
    print(f"\n Will use {num_chunks * ac.max_chunk_size_mb // 1024}GB of disk space.")
    print("chunk_len", ac.get_tensor_len())
    for chunk_num in tqdm.trange(num_chunks + 1):
        chunk = ac.get_tensor_to_fill(batch_size)
        for i in range(0, len(chunk), batch_size):
            # print(i, len(chunk))
            b = buffer.next()
            chunk[i : i + batch_size] = b.to(DTYPES[ac.storage_dtype])

        ac.save_chunk(chunk, chunk_num)
        chunk_num += 1
        # del chunk
