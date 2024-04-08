import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from nqgl.sc_sae.models.configs import SAEConfig
from unpythonic import box
from jaxtyping import Float
import json
from pathlib import Path
from dataclasses import asdict
from nqgl.mlutils.components.normalizer.anth_l2 import L2Normalizer, l2normalized

SAVE_DIR = Path.home() / "workspace"
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir()


class BaseSAE(nn.Module):
    MODEL_TYPE = None

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        self.W_enc: Float[Tensor, "d_data d_dict"] = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(
                    cfg.d_data,
                    cfg.d_dict,
                )
            )
        )
        self.W_dec: Float[Tensor, "d_dict d_data"] = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(
                    cfg.d_dict,
                    cfg.d_data,
                )
            )
        )
        if cfg.tied_init:
            self.W_dec.data[:] = self.W_enc.data.t()
        self.b_enc: Float[Tensor, "d_dict"] = nn.Parameter(torch.zeros(cfg.d_dict))
        self.b_dec: Float[Tensor, "d_data"] = (
            nn.Parameter(torch.zeros(cfg.d_data)) if self.cfg.use_b_dec else 0
        )
        self.norm_dec()
        assert self.MODEL_TYPE == self.cfg.sae_type

    @l2normalized
    def forward(
        self,
        x: Float[Tensor, "batch d_data"],
        acts_box: box = None,
        spoofed_acts_box: box = None,
    ):
        acts_box = acts_box or box()
        spoofed_acts_box = spoofed_acts_box or box()
        acts = self.encode(x, spoofed_acts_box=spoofed_acts_box)
        acts_box << acts
        if spoofed_acts_box.x is None:
            spoofed_acts_box << acts
        return self.decode(acts)

    def encode(self, x, spoofed_acts_box):
        raise NotImplementedError

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def post_backward(self):
        self.orthogonalize_dec_grads()

    def post_step(self):
        self.norm_dec()

    @torch.no_grad()
    def norm_dec(self):
        norm = self.W_dec.norm(dim=-1, keepdim=True)
        normed = self.W_dec / norm
        self.W_dec[:] = (
            torch.where(norm > 1, normed, self.W_dec)
            if self.cfg.selectively_norm_dec
            else normed
        )

    @torch.no_grad()
    def orthogonalize_dec_grads(self):
        grad = self.W_dec.grad
        dec_normed = self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True)
        grad_orth = grad - (dec_normed * grad).sum(dim=-1, keepdim=True) * dec_normed
        self.W_dec.grad[:] = grad_orth

    @classmethod
    def get_next_version(cls, name=None, save_dir=None):
        save_dir = SAVE_DIR if save_dir is None else Path(save_dir)

        search = cls.MODEL_TYPE + f"_{name}" if name is not None else cls.sae_type
        import glob

        type_files = glob.glob(str(save_dir) + (f"/*_{search}*_cfg.json"))
        version_list = [int(file.split("/")[-1].split("_")[0]) for file in type_files]
        if len(version_list):
            return 1 + max(version_list)
        else:
            return 0

    def save(self, name=""):
        version = self.__class__.get_next_version()
        vname = str(version) + "_" + self.__class__.MODEL_TYPE + "_" + name
        torch.save(self.state_dict(), SAVE_DIR / (vname + ".pt"))
        with open(SAVE_DIR / (str(vname) + "_cfg.json"), "w") as f:
            json.dump(asdict(self.cfg), f)

    @classmethod
    def load(cls, version="*", name=None, cfg=None, save_dir=None, omit_type=False):
        save_dir = SAVE_DIR if save_dir is None else Path(save_dir)
        # get correct name with globbing
        import glob

        vname = str(version) + "_" + cls.MODEL_TYPE if not omit_type else str(version)
        vname = "*" + vname + "*" + name if name is not None else vname
        if cfg is None:
            cfg_search = str(save_dir) + f"/{vname}*_cfg.json"
            print("seeking", cfg_search)
            cfg_name = glob.glob(cfg_search)
            cfg = json.load(open(cfg_name[0]))

            cfg = cls.CONFIG(**cfg)
        print(vname)
        pt_name = glob.glob(str(save_dir / (str(vname) + "*.pt")))
        self = cls(cfg=cfg)
        self.load_state_dict(torch.load(pt_name[0]))
        return self
