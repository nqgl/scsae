# thanks to https://discuss.pytorch.org/t/torch-save-like-open-mode-a/137114
# for code snippets and setting me on the right path
import tables
from pathlib import Path
import torch
from typing import Union, List


path = Path("data/table_test.h5")
t = torch.arange(32).reshape(2, 16)

dtype_to_atom = {
    torch.float32: tables.Float32Atom(),
    torch.int32: tables.Int32Atom(),
    torch.int64: tables.Int64Atom(),
    torch.float16: tables.Float16Atom(),
}


class AppendDiskTensor:
    def __init__(
        self,
        path: Union[str, Path],
        dtype: torch.dtype,
        fixed_shape: List[int],
    ):
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        self.dtype = dtype
        self.fixed_shape = fixed_shape

    def init_file(self, force=False):
        assert force or not self.path.exists()
        self.path = tables.open_file(str(path), mode="w")
        self.path.create_earray(
            self.path.root,
            "batches",
            atom=dtype_to_atom[self.dtype],
            shape=(0, *self.fixed_shape),
        )
        self.path.close()

    def write(self, t: torch.Tensor):
        if not path.exists():
            self.init_file()
        assert t.dtype == self.dtype
        assert t.shape[1:] == self.fixed_shape
        self.path = tables.open_file(str(path), mode="a")

        self.path.root.batches.append(t.cpu().numpy())
        self.path.close()

    def read(self):
        self.path = tables.open_file(str(path), mode="r")
        t = torch.tensor(self.path.root.batches[:])
        self.path.close()
        return t

    def shuffle(self):
        t = self.read()
        t = t[torch.randperm(t.shape[0])]
        self.init_file(force=True)
        self.write(t)


class Piler:
    def __init__(self, path: Union[str, Path], dtype, fixed_shape, num_piles=None):
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        if num_piles is None:
            g = path.glob("pile*")
            num_piles = len(list(g))
        else:
            path.mkdir(parents=True)
        self.dtype = dtype
        self.fixed_shape = fixed_shape
        self.num_piles = num_piles
        self.piles = [
            AppendDiskTensor(path / f"pile_{i}.h5", dtype, fixed_shape)
            for i in range(num_piles)
        ]

    def randpile(self, t: torch.Tensor):
        i = torch.randint(0, self.num_piles, [t.shape[0]])
        for pile in range(self.num_piles):
            self.piles[pile].write(t[i == pile])

    def shuffle_piles(self):
        for pile in self.piles:
            pile.shuffle()


t_app = AppendDiskTensor(
    path,
    torch.int64,
    [16],
)

l = []
for i in range(4):
    t = torch.arange(32).reshape(2, 16)
    t_app.write(t)
    l.append(t)
