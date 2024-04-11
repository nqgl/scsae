import torch


def acts_generator(batch_size, d_data):
    for i in range(1024):
        yield torch.randn(batch_size, d_data)


num_piles = 64
torch.randint(
    0,
    num_piles,
)
