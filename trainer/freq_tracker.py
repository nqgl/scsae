import torch


class FreqTracker:
    def __init__(self, d_dict):
        self.activation_count = torch.zeros(d_dict, dtype=torch.float32, device="cuda")
        self.count = 0

    @torch.no_grad()
    def update(self, acts):
        self.activation_count += (acts > 0).float().mean(dim=0)
        self.count += 1

    @property
    def freqs(self):
        return self.activation_count / self.count

    def reset(self):
        self.activation_count = torch.zeros_like(self.activation_count)
        self.count = 0


class EMAFreqTracker:
    def __init__(self, d_dict, beta=0.999):
        self.activation_freqs = (
            torch.zeros(d_dict, dtype=torch.float32, device="cuda") + 1e-5
        )
        self.beta = beta

    @torch.no_grad()
    def update(self, acts):
        freqs = (acts > 0).float().mean(dim=0)
        self.activation_freqs.lerp_(freqs, 1 - self.beta)

    @property
    def freqs(self):
        return self.activation_freqs

    def reset(self):
        self.activation_freqs = torch.zeros_like(self.activation_freqs) + 1e-5
