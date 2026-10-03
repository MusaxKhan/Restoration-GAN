"""Hard-routed restoration (Task 2): classifier -> specialist autoencoder, identity for clean."""
from __future__ import annotations

import torch
import torch.nn as nn


class HardRouter(nn.Module):
    """x_hat = x                      if r = clean
               A_salt/blur/occ(x)     otherwise, where r = argmax C(x) (or a supplied oracle label)."""

    def __init__(self, classifier: nn.Module, experts: list[nn.Module]):
        super().__init__()
        self.classifier = classifier
        self.experts = nn.ModuleList(experts)  # order: salt, blur, occlusion (labels 1, 2, 3)

    @torch.no_grad()
    def forward(self, x, route=None):
        """Returns (restored, route, probabilities). `route` = None -> predicted routing."""
        probs = torch.softmax(self.classifier(x), 1)
        if route is None:
            route = probs.argmax(1)
        out = x.clone()
        for k, expert in enumerate(self.experts):
            m = route == (k + 1)
            if m.any():
                out[m] = expert(x[m]).to(out.dtype)  # experts may run in fp16 under autocast
        return out, route, probs
