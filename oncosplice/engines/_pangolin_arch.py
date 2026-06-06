"""Independent re-implementation of the Pangolin network architecture.

NOTICE
------
This module is an **independent re-implementation** of the splice-prediction
network described in:

    Zeng & Li, "Predicting RNA splicing from DNA sequence using Pangolin",
    Genome Biology 23, 103 (2022).  https://doi.org/10.1186/s13059-022-02664-4

The original Pangolin source code is licensed GPL-3.0 and is **not** used here.
Only the published architectural facts are reproduced — the residual-CNN
backbone (the ``W``/``AR`` dilation schedule, ``L=32`` channels) and the eight
output heads — together with the parameter names required to load the
publicly released trained weights. Layer/sub-module names are dictated by the
released ``state_dict`` (a functional interface, not creative expression) so
that the original weight files can be consumed; everything else (control flow,
comments, organisation) is written from scratch for oncosplice.

The trained weight files themselves remain © the original Pangolin authors and
are distributed under their own (GPL-3.0) terms. oncosplice does not bundle
them; they are fetched separately into the user weight cache. This re-write
exists so the *architecture code* needed to run those weights is MIT-clean and
self-contained, leaving oncosplice itself unencumbered.
"""
from __future__ import annotations

import numpy as np

# Channel width and the per-residual-unit convolution-window / atrous-rate
# schedule. These are published facts (Zeng & Li 2022) and are required to be
# exactly these values for the released weights to load and run correctly.
PANGOLIN_CHANNELS = 32
PANGOLIN_W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                         21, 21, 21, 21, 41, 41, 41, 41])
PANGOLIN_AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                          10, 10, 10, 10, 25, 25, 25, 25])


def build_pangolin_class():
    """Return the ``Pangolin`` ``nn.Module`` class. Imports torch lazily so the
    rest of oncosplice stays importable without a torch install.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def _same_padding(window: int, atrous: int) -> int:
        # "same"-length 1-D dilated convolution: pad each side by half the
        # receptive-field growth. window is always odd, so this is exact.
        return atrous * (window - 1) // 2

    class _ResidualUnit(nn.Module):
        """Pre-activation residual unit: (BN→ReLU→conv) twice, then add input.

        Sub-module names (``bn1``/``conv1``/``bn2``/``conv2``) are fixed by the
        released Pangolin weights.
        """

        def __init__(self, channels: int, window: int, atrous: int):
            super().__init__()
            pad = _same_padding(window, atrous)
            self.bn1 = nn.BatchNorm1d(channels)
            self.conv1 = nn.Conv1d(channels, channels, window, dilation=atrous, padding=pad)
            self.bn2 = nn.BatchNorm1d(channels)
            self.conv2 = nn.Conv1d(channels, channels, window, dilation=atrous, padding=pad)

        def forward(self, x):
            h = self.conv1(torch.relu(self.bn1(x)))
            h = self.conv2(torch.relu(self.bn2(h)))
            return x + h

    class Pangolin(nn.Module):
        """Pangolin splice-prediction network (independent re-implementation).

        Input  : one-hot sequence, shape ``(B, 4, L)``.
        Output : ``(B, 12, L - CL)`` where ``CL = 2*sum(AR*(W-1)) = 10000`` — i.e.
                 5,000 bp are cropped from each end. The 12 channels are four
                 tissue groups of three (a 2-way splice softmax flattened to its
                 positive channel plus a usage sigmoid), interleaved exactly as
                 the released heads emit them.
        """

        def __init__(self, channels=PANGOLIN_CHANNELS, W=PANGOLIN_W, AR=PANGOLIN_AR):
            super().__init__()
            self._W = np.asarray(W)
            self._AR = np.asarray(AR)
            # Stem + the running skip accumulator's input projection.
            self.conv1 = nn.Conv1d(4, channels, 1)
            self.skip = nn.Conv1d(channels, channels, 1)
            # 16 residual units; a 1x1 conv taps the skip path after every
            # 4th unit (and after the last). Names `resblocks`/`convs` match
            # the released weights.
            self.resblocks = nn.ModuleList()
            self.convs = nn.ModuleList()
            n = len(self._W)
            for i in range(n):
                self.resblocks.append(
                    _ResidualUnit(channels, int(self._W[i]), int(self._AR[i]))
                )
                if (i + 1) % 4 == 0 or (i + 1) == n:
                    self.convs.append(nn.Conv1d(channels, channels, 1))
            # Eight output heads: alternating 2-channel (splice) and 1-channel
            # (usage) projections, one (splice, usage) pair per tissue.
            self.conv_last1 = nn.Conv1d(channels, 2, 1)
            self.conv_last2 = nn.Conv1d(channels, 1, 1)
            self.conv_last3 = nn.Conv1d(channels, 2, 1)
            self.conv_last4 = nn.Conv1d(channels, 1, 1)
            self.conv_last5 = nn.Conv1d(channels, 2, 1)
            self.conv_last6 = nn.Conv1d(channels, 1, 1)
            self.conv_last7 = nn.Conv1d(channels, 2, 1)
            self.conv_last8 = nn.Conv1d(channels, 1, 1)
            # Total context cropped by the network (5,000 bp per side).
            self._crop = int(np.sum(self._AR * (self._W - 1)))

        def forward(self, x):
            h = self.conv1(x)
            skip = self.skip(h)
            tap = 0
            n = len(self.resblocks)
            for i in range(n):
                h = self.resblocks[i](h)
                if (i + 1) % 4 == 0 or (i + 1) == n:
                    skip = skip + self.convs[tap](h)
                    tap += 1
            # Crop the context flanks (negative padding == centre crop).
            skip = F.pad(skip, (-self._crop, -self._crop))
            heads = [
                F.softmax(self.conv_last1(skip), dim=1),
                torch.sigmoid(self.conv_last2(skip)),
                F.softmax(self.conv_last3(skip), dim=1),
                torch.sigmoid(self.conv_last4(skip)),
                F.softmax(self.conv_last5(skip), dim=1),
                torch.sigmoid(self.conv_last6(skip)),
                F.softmax(self.conv_last7(skip), dim=1),
                torch.sigmoid(self.conv_last8(skip)),
            ]
            return torch.cat(heads, dim=1)

    return Pangolin
