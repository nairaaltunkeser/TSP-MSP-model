

import numpy as np
import torch
import string

from data import ALPHABET_SIZE, sounds_idx, WORD, WORD_LEN
from model import SILENT

A_IDX = WORD[0]           # 'A'
B_IDX = WORD[1]
C_IDX = WORD[2]
D_IDX = WORD[3]

DEVIANT_TYPES = ["standard", "AB", "C", "D", "gap", "silent", "ABAD", "scramble"]


def _wrong_item(rng, correct_idx):
    """Pick an item index that is neither A (reserved for onsets) nor `correct_idx`."""
    while True:
        c = rng.randint(0, ALPHABET_SIZE)
        if c != A_IDX and c != correct_idx:
            return c


def _bg_item(rng):
    """Background item: never A."""
    while True:
        c = rng.randint(0, ALPHABET_SIZE)
        if c != A_IDX:
            return c


def _build_word(dev_type, rng):
    """
    Return (items, dev_position) for one word of the given deviant type.
      items         : list[int] of the presented items (length 4 or 5 for gap)
      dev_position  : int position index that is the violation, or -1 for standard
    """
    if dev_type == "standard":
        return [A_IDX, B_IDX, C_IDX, D_IDX], -1
    if dev_type == "AB":
        return [A_IDX, _wrong_item(rng, B_IDX), C_IDX, D_IDX], 1
    if dev_type == "C":
        return [A_IDX, B_IDX, _wrong_item(rng, C_IDX), D_IDX], 2
    if dev_type == "D":
        return [A_IDX, B_IDX, C_IDX, _wrong_item(rng, D_IDX)], 3
    if dev_type == "gap":
        # A B [gap] D  — still 4 pips. C is OMITTED and replaced by a
        # background/silent filler at position 2; D stays at position 3.
        # The deviation is the missing-C slot (position 2).
        return [A_IDX, B_IDX, _bg_item(rng), D_IDX], 2
    if dev_type == "silent":
        # A B _ D  — C is OMITTED: position 2 is a SILENT pip (index -1 ->
        # empty EC_in current slot). Still 4 pips; D stays at position 3.
        return [A_IDX, B_IDX, SILENT, D_IDX], 2
    if dev_type == "ABAD":
        # Standard ABCD with C replaced specifically by A → A B A D.
        # Deviation at position 2 (the C slot now holds A).
        return [A_IDX, B_IDX, A_IDX, D_IDX], 2
    if dev_type == "scramble":
        tail = [B_IDX, C_IDX, D_IDX]
        rng.shuffle(tail)
        # dev position = first spot where it differs from the canonical order
        dev = -1
        for i, (a, b) in enumerate(zip(tail, [B_IDX, C_IDX, D_IDX]), start=1):
            if a != b:
                dev = i
                break
        return [A_IDX] + tail, dev
    raise ValueError(f"unknown deviant type {dev_type!r}")


def make_deviant_stream(
    batch_size:   int = 64,
    T:            int = 400,
    dev_type:     str = "D",
    dev_fraction: float = 0.5,     # fraction of embedded words that are deviant
    p_word_onset: float = 0.05,
    min_gap:      int = 8,         # a bit larger to fit length-5 gap words
    seed:         int = None,
    mix_types:    list[str] | None = None,   # if set, sample deviant type per word
):
    """
    Continuous stream with embedded words, some standard and some deviant.

    If mix_types is given (e.g. ["AB","D","gap"]), each *deviant* word draws its
    type uniformly from that list; otherwise all deviants use `dev_type`.
    Standard words are always the canonical ABCD.

    Returns 7 tensors (see module docstring): seq, prev_seq, next_seq,
    word_pos, dev_pos, onset_mask, is_dev.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    seq        = np.zeros((batch_size, T), dtype=np.int64)
    word_pos   = np.full((batch_size, T), -1, dtype=np.int64)
    dev_pos    = np.full((batch_size, T), -1, dtype=np.int64)
    onset_mask = np.zeros((batch_size, T), dtype=bool)
    is_dev     = np.zeros((batch_size, T), dtype=bool)

    max_word_len = 4  # all words (incl. A B gap D) are 4 pips

    for b in range(batch_size):
        t = 0
        last_onset = -(10 ** 9)
        while t < T:
            can_start = (t <= T - max_word_len) and (t - last_onset >= min_gap)
            if can_start and (rng.rand() < p_word_onset):
                make_dev = rng.rand() < dev_fraction
                if make_dev:
                    this_type = (rng.choice(mix_types) if mix_types else dev_type)
                else:
                    this_type = "standard"

                items, dpos = _build_word(this_type, rng)
                L = len(items)
                if t + L > T:
                    # not enough room; fill with bg and move on
                    seq[b, t] = _bg_item(rng); t += 1; continue

                seq[b, t:t + L]      = items
                word_pos[b, t:t + L] = np.arange(L)
                onset_mask[b, t]     = True
                if this_type != "standard":
                    is_dev[b, t:t + L] = True
                    if dpos >= 0:
                        dev_pos[b, t + dpos] = dpos
                last_onset = t
                t += L
            else:
                seq[b, t] = _bg_item(rng)
                t += 1

    prev_seq = np.concatenate([seq[:, :1], seq[:, :-1]], axis=1)
    next_seq = np.concatenate([seq[:, 1:], seq[:, :1]], axis=1)

    to_t = lambda a, dt: torch.tensor(a, dtype=dt)
    return (
        to_t(seq,        torch.long),
        to_t(prev_seq,   torch.long),
        to_t(next_seq,   torch.long),
        to_t(word_pos,   torch.long),
        to_t(dev_pos,    torch.long),
        to_t(onset_mask, torch.bool),
        to_t(is_dev,     torch.bool),
    )


if __name__ == "__main__":
    # quick sanity print
    for dt in ["standard", "AB", "D", "gap"]:
        seq, prev, nxt, wp, dp, onset, isd = make_deviant_stream(
            batch_size=1, T=40, dev_type=dt, dev_fraction=1.0, seed=1)
        print(f"\n=== dev_type={dt} ===")
        print("cur :", seq[0].tolist())
        print("wpos:", wp[0].tolist())
        print("dpos:", dp[0].tolist())
