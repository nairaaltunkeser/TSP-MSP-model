"""
deviants.py  –  Oddball / sequence-violation streams for HipCat.

Builds streams in which the learned word A B C D is presented either intact
(standard) or with a controlled violation (deviant), so we can compare CA3 /
CA1 representations at the deviating position, and decode CA1 sequence
identity for standards vs deviants.

Every trial in HipCat is a two-hot moving window (current + previous item) and
state DECAYS between trials, so a "deviant" is fully specified by the sequence
of item indices we feed in. We therefore build explicit per-position item
sequences and hand them to the model timestep-by-timestep, exactly like a
normal stream. The generator returns, for every embedded word, a per-position
label of WHICH position is the deviating one (dev_pos), so analysis code can
align on it.

Deviant types
-------------
  "standard"  A  B  C  D                      (no violation; control)
  "AB"        A  X  C  D    X≠B                (early violation at position 1)
  "D"         A  B  C  X    X≠D                (terminal violation at position 3)
  "gap"       A  B  G  C  D  G=background      (intruding item breaks B→C adjacency;
                                               word is length-5, deviation at the
                                               inserted slot, original C/D shifted)
  "silent"    A  B  _  D    _ = SILENT (-1)     (C truly omitted: empty EC_in
                                               current slot at position 2)
  "C"         A  B  X  D    X≠C                (mid violation at position 2; bonus)
  "scramble"  random permutation of B C D after A (bonus)

Layout of each returned stream
------------------------------
  seq       [B, T] long   current item
  prev_seq  [B, T] long   previous item (right-shifted, t0=seq[:,0])
  next_seq  [B, T] long   next item     (left-shifted, wraps at T-1)
  word_pos  [B, T] long   -1 bg; 0..(L-1) position inside the presented word
  dev_pos   [B, T] long   -1 everywhere EXCEPT the deviating timestep of a
                          deviant word, where it holds that word-position index.
                          For standard words it is -1 everywhere.
  onset_mask[B, T] bool   True at each word onset (position 0)
  is_dev    [B, T] bool   True across ALL timesteps of a deviant word

Design choices
--------------
* Background never uses letter A (so a chance A can't start a word), matching
  data.py.
* Each deviant word draws its violating item X uniformly from the "wrong" set
  (not A, not the correct item, and for "AB"/"C"/"D" also not the background-
  forbidden nothing — A is allowed as an intruder value only for the "gap"
  filler? No: to stay clean we forbid A everywhere as an item value so onsets
  stay unambiguous).
* By default a stream is homogeneous in deviant type (all deviant words are the
  same type) so RSMs/decoders are clean, but you can request a mix.
"""

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