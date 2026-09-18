"""
data.py  –  Stream generation for HipCat (faithful) on the ABCD-in-noise paradigm.

Faithful changes vs. previous version
-------------------------------------
* Returns prev_seq alongside seq and next_seq, so model.py can build a
  Schapiro-style two-hot moving window on EC_in:
      EC_in[current_item]  = 1.0
      EC_in[previous_item] = 0.9     (temporal asymmetry – forward-biased)
  This is how Schapiro 2017 gives the network access to the preceding item
  without any layer carry-over between trials.
* word_pos still encodes position inside the word (-1=bg, 0=A,1=B,2=C,3=D).
* Background timesteps never use letter A, so a chance "A" never starts a
  pseudo-word and inflates A->B baseline accuracy.
* The "previous" item at t=0 is set to itself (no real history); training
  code should mask t=0 out of analyses.

Outputs of make_embedded_stream
-------------------------------
  seq        : [B, T] long  – current item
  prev_seq   : [B, T] long  – previous item (seq shifted RIGHT, t=0 = seq[:,0])
  next_seq   : [B, T] long  – next item     (seq shifted LEFT,  t=T-1 wraps)
  word_pos   : [B, T] long  – -1 background, 0..3 inside-word position
  onset_mask : [B, T] bool  – True at A onsets
"""

import torch
import numpy as np
import string

# ── Alphabet ──────────────────────────────────────────────────────────────────
ALPHABET_SIZE = 16
letters       = list(string.ascii_uppercase[:ALPHABET_SIZE])
sounds        = letters
sounds_idx    = {ch: i for i, ch in enumerate(sounds)}
sound_size    = len(sounds)

# The embedded word: A B C D  (indices 0,1,2,3)
WORD     = [sounds_idx["A"], sounds_idx["B"], sounds_idx["C"], sounds_idx["D"]]
WORD_LEN = len(WORD)


# ── Embedded-word stream ──────────────────────────────────────────────────────

def make_embedded_stream(
    batch_size:   int   = 64,
    T:            int   = 400,
    p_word_onset: float = 0.05,
    min_gap:      int   = 6,
    seed:         int   = None,
):
    """Continuous stream with random background and occasional ABCD word."""
    rng = np.random.RandomState(seed) if seed is not None else np.random

    seq        = np.zeros((batch_size, T), dtype=np.int64)
    word_pos   = np.full((batch_size, T), -1, dtype=np.int64)
    onset_mask = np.zeros((batch_size, T), dtype=bool)

    for b in range(batch_size):
        t = 0
        last_onset = -(10 ** 9)
        while t < T:
            can_start  = (t <= T - WORD_LEN) and (t - last_onset >= min_gap)
            start_word = can_start and (rng.rand() < p_word_onset)
            if start_word:
                end = min(t + WORD_LEN, T)
                n   = end - t
                seq[b, t:end]      = WORD[:n]
                word_pos[b, t:end] = np.arange(n)
                onset_mask[b, t]   = True
                last_onset         = t
                t                 += n
            else:
                # avoid letting bg coincidentally start the ABCD pattern
                while True:
                    c = rng.randint(0, sound_size)
                    if c != WORD[0]:           # bg never == 'A'
                        break
                seq[b, t] = c
                t        += 1

    # prev_seq = seq shifted RIGHT by 1, with t=0 set to seq[:,0] (no real history)
    prev_seq = np.concatenate([seq[:, :1], seq[:, :-1]], axis=1)
    # next_seq = seq shifted LEFT by 1, with last col wrapping (t=T-1 ignored later)
    next_seq = np.concatenate([seq[:, 1:], seq[:, :1]], axis=1)

    return (
        torch.tensor(seq,        dtype=torch.long),
        torch.tensor(prev_seq,   dtype=torch.long),
        torch.tensor(next_seq,   dtype=torch.long),
        torch.tensor(word_pos,   dtype=torch.long),
        torch.tensor(onset_mask, dtype=torch.bool),
    )


# ── Pure-random control stream ────────────────────────────────────────────────

def make_random_stream(
    batch_size: int = 64,
    T:          int = 400,
    seed:       int = None,
):
    """Pure i.i.d. stream over the full alphabet. No word embedded."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    seq = rng.randint(0, sound_size, size=(batch_size, T)).astype(np.int64)
    prev_seq   = np.concatenate([seq[:, :1], seq[:, :-1]], axis=1)
    next_seq   = np.concatenate([seq[:, 1:], seq[:, :1]], axis=1)
    word_pos   = np.full((batch_size, T), -1, dtype=np.int64)
    onset_mask = np.zeros((batch_size, T), dtype=bool)
    return (
        torch.tensor(seq,        dtype=torch.long),
        torch.tensor(prev_seq,   dtype=torch.long),
        torch.tensor(next_seq,   dtype=torch.long),
        torch.tensor(word_pos,   dtype=torch.long),
        torch.tensor(onset_mask, dtype=torch.bool),
    )


# Backward-compatible alias
def make_random_control_stream(batch_size=64, T=400, seed=None):
    return make_random_stream(batch_size=batch_size, T=T, seed=seed)


if __name__ == "__main__":
    seq, prev, nxt, wp, _ = make_embedded_stream(batch_size=4, T=60, seed=0)
    print("t   prev  cur  next  pos")
    for t in range(60):
        print(f"{t:2d}   {prev[0,t].item():2d}    {seq[0,t].item():2d}    "
              f"{nxt[0,t].item():2d}    {wp[0,t].item():2d}")
    # Sanity: at C (pos==2) the next item must be D
    for b in range(4):
        for t in range(59):
            if wp[b, t].item() == 2:
                assert nxt[b, t].item() == WORD[3], f"C->D violated at b={b} t={t}"
    print("\n✓ word_pos sanity check passed")