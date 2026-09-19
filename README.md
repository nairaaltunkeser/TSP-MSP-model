
deviants.py  –  sequence-violation streams 
--------
Builds streams in which the learned word A B C D is presented either intact
(standard) or with a controlled violation (deviant), so we can compare CA3 /
CA1 representations at the deviating position, and decode CA1 sequence
identity for standards vs deviants.

Deviant types

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

Design choices

 Background never uses letter A (so a chance A can't start a word), matching
  data.py.
Each deviant word draws its violating item X uniformly from the "wrong" set
  (not A, not the correct item, and for "AB"/"C"/"D" also not the background-
  forbidden nothing — A is allowed as an intruder value only for the "gap"
  filler? No: to stay clean we forbid A everywhere as an item value so onsets
  stay unambiguous).
By default a stream is homogeneous in deviant type (all deviant words are the
  same type) so RSMs/decoders are clean.
"""

TRIANING
----
training.py  –  Train forward-prediction model 

What it tracks per epoch

  pred_err_all          mean MSE forward-pred error over all timesteps
  pred_err_word         mean over within-word transitions (B/C/D positions)
  pred_err_bg           mean over background transitions
  acc_all               argmax accuracy on all timesteps
  acc_AB, acc_BC, acc_CD, acc_Dbg, acc_bg   per-transition accuracy
  hist_per_pos          per-epoch dict with all of the above

Key result to look for:
  acc_BC and acc_CD should rise above chance (1/16) and approach 1.0
  faster than acc_bg (which should stay at chance).
"""


TSP CURVES 
-----
tsp_restore_curves.py  –  Training-time TSP ablation with test-time restore.

Question

If the trisynaptic pathway (EC_in->DG, EC_in->CA3, DG->CA3, CA3->CA3, CA3->CA1)
is DISABLED throughout learning, what does the network learn — and does turning
TSP back ON at test time help, hurt, or do nothing?

Three training conditions, each run over multiple seeds:
    intact     : all pathways on during training
    tsp_off    : TSP disabled during training
    (optional) msp_off via --conditions

At every eval epoch, each model is evaluated TWICE on the same frozen weights:
    "test lesioned"  – pathway state identical to training
    "test restored"  – model.reset_lesions(), i.e. all pathways back on

For the intact condition the two evaluations are identical by construction; it
is plotted as the upper-bound reference.

Outputs

    <figdir>/tsp_restore_curves.png   mean +/- SEM accuracy curves over seeds
    <figdir>/tsp_restore_results.npz  raw per-seed arrays for re-plotting

Usage

    python tsp_restore_curves.py --seeds 0 1 2 --epochs 30
    python tsp_restore_curves.py --seeds 0 1 2 3 4 --epochs 30 --eval_every 2


DEVIANT ANALYSIS PY
-----
heatmap analysis figures

ANALYSIS PY 
-------
Figures

 / fig01_training_curves.png        per-transition accuracy + MSE over epochs
 
  fig02_position_accuracy.png      bars: accuracy at A→B / B→C / C→D / bg
  
  fig03_onset_aligned_accuracy.png trace aligned to A onset (-12..+12 steps)
  
  fig04_decoder_readout.png        linear-decoder accuracy from each layer
  
  fig05_overlap_rsm.png            sparse-code overlap RSM, all layers, init+settled
  
  fig06_initial_vs_settled.png     side-by-side init vs settled RSM (Schapiro Fig 2)
  
  fig07_confusion.png              confusion matrix: argmax(ECout) per word position
  
  fig08_rasters.png                population activity rasters
  
  fig09_weight_distributions.png   weight histograms per projection

  fig10_lesion_train.png           training-time lesion sweep
  
  fig11_lesion_test.png            test-time lesion (intact training, ablated eval)
  
  summary.txt                      key numbers



  example commands used:
  -----
