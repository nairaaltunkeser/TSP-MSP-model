# Pupil analysis scripts

These scripts analyse pupil and behavioural responses to normal and deviant auditory sequences, with a focus on control vs optogenetic manipulation sessions.

## Main scripts

- `pupil_analysis_main.py` — main normal vs deviant pupil analysis, including session/group summaries, difference traces and statistical tests.
- `pupil_analysis_dev.py` — earlier/development version of the main pupil analysis.
- `pupil_difference_control_vs_opto.py` — focuses on deviant-minus-normal pupil traces for control vs opto sessions and runs cluster-permutation tests.
- `pupil_thesis_figure.py` — generates the main thesis-style control vs opto figure and associated statistics.
- `pupil_per_session_traces.py` — plots normal and deviant pupil traces separately for each control/opto session.

## Adaptation and learning

- `pupil_adaptation.py` — compares early and late test-phase responses across session groups.
- `pupil_adaptation_control_opto.py` — same adaptation analysis restricted to control and opto sessions.
- `pupil_learning.py` — tests across-session learning within the June block.
- `pupil_learning_june_july.py` — runs the learning analysis separately for June and July blocks.

## Behavioural controls

- `pupil_behaviour_control.py` — checks hit rate, lick timing and other behavioural measures across groups.
- `pupil_behaviour_control_opto.py` — behavioural control analysis restricted to control and opto sessions.
- `pupil_within_session_hitrate.py` — tests whether hit rate changes across a session similarly for normal and deviant trials.

comments
----
The scripts expect the relevant `.h5` data files to be in the same folder and save figures/statistics into their own output directories.


main files use control, opto (standard optogenetically manipulated), opto_late (data collected after virus exp date), control with different deviant

figures used in the end were with only control and opto groups
