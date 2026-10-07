# Pre-registration: v22, the recipe on Qwen3.5-0.8B-Base

Committed before training. Same discipline as v9-v21. A torso swap for a constrained target,
not a candidate for the reference model: v21 stays the default whatever the result.

## Why

The Jibo port ([docs/jibo-port.md](../../docs/jibo-port.md)) has a hard ceiling of about
400 MB of memory for the whole decider process, and aims for under 2 seconds per decision.
Qwen3.5-2B's 24 layers hold 1.37 billion parameters, about 770 MB even at 4 bits, so v21
cannot run there. Qwen3.5-0.8B-Base has the same architecture (24 layers, 18 Gated DeltaNet
and 6 gated attention, the same tokenizer) with 0.50 billion layer parameters, about 280 MB
at 4 bits. Nothing has measured what the recipe gives on it.

What is known: generation matters more than size (Qwen3.5 at 2B beats Qwen3 at 8B read
frozen), within Qwen3.5 size still matters (2B to 4B is +0.147 frozen), and training adds
more on a weaker torso (+0.11 over Qwen3-1.7B's frozen reading, +0.10 over Qwen3.5-2B's
for v21) ([history.md](../history.md#what-the-torso-knows-untrained)).

## What is being changed

`base_model: "Qwen/Qwen3.5-0.8B-Base"` in v21b's config, and nothing else: the same rows,
teacher file, LoRA rank and targets (the 0.8B has the same module names), pointer head,
seed 5, one epoch of the same steps (`configs/experiments/v22.yaml`). Data from
`training/recipe.sh build fetch multistep generated adequacy distill`, checked against
`data/SHA256SUMS` as every frozen-target run is.

Hardware: one NVIDIA A100 80 GB on Hugging Face Jobs, with v21b's FAST settings (speed
only) except gradient checkpointing (see the amendment below). This is the first run off the RTX 3090 and the H100 hosts. Bf16 rounding differs
between GPUs; that is noise, not a change of recipe.

Budget: under $15 of Hugging Face credits for this run and its evaluation, at $2.50 an hour.
The throughput of the first steps is read from the log; if it projects past the budget, the
run is cancelled and reported here as not completed, with no results taken from it. The Job
also has a hard timeout that keeps the worst case inside the budget.

*Amended before training (2026-10-07): the first launch ran out of GPU memory between steps 40
and 60. v21b's FAST settings turn gradient checkpointing off, which fit when the AWS runner split
each micro-batch over 8 GPUs, but not on one A100 80 GB. Checkpointing is back on; it recomputes
activations instead of storing them and changes no result. The failed launch cost about $1.10 of
the budget. Nothing else changed.*

## Baselines

v21 as released (seed 5), and the six-seed mean and SD of v21b on one host
([results.md](../../evaluation/results.md#v21-the-released-model)):

| evaluation | v21 (seed 5) | v21b six-seed mean (SD) |
| --- | --- | --- |
| JevBench public at 4096, tasks right of 231 | 176 | 175.0 (2.4) |
| JevBench tiers easy / standard / hard | 48 / 67 / 61 | |
| JevBench Brier / ECE | 0.323 / 0.064 | 0.331 (0.008) Brier |
| MuSiQue / ContractNLI / BoardgameQA | 0.882 / 0.865 / 0.821 | 0.889 / 0.863 / 0.811 |
| HotpotQA (never trained on) | 0.746 | 0.752 (0.014) |
| held-out short tasks (6,000) | 0.650 | 0.644 (0.004) |
| HelpSteer2 adequacy / generated adequacy (balanced) | 0.739 / 0.788 | 0.735 / 0.802 |
| generated documents, v16's / v18's set | 0.849 / 0.741 | 0.856 / 0.757 |

## Predictions

Ranges, not point bets; a result outside its range fails the prediction.

1. **JevBench public at 4096: 145 to 165 tasks** (0.628 to 0.714). Below v21 by 11 to 31
   tasks; above v7's 154 on Qwen3-1.7B is not predicted either way.
2. **Easy tier at least 46 of 48.** The loss is in standard and hard: standard 55 to 64,
   hard 45 to 56.
3. **Multi-step sets fall, HotpotQA the most:** MuSiQue 0.80 to 0.87, ContractNLI 0.80 to
   0.86, BoardgameQA 0.70 to 0.80, HotpotQA 0.62 to 0.72.
4. **Short classification falls least:** held-out short tasks 0.60 to 0.65.
5. **Adequacy:** HelpSteer2 0.66 to 0.73, generated 0.72 to 0.80.
6. **Calibration survives the smaller torso:** JevBench ECE at most 0.09 after the recipe's
   calibration; Brier 0.34 to 0.40.

## The bar for the Jibo port

Decided before the run. The 0.8B checkpoint goes forward as the port's model if it scores at
least **150 of 231** on JevBench at 4096, at least **46 of 48** on the easy tier, and an ECE of
at most **0.10**. Below that, the port continues with it as an engineering vehicle only (its
runtime work does not depend on accuracy), and the owner decides between a further
preregistered run and stopping.

## Also reported (no prediction)

- JevBench public at a **512-token window**, the Jibo runtime's default, against v21 at 512.
  Expected lower for both, through `long_policy` and the document families, which do not
  fit; the difference between the two models at 512 is what matters for the robot.
- Training throughput (steps per second), peak GPU memory and wall clock on the A100, and
  the exact Hugging Face cost of the run.

## How it will be used

Only by the Jibo port. The checkpoint is published to the owner's Hugging Face account as a
private model, exported to the port's weight format, and taken through the port's parity
stages. A further run truncating the torso to fewer layers would need its own
preregistration.

## Outcome (added after the run)

Nothing above this section was edited after training. **v22 meets the bar for the Jibo port**
(156 of 231 at 4096, 48 of 48 easy, ECE 0.054) and goes forward as the port's model. Of the six
predictions, four held; 3 failed because MuSiQue came in *above* its range, and 6 failed on
Brier.

| | prediction | v21 (seed 5) | v22 | |
| --- | --- | --- | --- | --- |
| 1 | JevBench at 4096: 145 to 165 | 176 | 156 | pass |
| 2 | easy >= 46; standard 55 to 64; hard 45 to 56 | 48 / 67 / 61 | 48 / 55 / 53 | pass |
| 3 | MuSiQue 0.80-0.87, ContractNLI 0.80-0.86, BoardgameQA 0.70-0.80, HotpotQA 0.62-0.72 | 0.882 / 0.865 / 0.821 / 0.746 | **0.890** / 0.845 / 0.774 / 0.635 | FAIL (MuSiQue above) |
| 4 | held-out short tasks 0.60 to 0.65 | 0.650 | 0.640 | pass |
| 5 | HelpSteer2 0.66 to 0.73, generated 0.72 to 0.80 | 0.739 / 0.788 | 0.688 / 0.738 | pass |
| 6 | JevBench ECE <= 0.09; Brier 0.34 to 0.40 | 0.064; 0.323 | 0.054; **0.425** | FAIL (Brier) |

**The smaller torso lost where reading is long and indirect, not where it is short.** HotpotQA,
never trained on, fell most (0.746 -> 0.635), then BoardgameQA (0.821 -> 0.774) and the
adequacy sets (about -0.05 each); held-out short tasks barely moved (0.650 -> 0.640), and
MuSiQue held level (0.882 -> 0.890), which the prediction did not allow for. The generated
documents split: v16's set 0.849 -> 0.809, v18's 0.741 -> 0.749. Instruction-flip pairs are at
chance (both right 0.217), as expected for v21b's rows, which hold no flips.

**JevBench: 156 of 231 at 4096**, 20 below v21. Easy is untouched; the loss is standard
(67 -> 55) and hard (61 -> 53). Calibration held in the ECE sense (0.064 -> 0.054) but the Brier
score rose 0.323 -> 0.425: the 0.8B is wrong more often, and its wrong answers are not
uncertain enough to keep Brier in range. JevBench's paraphrase consistency 0.806 (29 of 36
pairs agree; v21 at 512, 0.917).

**At the Jibo runtime's 512-token window the score does not move**: v22 156 at 512 (9 tasks
gained and 9 lost, all in the hard tier; ECE 0.045, Brier 0.415), and v21 175 at 512 against
176 at 4096. At 512, v21 against v22 is +27 / -8 for v21 (McNemar p = 0.002), the difference
entirely in standard (-12) and hard (-7). The window the robot can afford costs nothing
measurable on JevBench; the torso it can afford costs 19 tasks.

**Run.** One NVIDIA A100-SXM4 80 GB on Hugging Face Jobs: 3,738 steps in 2 h 55 min
(0.40 steps/s at the end, with gradient checkpointing), peak GPU memory 7.0 GiB; final
validation loss 0.463, accuracy 0.835. Cost: $7.78 for the run (3 h 7 min including
calibration and evaluation), $1.03 for the launch that ran out of memory, and $0.05 for
JevBench on an L4 (rerun separately: the run's own JevBench step failed on a path error in the
Job script, before loading anything); **$8.86 in all**, inside the $15 budget.
