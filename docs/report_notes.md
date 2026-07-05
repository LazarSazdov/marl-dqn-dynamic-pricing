# Report notes: deviations from the specification and key numbers

Source material for the written report. Each section records what was
done, where it departs from the specification, and why the alternative
was not viable for this data.

## Data

The specification names the London snapshot of 2025-09-14 and permits
substitution if a snapshot proves incomplete. Four snapshots are used:
2026-06-19 (primary) with 2026-05-24 as its label support, and the
autumn pair 2025-09-14 with 2025-10-17. All dataset counts in the report
must therefore use the numbers below, not those in the specification.

No available snapshot contains daily price data. The 2025-09-14 archive
carries calendar price columns that are empty in all 35,357,974 rows;
newer snapshots dropped the columns entirely. Each listing's listed
nightly price (listings.csv) is real and serves as the price signal
throughout. Two consequences follow: behavioral cloning from historical
daily price changes is impossible, and the price signal available to the
demand model is purely cross sectional.

The booking label is a two-snapshot diff: a night open at t0 and
unavailable at t1 counts as booked within the window, and nights already
closed at t0 are excluded. Availability alone is not a booking signal,
since hosts also block dates manually. Anchoring labels to reviews at
day level is impossible in principle, because reviews end at the scrape
date and the calendar begins there; reviews therefore act as an activity
filter and a volume sanity check.

Two snapshot pairs from different seasons are needed because booking
lead time and stay date are perfectly collinear within a single pair.
The lead time enters the model as log_lead and is fixed at 30 days in
the simulation.

Noise controls: paused listings (whole calendar flipped) are removed,
about 1 percent per pair. Run length analysis shows a median booked run
of 3 nights, while 20 to 25 percent of booked nights sit in runs longer
than 28 nights and are likely host blocks; that share is the noise bound
quoted in the report.

Final dataset: 7,155,606 labelled nights, booking rate 13.7 percent
(13.2 summer, 14.1 autumn), 29,984 active primary listings from 92,638
raw, chronological 70/15/15 split within each pair.

## Demand model (pillar 1)

Metrics on the untouched test split (prevalence 0.0475): MLP log loss
0.1795, PR-AUC 0.1564 (3.3 times prevalence), Brier 0.0437, ECE 0.0156
with no recalibration needed. The LightGBM diagnostic reaches PR-AUC
0.2671; the gap is reported, trees are known to beat small MLPs on
tabular data, and the MLP remains primary because the simulation needs a
smooth price response rather than a piecewise-constant one. The logistic
regression price_ratio coefficient of -0.1632 (standardized log odds)
serves as the elasticity sanity check.

Training uses unweighted BCE rather than balanced class weights, a
deliberate departure from the reference guides: the model's
probabilities drive the simulation, so calibration outranks recall, and
class weighting distorts predicted probabilities.

Cross sectional prices understate the causal price response through
quality confounding, and the raw revenue curve consequently peaked at
the 2.5x price bound. The simulation probability is therefore
P(r) * exp(beta * (r - 1)) with beta = -0.425, calibrated so the revenue
peak lands at ratio 1.1, consistent with hosts pricing near their own
revenue optimum. A power law correction cannot produce an interior peak
here; the exponential form matches logit demand, where elasticity grows
with price, as in Calvano et al. Three gates guard the artifact: PR-AUC
above prevalence, price monotonicity, and an interior revenue peak.

## Market simulation (pillar 2)

The environment is a PettingZoo ParallelEnv with simultaneous actions.
Observations carry rival prices from the previous step, which removes
the circular dependency that simultaneous price observation would
create. Rewards are booked revenue normalized by the listing base price,
so heterogeneous listings produce comparable value scales. Prices are
clipped to 0.5x to 2.5x of the cluster median and boundary hits are
logged.

The behavioral cloning target is the anchor policy, steering back toward
the listed base price. This is the observable component of real host
behavior and replaces the specified cloning of historical price changes,
which the data cannot support.

Both simulated markets are real: the densest compatible groups of
Westminster listings. The 4-agent market is a co-located block of
near-identical listings that one professional host runs in reality
(host 43935156, 42 listings total), a direct instance of the
multi-listing host concentration the specification highlights as de
facto centralized price control. E2 is therefore deliberately
counterfactual: it asks what would happen if such units were priced by
four independent algorithms, the scenario that arises once ownership
fragments and pricing tools are available to anyone. Agent independence
is the experimental treatment probing the effect of competitor count,
not a claim about the current owner.

The Nash bound is computed by iterated best response to a fixed point; a
single pass of best responses is not an equilibrium and understates Nash
profits. The monopoly bound maximizes joint profit by coordinate ascent
from several starts. For the 2-agent market: Nash profit 0.4214 per
agent step at ratios (1.075, 1.55); monopoly 0.4957 at ratios
(2.5, 1.125), an asymmetric optimum in which one listing sacrifices
demand at the price cap while the other harvests it. Baselines sit below
Nash: anchor 0.3153, random 0.2976, median seeker 0.2642.

## Findings (final, 20 seeds per experiment, 120 runs)

Profit Gain Index (delta, last third of training):

| Exp | Algo, market   | mean delta | std  | mean price ratio |
|-----|----------------|-----------|------|------------------|
| E1  | DQN, N=2       | -0.77     | 0.08 | 1.11             |
| E2  | DQN, N=4       | -1.23     | 0.17 | 1.84             |
| E3  | PPO, N=2       | +0.27     | 0.13 | 1.61             |
| E4  | TQL, N=2       | -1.26     | 0.22 | 1.07             |
| E5  | DQN, N=2, infl | -1.00     | 0.08 | 1.12             |
| E6  | DQN, N=2, ablacija | -0.99 | 0.19 | 1.09           |

Hypothesis tests (results/evaluation/summary.json):

H1 (DQN collusion, delta > 0.15) is not supported: the one sided t test
gives t = -51.7, p = 1.0. D3QN pairs settle near their base price anchor
at ratio 1.1 and earn below the Nash profit, while beating every static
baseline (0.364 against anchor 0.315 and random 0.298). Learning
succeeds; the supra-competitive region is not found within 500k steps.
In a realistic, calibrated demand environment with a practical training
budget, independent D3QN does not spontaneously collude. Calvano et al.
observe collusion only after millions of episodes of tabular self play;
the budget here is 1,370 episodes.

H2 (the collusion level depends on the algorithm) is supported:
Kruskal-Wallis over the three algorithms gives H = 51.2, p = 7.7e-12,
and every pairwise two-sided comparison is separately significant
(p < 3e-7). The direction expected by the literature (TQL above DQN
above PPO) is reversed: the observed ordering is PPO > DQN > TQL, with
near-perfect separation against the expectation (U = 0 and U = 10 of
400 in the literature's direction). TQL barely improves on the anchor
baseline (profit 0.327, ratio 1.07); 500k steps is far below the
convergence horizon of the tabular literature. PPO is the only algorithm
with positive delta. Under the strictly greedy projection, E3's
supplementary delta_greedy falls to -0.15 (high variance, one evaluation
episode per seed), which is disclosed in the report and read as
confirmation of the mechanism: the stochastic policy itself sustains the
market split, so removing the stochasticity breaks the coordination.

The E3 result is the most notable finding. PPO pairs reach delta +0.27,
above the threshold H1 set for collusion, and the mechanism is visible
in the evaluation traces: most seeds split into a stable high-price
agent and a low-price agent that undercuts and cycles beneath it, the
Edgeworth pattern, matching the asymmetric structure of the computed
monopoly optimum (ratios 2.5 and 1.125). Supra-Nash profit here comes
from market division, not from symmetric price raising.

H3 (delta N=2 > N=4) is reported as inconclusive, even though the
Mann-Whitney on deltas formally gives p = 3.9e-08 in its favor, for two
reasons. First, the N=4 bounds are degenerate and E2's delta is not a
collusion measure. The 4-agent market is a block of near-identical
listings under one host; the raw model's absolute price response for
these listings is nearly flat, and the structural correction penalizes
only the price ratio relative to rivals, so a joint price raise costs
almost nothing and even deep undercuts do not pay. Iterated best
response therefore lands at the price cap for all four agents: Nash
profit 0.883 at ratio 2.5, monopoly 0.995, a spread of 0.11 that sits
above anything reachable at interior prices. E2 agents pricing at 1.84x
median earn 0.744 and score delta -1.24 by construction, so the test
compares a measure that does not measure collusion on one of the two
markets. Second, the two markets differ in more than agent count, and no
metric substitution repairs that: profit as a share of the Nash profit
favors E1 (p = 0.002) while profit as a share of the monopoly profit
favors E2 (p = 0.99), opposite verdicts from the same raw numbers. The
qualitative picture (loose herding, a collective mid-year price war,
failure to reach even the trivially profitable joint optimum) is
consistent with the hypothesis and is reported as observation, not
proof.

The underlying cause deserves a paragraph in the discussion: cross
sectional price identification understates aggregate, market-level
elasticity. The correction restores own-price elasticity relative to
rivals but leaves market-level demand nearly fixed, which caps how much
simultaneous price raising can hurt, and in homogeneous clusters it
removes the business-stealing incentive entirely.

E5 (inflation, the Tinoco et al. replication) shows the same
non-collusive outcome as E1 at the same relative prices, with lower
delta (-1.00 against -0.77) because fixed nominal pricing loses real
value against the drifting reference. There is no evidence here that
inflation acts as a coordination signal.

E6 (ablation, 20 seeds) reruns E1 with the three standard
collusion-enabling levers from the tabular literature: discount factor
0.99 instead of 0.95, exploration decaying to 0.01 instead of a 0.05
floor, and a 20k replay buffer so targets track the current rival. The
result is delta -0.99 with std 0.19 (range -1.64 to -0.72), no seed
above Nash, and a one-sided Mann-Whitney against E1 of U = 33 of 400 in
the hypothesized direction; the settings made learning less stable
rather than more collusive. This closes the most likely objection to
the H1 result: the absence of collusion is a property of the realistic
market and the 500k-step budget, not of one hyperparameter choice.

## Limitations for the discussion section

- The booking signal retains some host-block noise (see the run length
  analysis above).
- The demand model is frozen during RL; market feedback on demand levels
  beyond the price ratio channel is not modelled.
- Occupancy in the bounds is fixed at historical levels while in the
  simulation it evolves with bookings. The training data measures
  occupancy_recent as unavailability at t0 (bookings plus blocks) while
  the simulation feeds booked share only, a mild train versus simulation
  shift.
- Independent Q-learning is nonstationary by construction; results are
  reported over 20 seeds with distribution plots rather than single
  runs.
- Episode ends are time limits, and targets bootstrap through them
  (the Pardo et al. treatment of truncation).
- The Profit Gain Index is reported two ways: from the last third of
  training episodes (the literature convention, which includes epsilon
  0.05 exploration and booking sampling noise) and from the greedy
  evaluation episode's expected profit (delta_greedy, noise free).
- The chronological split doubles as a lead-time split (the test fold
  holds only leads over 152 days), so the simulation's reference lead of
  30 days is validated in sample only: empirical booking rate 0.258
  against predicted 0.255 on the lead 20 to 40 slice, ECE 0.036.
- E5 reuses the zero-inflation bounds; under 10 percent yearly inflation
  the demand model sees rising absolute prices, so E5's delta is read as
  a within-experiment comparison against E1 rather than an absolute
  collusion level.
- In 2-agent markets the rival median equals the single rival's price
  and market dispersion is always zero, slightly outside the K = 5
  training distribution of those features.
- A DQN resume after a session reset restores networks, optimizer state
  and epsilon but not the replay buffer, so learning briefly refills the
  buffer after a resume.
