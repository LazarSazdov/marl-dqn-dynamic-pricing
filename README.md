# Multi-Agent Reinforcement Learning for Dynamic Pricing

Do independent pricing algorithms learn to collude? This project studies
algorithmic collusion on a simulated London short-term rental market:
Dueling Double DQN agents, each controlling one real Airbnb listing, set
nightly prices inside a market simulation whose demand mechanics are
learned from Inside Airbnb data. Collusion is quantified with the Profit
Gain Index Δ against empirically computed Bertrand-Nash and monopoly
bounds (Calvano et al., 2020).

**Students:** Milan Sazdov (SV21/2023), Lazar Sazdov (SV25/2023)
**Course:** Osnovi računarske inteligencije, FTN, University of Novi Sad

## Results

100 training runs (5 experiments × 20 seeds × 500k steps). Δ = 0 means
competitive (Nash) profits, Δ = 1 full cartel; the collusion threshold
tested is Δ > 0.15.

| Experiment | Setup | mean Δ ± std | mean price / median |
|---|---|---|---|
| E1 | D3QN, 2 agents | −0.77 ± 0.08 | 1.11 |
| E2 | D3QN, 4 agents | −1.23 ± 0.17 | 1.84 |
| E3 | PPO, 2 agents | **+0.27 ± 0.13** | 1.61 |
| E4 | Tabular Q, 2 agents | −1.26 ± 0.22 | 1.07 |
| E5 | D3QN, 2 agents, 10%/yr inflation | −1.00 ± 0.08 | 1.12 |

- **H1 (DQN collusion): not supported** (one-sided t-test, p = 1.0). DQN
  agents beat every static baseline yet price near the market median,
  below Nash profits. In a realistic, data-calibrated market with a
  practical training budget, collusion is not an automatic emergent
  property; the classic results arise after millions of episodes in
  stylized markets.
- **H2 (collusion depends on the algorithm): supported** (Kruskal-Wallis
  H = 51.2, p = 7.7e-12), with the observed ordering reversing the
  literature's expectation. PPO is the only algorithm with
  supra-competitive profits, achieved through tacit market division: one
  agent holds the price ceiling while the other undercuts beneath it,
  mirroring the asymmetric structure of the computed monopoly optimum.
- **H3 (fewer agents collude more): inconclusive by design.** The formal
  Δ test is not interpretable here: the 4-agent market's modeled demand is
  nearly price-inelastic, its computed Nash sits at the price cap, and
  alternative profit normalizations yield opposite verdicts. The
  qualitative picture (collective price wars, no sustained coordination
  among four agents) is consistent with the hypothesis but is reported as
  observation, not proof.
- **E5 (inflation): no coordination signal.** Behavior matches E1 at the
  same relative prices; nominal pricing loses real value instead.

Full numbers in `results/evaluation/`, figures via `scripts/make_figures.py`,
working notes and all specification deviations in `docs/report_notes.md`,
slide deck in `docs/presentation.html`.

## Method and design rationale

The system is two pillars: a supervised demand model learned from data,
then a market simulation that uses it as world mechanics. Historical data
alone cannot train pricing agents, because it only records the market's
response to prices that were actually set; agents must probe arbitrary
prices and observe outcomes, which requires a model-based simulator.

### Booking labels (two-snapshot diff)

A night is labelled booked if it was open in one snapshot and unavailable
in the next (26-day window), on listings with recent review activity.

- `available = false` alone is not a booking label: it conflates bookings
  with host blocks. The diff isolates transitions, and run-length analysis
  bounds the remaining block noise (20-25%), reported rather than hidden.
- Each listing's listed nightly price (listings.csv) is real and serves
  as the price signal throughout. What the snapshots lack is daily price
  history: the calendar price columns are empty or absent in every
  available snapshot (verified across 35.4M rows), which rules out
  behavioral cloning from historical price series.
- Review-based day labels are impossible in principle here: reviews end
  at the scrape date and the calendar starts there, so the two never
  overlap. Reviews serve as an activity filter instead.
- Two seasonal snapshot pairs (May→June 2026, Sept→Oct 2025) are used
  because within a single pair, booking lead time and stay date are
  perfectly collinear; a second season separates them.

Final dataset: 7.16M labelled nights from 29,984 active listings (of
92,638 raw), booking rate 13.7%, chronological 70/15/15 split.

### Demand model (pillar 1)

An MLP learns P(booking | price, context) over 20 features; logistic
regression provides the baseline and an interpretable price-elasticity
check (standardized price coefficient −0.163).

- The MLP is primary despite LightGBM scoring higher (PR-AUC 0.267 vs
  0.156, both well above the 0.048 prevalence): trees produce a
  piecewise-constant price response, which is a poor revenue-optimization
  surface, and the smooth MLP is what the simulation differentiates
  against. LightGBM is kept as a diagnostic ceiling and the gap reported.
- Training uses unweighted BCE rather than class weighting: the model's
  probabilities drive the simulator, so calibration outranks recall.
  Result: ECE 0.016 with no recalibration needed, which was a hard gate.
- Cross-sectional prices understate the causal price response (quality
  confounding; the raw revenue curve peaked at the price cap). The
  simulation applies P(r)·exp(β(r−1)) with β = −0.425, calibrated so the
  revenue peak lands at ratio ≈ 1.1. A power-law correction cannot
  produce an interior revenue peak here; the exponential form matches
  logit demand, where elasticity grows with price, as in Calvano et al.

### Market simulation (pillar 2)

A PettingZoo `ParallelEnv` in which each agent is a real Westminster
listing. Observations: own price vs market median, recent occupancy,
seasonality encoding, and rivals' previous prices. Actions: price change
in {−10%, −5%, 0, +5%, +10%}. Reward: booked revenue normalized by the
listing's base price. Episodes are 365 days; prices clipped to 0.5-2.5×
the cluster median with boundary hits logged.

- PettingZoo over plain Gymnasium because simultaneous price-setting is
  inherently multi-agent; over RLlib because a custom loop keeps one
  small, debuggable code path shared by local machines and Colab.
- Rivals' prices enter observations lagged one step: with simultaneous
  actions, observing current rival prices is circular. The lag resolves
  the simultaneity paradox at the cost of one step of staleness.
- Rewards are normalized per listing so heterogeneous base prices produce
  comparable Q-value scales across agents.

### Agents

- **D3QN** (Double + Dueling DQN, replay buffer, target network): Double
  counters Q-overestimation, which independent-learning nonstationarity
  amplifies; Dueling separates state value from action advantages, useful
  when neighboring price actions have near-identical values. Both are
  low-cost additions to the specified DQN.
- **Behavioral cloning warm start** imitates the anchor policy (steer
  toward the listed base price), the observable component of real host
  behavior, since cloning historical price changes is impossible without
  price history. Without any warm start, early episodes are wasted on
  uniformly random pricing.
- **PPO** is a minimal custom implementation (~200 lines, clipped
  surrogate + GAE) rather than Stable-Baselines3: SB3 does not natively
  drive PettingZoo multi-agent loops, and wrapper stacks would split the
  code path the other agents share.
- **Tabular Q-learning** with a binned state space bridges the results to
  the Calvano literature; it is not viable as a primary method because
  the realistic state space defeats tabular enumeration.

### Measuring collusion

Nash prices are computed by iterated best response on the actual
simulated market until a fixed point; a single pass of best responses is
not an equilibrium and understates Nash profits. The monopoly point
maximizes joint profit by coordinate ascent from multiple starts. Both
use the same demand model and reward normalization as training, so Δ is
internally consistent. Analytic bounds (as in logit-demand papers) are
unavailable because demand here is a learned model with no closed form.
Every experiment runs 20 seeds and reports distributions: independent
learning is nonstationary by construction, and single runs are anecdotes.

## Repository layout

```
configs/            data, demand, env and per-experiment YAML (+ smoke variants)
scripts/            download_data, preprocess, train_demand, train_rl,
                    compute_bounds, evaluate, make_figures, run_local_grid.sh
src/airbnb_marl/    data, features, demand, env, agents, training, analysis
notebooks/          01 EDA, 02 demand model, 03 RL training (Colab), 04 results
tests/              pytest suite (env API, labels, schema, bounds sanity)
results/            demand artifact and per-seed run outputs (shared via git),
                    evaluation tables; figures are regenerated, not committed
docs/               report_notes.md (all deviations, findings), presentation.html
```

## Reproduction

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # skip on GPU/Colab
pip install -r requirements.txt && pip install -e .
pytest

python3 scripts/download_data.py      # Inside Airbnb snapshots (~130 MB)
python3 scripts/preprocess.py         # labels, clusters, features -> parquet
python3 scripts/train_demand.py       # or reuse the committed artifact
python3 scripts/compute_bounds.py --config configs/experiments/E1_dqn_n2.yaml
python3 scripts/train_rl.py --config configs/experiments/E1_dqn_n2.yaml --seed 42
python3 scripts/evaluate.py && python3 scripts/make_figures.py
```

Full grids run via `bash scripts/run_local_grid.sh <config> [name] [email]
[seed_from] [seed_to]` (bootstraps everything, resumes interrupted seeds,
pushes completed ones) or the Colab notebook `03_rl_training.ipynb`.
Smoke configs in `configs/smoke/` verify the pipeline in minutes.

## Data and license

All data comes from [Inside Airbnb](https://insideairbnb.com/get-the-data/)
(CC BY 4.0), London snapshots 2026-06-19, 2026-05-24, 2025-10-17 and
2025-09-14. Raw and processed data are gitignored and rebuilt by script.

## References

Calvano, Calzolari, Denicolò, Pastorello (2020), *AI, Algorithms and
Collusion*, AER 110(10). Deng, Schiffer, Bichler (2024), arXiv:2406.02437.
Tinoco, Abeliuk, Ruiz-del-Solar (2025), arXiv:2504.05335. Mnih et al.
(2015), Nature 518. Sutton & Barto (2018), *Reinforcement Learning*.
