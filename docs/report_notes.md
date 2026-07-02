# Report notes: deviations from the specification and key numbers

Working notes for the written report. Each deviation states what changed,
why, and where it lives in the code.

## Data

- Snapshots used: London 2026-06-19 (primary) plus 2026-05-24, and the
  autumn pair 2025-09-14 plus 2025-10-17. The spec named 2025-09-14 alone;
  the spec itself allows snapshot substitution. All dataset counts in the
  report must use the new numbers below.
- No calendar price data exists in any available snapshot. The 2025-09-14
  archive has price columns that are empty in all 35,357,974 rows, newer
  snapshots dropped the columns entirely. Consequences: behavioral cloning
  from historical daily price changes is impossible, and the price signal
  for the demand model is purely cross sectional (the t0 listing price).
- Booking label: two snapshot diff. A night open at t0 and unavailable at
  t1 counts as booked during the window. Nights closed at t0 are excluded.
  Review anchoring at day level is impossible (reviews end at the scrape,
  the calendar starts there), so reviews serve as an activity filter and a
  volume sanity check only.
- Two pairs from different seasons separate stay date seasonality from
  booking lead time (perfectly collinear within one pair). log_lead is a
  model feature, fixed at 30 days in the simulation.
- Noise controls: paused listings (whole calendar flipped) removed, about
  1 percent of listings per pair; run length analysis shows a median booked
  run of 3 nights while 20 to 25 percent of booked nights sit in runs over
  28 nights and are likely host blocks. That share is the honest noise
  bound quoted in the report.
- Dataset: 7,155,606 labelled nights, overall booking rate 13.7 percent
  (13.2 summer, 14.1 autumn), 29,984 active primary listings from 92,638
  raw. Chronological 70/15/15 split within each pair.

## Demand model (pillar 1)

- Metrics on the untouched test split (prevalence 0.0475): MLP log loss
  0.1795, PR-AUC 0.1564 (3.3x prevalence), Brier 0.0437, ECE 0.0156 with no
  recalibration needed. LightGBM diagnostic ceiling PR-AUC 0.2671; the gap
  is reported, trees beat small MLPs on tabular data and the MLP remains
  the primary model per the spec.
- LR price_ratio coefficient -0.1632 (standardized log odds), the price
  elasticity sanity check.
- Trained with unweighted BCE instead of balanced class weights: the
  probabilities drive the simulation, so calibration outranks recall
  (deviation from the reference guides).
- Structural elasticity correction: cross sectional prices understate the
  causal price response (quality confounding; the raw revenue curve peaked
  at the 2.5x price bound). The simulation probability is
  P(r) * exp(beta * (r - 1)) with beta = -0.425, calibrated so the revenue
  peak lands at ratio 1.1 (hosts pricing near their own revenue optimum).
  A power law correction cannot produce an interior peak here; the exp form
  matches logit demand where elasticity grows with price, as in Calvano et
  al. Three gates guard the artifact: PR-AUC over prevalence, price
  monotonicity, interior revenue peak.

## Simulation (pillar 2)

- PettingZoo ParallelEnv, simultaneous actions; observations use rival
  prices from the previous step, so the simultaneity circularity cannot
  occur. Rewards are booked revenue normalized by the listing base price.
  Price bounds 0.5x to 2.5x of the cluster median, boundary hits logged.
- Behavioral cloning target is the anchor policy (steer back toward the
  base price), the observable part of real host behavior, instead of the
  impossible historical price changes.
- The simulated Westminster market is real: the densest compatible group
  of listings. The 4 agent market is a co-located block run by one
  professional host, which directly illustrates the multi listing host
  concentration the spec highlights.
- Nash bound via iterated best response to a fixed point (a single pass
  optimization is not an equilibrium); monopoly via coordinate ascent on
  the joint profit from several starts. Bounds for the 2 agent market:
  Nash profit 0.4214 per agent step (ratios 1.075, 1.55), monopoly 0.4957
  (ratios 2.5, 1.125, an asymmetric optimum, one listing sacrifices demand
  at the price cap while the other harvests it). Baselines sit below Nash:
  anchor 0.3153, random 0.2976, median seeker 0.2642.

## Open caveats for the discussion section

- The booking signal keeps some host block noise (see run lengths).
- The demand model is frozen during RL; market feedback on demand levels
  beyond the price ratio channel is not modelled.
- Occupancy in the bounds is fixed at historical levels; in the simulation
  it evolves with bookings.
- Independent Q learning is nonstationary by construction; results are
  reported over 20 seeds with distribution plots rather than single runs.
