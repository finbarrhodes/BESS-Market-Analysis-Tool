# Methodology & Data Sources

This page documents the modelling approach, assumptions, and data sources used by the
forecasting and dispatch model.

```js
const coverage = await FileAttachment("data/coverage.json").json();
const manifest = await FileAttachment("data/manifest.json").json();
const p = manifest.ml_mpc.params;
const fmt = (d) => new Date(d).toLocaleDateString("en-GB", {year: "numeric", month: "short"});
```

## Two-stage participation model

The backtester uses a two-stage model to separate FR availability revenue from energy
arbitrage revenue without double-counting the same physical capacity.

**Stage 1 — day-ahead capacity allocation.** For each day D, the model decides how many MW
to commit to FR services versus hold back for arbitrage. Two signals are compared using
only information available at the end of day D-1:

- **FR value per MW** — the confirmed clearing price for day D from the EAC day-ahead
  auction (which clears on D-1), summed across selected services and EFA blocks. No
  forecasting needed; this price is already known.
- **Shadow arbitrage value per MW** — a per-unit estimate of net arbitrage profit for day D,
  derived from the same price forecast used for dispatch:
  `(avg_discharge − avg_charge / η − cycling_cost) × duration_h`.

Capacity is allocated proportionally, `fr_fraction = fr_value / (fr_value + arb_value)`, so
more MW flows toward whichever stream looks more attractive that day — without
all-or-nothing switching.

**Stage 2 — intraday dispatch.** Within the allocated arbitrage MW, the selected strategy
schedules charge/discharge against forecast prices and realises revenue against actual
day-D prices. The same price signal drives both stages.

The MPC engine tracks SoC continuously at 30-minute resolution, enforcing the FR headroom
band `[10%, 90%]` as a hard constraint at every step of the planning horizon. The remaining
simplification is that Stage 1 allocation is a daily heuristic rather than being jointly
co-optimised with the intraday LP — see [known limitations](#known-limitations).

## Dispatch strategies

Intraday dispatch is driven by a **rolling Model Predictive Control (MPC) linear programme**,
re-solved at every 30-minute settlement period. At each period *t* the LP plans over a
${p.horizon}-period (${p.horizon / 2}-hour) horizon, returns only the first period's
decision, then re-solves — a receding-horizon approach reflecting the real constraint that
dispatch must be committed before future prices are known.

**LP formulation.** Decision variables are charge power *p_chg[t]*, discharge power
*p_dis[t]*, and state of charge *SoC[t+1]* over the horizon. The objective maximises net
arbitrage revenue minus cycling degradation cost:

```
maximise  Σ price[t] × (p_dis[t] − p_chg[t]) × 0.5h  −  cycling_cost × Σ p_dis[t] × 0.5h
```

subject to:

- SoC state equation with round-trip efficiency applied on the charge side
- FR feasibility band `[10%, 90%]` enforced as a **hard constraint** at all SoC points,
  forcing the battery to pre-condition SoC for upcoming FR delivery obligations
- Power bounded to the residual MW available for arbitrage after FR commitment

Mutual exclusion of simultaneous charge and discharge is handled by LP relaxation: because
the objective penalises cycling, simultaneous charge and discharge is never optimal at a
positive spread, so no binary variables are required. Solved with the **CLARABEL**
interior-point solver bundled with cvxpy
([Diamond & Boyd, 2016](https://www.jmlr.org/papers/v17/15-408.html)).

**Three price signals are benchmarked** — all run the identical dispatch engine; only the
forecast fed to the LP differs.

| Strategy | Price signal fed to LP | What it represents |
|---|---|---|
| **Perfect Foresight** | Actual day-D wholesale prices | Theoretical ceiling — needs advance knowledge of the future |
| **Naive (D-1 prices)** | Yesterday's 48 half-hourly prices | Zero-skill floor; any real model must beat this |
| **ML Model** | Random Forest forecast for day D | Realistic best case using features available at end of D-1 |

Dispatch decisions execute unconditionally at actual prices. Per-period revenue can be
negative when forecast error causes an unfavourable trade — that is the realistic
operational outcome and is intentional.

The **foresight ratio** summarises how much of the theoretical ceiling each strategy
achieves. For LP-based joint co-optimisation of arbitrage and frequency response in GB, see
[Swierczynski et al. (2021)](https://doi.org/10.3390/en14248365).

## Ancillary service availability revenue

- Revenue = `clearing_price (£/MW/h) × MW committed to FR × 4 hours per EFA block`
- Services of different response speeds (DC, DR, DM) can be stacked on the same physical MW
  in the GB market — each earns a separate availability payment.
- **High and Low name the frequency excursion, not the battery's power direction.** A Low
  service (DCL, DRL, DML) answers a *low*-frequency event by injecting power — the battery
  discharges. A High service (DCH, DRH, DMH) answers a *high*-frequency event by absorbing
  power — the battery charges.
- High (charge) and Low (discharge) services are modelled as independent and simultaneous,
  assuming sufficient SoC headroom to respond in both directions.
- Clearing prices from the NESO Data Portal (legacy DC/DR/DM auctions Sep 2021 – Nov 2023,
  EAC service Nov 2023 – present). NESO publishes EAC results as one resource per fiscal
  year plus a live current-year feed, rotating the live feed into a new archive each April;
  collection stitches these segments together, de-duplicating the one-day overlap where
  adjacent segments meet.
- Ancillary revenue is identical across all three dispatch strategies — it does not depend
  on price forecasting.

## Wholesale energy arbitrage revenue

- Computed period-by-period as the LP dispatches:
  `revenue[t] = actual_price[t] × (e_dis[t] − e_chg[t])`, summed across all 48 settlement
  periods in the day.
- Power in each period is bounded to the residual MW available for arbitrage (total power
  minus MW committed to FR for that EFA block). Round-trip efficiency
  (${(p.efficiency_rt * 100).toFixed(0)}%) is applied to the charge side of the SoC state
  equation.
- The cycling wear cost (£${p.cycling_cost_per_mwh}/MWh discharged) is deducted each period
  and enters the LP objective, so the optimiser naturally avoids unprofitable cycles.
- **Price reference: APXMIDP market index** (APX Power UK) from Elexon Insights. This is the
  actual GB spot settlement reference, giving a materially more realistic daily spread than
  the imbalance settlement price (SSP), which can reach extreme negative values during
  high-renewable periods and would otherwise inflate arbitrage revenue.

## Negative clearing prices

GB frequency response auctions clear below zero more often than is widely appreciated:
**13.5% of auction records in this dataset (8,134 of 60,054) have a negative clearing
price**, concentrated in DR High (5,205 records) and DM High (2,816). As the storage fleet
has grown, procurement volumes have been outpaced and the High-side services in particular
have tipped into oversupply.

**These are included in revenue.** Negative prices are a real feature of a maturing
flexibility market, and excluding them overstates FR income — by roughly 12% of total
modelled revenue on this dataset. Earlier versions floored clearing prices at zero, which
silently removed those records; the floor is now opt-in rather than a default.

**Capacity allocation treats them differently, and deliberately so.** The Stage 1 allocator
splits each block's capacity in proportion to the value of each stream, and capacity cannot
rationally be allocated *toward* negative expected value — so the FR signal used for the
split is clamped at zero. Two consequences:

- A block whose services net out negative receives **no** FR commitment; the capacity is
  released to arbitrage. It earns nothing from FR either way, and committing it would bind
  the dispatch LP to the FR SoC band for no return.
- A block that nets out positive but contains a negative leg (5,178 of 10,773 blocks) **is**
  committed, and the negative leg is netted into revenue — modelling an operator bidding a
  service stack that is profitable overall while carrying one loss-making component, rather
  than one that cherry-picks each leg after the fact.

Without the clamp the proportional split is undefined: a negative numerator produces
negative committed MW, which propagates into the LP's power bounds, and the denominator can
cross zero and make the fraction unbounded.

## Why DR High clears negative

DR High is the clearest example of why negative prices had to be included rather than
floored. Taken alone it looks like a loss-making service; in context it is not.

| Year | DRH mean | DRL mean | DRH negative | DR pair (H+L) |
|---|---|---|---|---|
| 2022 | +£11.53 | +£13.00 | 0% | +£24.52 |
| 2023 | +£1.10 | +£10.66 | 14% | +£11.75 |
| 2024 | −£4.80 | +£8.18 | 88% | +£3.38 |
| 2025 | −£2.42 | +£13.21 | 77% | +£10.79 |
| 2026 | −£8.31 | +£16.41 | 92% | +£8.10 |

*All figures £/MW/h.*

DRH has cleared negative in the large majority of blocks since 2024, reaching 92% in 2026 —
while DRL has strengthened over the same period. **The pair, however, remains reliably
positive.** Adding the two legs gives +£8.10/MW/h in 2026, and fewer than 11% of blocks
price negative as a pair.

Two pieces of evidence indicate the market prices DR as a package rather than as two
independent services:

- **Cleared volumes are matched.** DRH and DRL clear within 5% of each other on 77% of
  blocks since 2025 (median difference 20 MW). The equivalent figures for DM and DC are 52%
  and 49% — DR is procured far more symmetrically than its neighbours.
- **The two legs are negatively correlated** (ρ = −0.21). When one rises the other falls, so
  the split moves while the sum stays comparatively stable — the signature of a package
  price being divided between two legs rather than two prices set independently.

The mechanism is the sustained delivery requirement described above. DR must be held for a
full 60 minutes, against 15 for DC and 30 for DM, so a provider holding a DR position must
keep state of charge near the midpoint to honour either direction for the whole window.
Committing to one leg effectively commits the asset's SoC posture to both, and the market
clears the resulting package asymmetrically.

**What this means for reading the revenue breakdown.** A negative DR High figure is not a
service an operator should drop — dropping it means not participating in DR at all. The
asymmetry reflects where charge and discharge headroom are scarce, not the profitability of
one leg in isolation. The service selector on the dispatch page allows DRH and DRL to be
toggled independently, which is useful for attribution but does not correspond to a
strategy an operator could actually run.

The correlation with renewable output runs the way the mechanism predicts. DRH prices are
*positively* correlated with the daily renewable share (ρ = +0.43) while DRL is weakly
negative (ρ = −0.18). DRH is the charge leg — the service that answers an over-frequency
excursion — and heavy renewable output is precisely what pushes the system toward surplus
and over-frequency, so the charge leg gains value on windy days even while clearing from a
negative base. Demand for the discharge leg does not track wind the same way, which is what
DRL's mild negative correlation reflects.

## Availability factor

- Applied as a uniform multiplier to all revenue streams and cycling costs.
- Models periods where the asset is unavailable through planned maintenance, unplanned
  faults, grid curtailment, or service delivery failures.
- The default of ${(p.availability_factor * 100).toFixed(0)}% reflects the minimum
  availability threshold mandated in NESO's Dynamic Containment and Enduring Auction
  Capability service specifications, and is consistent with observed GB fleet performance —
  Modo Energy's *GB Battery Storage Report* (2024) reports median fleet availability of
  95–97% across contracted windows.

## Cycling wear cost and battery degradation

- Applied to arbitrage trades only: `cycling wear cost (£/MWh) × MWh discharged per trade`.
- Ancillary service cycling (energy delivered during frequency events) is not separately
  modelled — it is minor relative to availability payments and is typically compensated via
  the service contract.
- *Why cycling matters beyond cost:* lithium-ion cells degrade through two primary
  mechanisms that accelerate with use — SEI layer growth, which irreversibly consumes
  cyclable lithium, and lithium plating at the anode, which increases with deeper discharge
  and higher charge rates. Each MWh cycled consumes a small fraction of finite cycle life.
  The cycling wear cost is a financial proxy for that physical degradation: aggressive
  dispatch earns more in the short run but consumes cycle life faster, reducing useful life
  and residual value. For rigorous treatments of cycle-based degradation cost, see
  [Xu et al. (2018)](https://arxiv.org/abs/1703.07968) and
  [Lee & Kim (2022)](https://doi.org/10.1016/j.ijepes.2021.107795).

## ML price forecast model

A **Random Forest regressor** predicts the 48 half-hourly APXMIDP prices for day D using
features available at the end of day D-1.

*Why Random Forest?* The feature set is tabular (lagged prices, generation-mix ratios,
temporal encodings) rather than sequential; trees need no feature scaling, are robust at
this data size, and give interpretable importances. This is consistent with the electricity
price forecasting literature, which finds tree-based methods competitive against deep
learning on short-horizon day-ahead tasks
([Lago et al., 2021](https://doi.org/10.1016/j.apenergy.2021.116983);
[Weron, 2014](https://doi.org/10.1016/j.ijforecast.2014.08.008)).

**Features (all available at end of day D-1):**

- Same-period lagged prices: 1, 2, 7 and 14 days prior
- Previous-day price statistics: mean, standard deviation, max, min across all 48 periods
- Generation mix (daily, from D-1): total generation, renewable and fossil fractions, and
  per-fuel breakdown
- Cyclical temporal encodings: settlement period, day-of-week and day-of-year as sin/cos
  pairs to preserve circularity (period 48 and period 1 are adjacent)
- Weekend and UK bank holiday flags
- **GB BESS fleet capacity** (`bess_fleet_mw`, monthly MW) from the DESNZ Renewable Energy
  Planning Database — captures the structural shift as a growing fleet competes for the
  same arbitrage spreads.
- **BESS spread suppression** (`bess_fleet_mw / gen_total`) — penetration as a fraction of
  total system generation, encoding the mechanism directly: as penetration grows, batteries
  charge cheap and discharge expensive, flattening the merit order and compressing spreads.

<div class="note">
<b>Reading the REPD fleet series.</b> REPD is a <i>planning</i> database published
quarterly, with two consequences. Recent months are revised upward as projects are
confirmed operational — the Q4 2025 extract put March 2026 at 3.4 GW, while the Q2 2026
extract puts the same month at 5.0 GW, so figures for the most recent year are
provisional. And the extract always lags the half-hourly price data, so trailing months
are projected from a 12-month linear trend anchored at the last measured value, keeping
the cumulative series monotonic. Projected months are flagged
<code>is_extrapolated</code>: currently
${coverage.bess_fleet.n_extrapolated} of ${coverage.bess_fleet.rows} months
(measured through ${fmt(coverage.bess_fleet.measured_end)}, projected to
${fmt(coverage.bess_fleet.end)}). Months before the first REPD entry are zero; months
after the last carry the most recent measured capacity forward rather than dropping to
zero.
</div>

**Train/test split.** A strict temporal split: training ends before **${p.test_start}**, so
the model never sees future prices. Training uses an expanding window; the held-out test
period runs from ${p.test_start} to the end of the data. The split date is held fixed as
data is extended, so each refresh adds to the out-of-sample period rather than to training —
and the test window stays longer than a full year, so seasonal performance can be assessed
across a complete annual cycle.

**Known limitations.** Tree-based models cannot extrapolate beyond price ranges seen in
training; electricity price forecasting is inherently noisy; and the model improves dispatch
quality on average without eliminating error on individual days. Current metrics and feature
importances are on the [Forecasting & Dispatch](./backtester) page.

## Known limitations

**Not yet modelled**

- Intraday / day-ahead market trading (APXMIDP used as a proxy; DA auction data not integrated)
- Balancing Mechanism direct trading
- Real-time dispatch constraints or grid connection limits

**Approximations in the current MPC LP**

- *Rolling horizon is not globally optimal.* A single LP over the full backtest would yield
  more revenue in theory, but the rolling approach reflects the real constraint that
  dispatch must be committed before future prices are known.
- *LP relaxation of charge/discharge mutual exclusion.* No binary variables prohibit
  simultaneous charge and discharge; because the objective penalises cycling, this is never
  optimal at a positive spread, so it is not binding in practice.
- *Stage 1 allocation is a daily heuristic.* A fully joint formulation would co-optimise
  allocation and intraday dispatch in a single LP/MIP — see
  [Swierczynski et al. (2021)](https://doi.org/10.3390/en14248365) and
  [Bai et al. (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0306261924015149).

**Battery degradation (not yet modelled)**

A real asset degrades through calendar ageing (capacity fade at rest) and cycle ageing
(accelerated by depth of discharge, C-rate and temperature). A more complete model would
track state-of-health across the backtest, apply a degradation-aware dispatch policy trading
short-term revenue against cycle-life consumption, and incorporate chemistry-specific
degradation curves (NMC, LFP), which differ materially. The cycling wear cost is a
simplified financial proxy and does not capture the compounding, path-dependent nature of
real degradation.

## Data sources

```js
const rows = [
  ["Frequency response auction results (DC/DR/DM)", "NESO Data Portal", coverage.auctions],
  ["APXMIDP market index price", "Elexon Insights Solution API", coverage.market_index],
  ["System buy/sell prices (SBP/SSP)", "Elexon Insights Solution API", coverage.system_prices],
  ["Generation by fuel type (daily)", "Elexon Insights Solution API", coverage.generation],
  ["GB BESS fleet capacity (monthly)", "DESNZ REPD", coverage.bess_fleet],
].map(([dataset, source, c]) => ({
  Dataset: dataset, Source: source,
  Coverage: `${fmt(c.start)} – ${fmt(c.end)}`,
  Records: d3.format(",")(c.rows),
}));

display(Inputs.table(rows, {rows: 6, width: {Dataset: 300, Source: 210}}));
```

Coverage is read from the processed datasets at build time rather than hardcoded, so this
table cannot drift out of date when the pipeline is re-run. Both NESO and Elexon APIs are
fully public and require no key.

<b>Overlapping pulls resolve to the most recent one.</b> Data is collected in date-ranged
batches that overlap, and the trailing days of any batch are provisional — Elexon moves
system prices through several settlement runs, and NESO revises embedded solar and wind
after the fact. Where two batches cover the same settlement period and disagree, the batch
collected later is kept, so a settled value always displaces the estimate it replaces.
[NESO Data Portal](https://www.neso.energy/data-portal) ·
[Elexon Insights](https://developer.data.elexon.co.uk/) ·
[DESNZ REPD](https://www.gov.uk/government/publications/renewable-energy-planning-database-monthly-extract)

## Literature & references

**Electricity price forecasting**

- Lago, J., Marcjasz, G., De Schutter, B., & Weron, R. (2021). Forecasting day-ahead
  electricity prices: A review of state-of-the-art algorithms, best practices and an
  open-access benchmark. *Applied Energy*, 293, 116983.
  [doi:10.1016/j.apenergy.2021.116983](https://doi.org/10.1016/j.apenergy.2021.116983)
- Weron, R. (2014). Electricity price forecasting: A review of the fundamental and
  econometric approaches. *International Journal of Forecasting*, 30(4), 1030–1081.
  [doi:10.1016/j.ijforecast.2014.08.008](https://doi.org/10.1016/j.ijforecast.2014.08.008)

**MPC dispatch engine & LP solver**

- Diamond, S., & Boyd, S. (2016). CVXPY: A Python-Embedded Modeling Language for Convex
  Optimization. *JMLR*, 17(83), 1–5.
  [jmlr.org/papers/v17/15-408](https://www.jmlr.org/papers/v17/15-408.html)
- Goulart, P., & Chen, Y. (2024). Clarabel: An interior-point solver for conic programs with
  quadratic objectives. *IEEE TAC*.
  [doi:10.1109/TAC.2024.3457633](https://doi.org/10.1109/TAC.2024.3457633)

**BESS dispatch optimisation & co-optimisation**

- Swierczynski, M., et al. (2021). Co-Optimizing Battery Storage for Energy Arbitrage and
  Frequency Regulation in the GB Market. *Energies*, 14(24), 8365.
  [doi:10.3390/en14248365](https://doi.org/10.3390/en14248365)
- Bai, X., et al. (2024). Smart optimization in battery energy storage systems: An overview.
  [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0306261924015149)
- Lee, J.-O., & Kim, Y.-S. (2022). Novel battery degradation cost formulation for optimal
  scheduling of battery energy storage systems. *IJEPES*, 137, 107795.
  [doi:10.1016/j.ijepes.2021.107795](https://doi.org/10.1016/j.ijepes.2021.107795)

**Battery degradation modelling**

- Xu, B., et al. (2018). Modeling of lithium-ion battery degradation for cell life
  assessment. *IEEE Transactions on Smart Grid*, 9(2), 1131–1140.
  [arXiv:1703.07968](https://arxiv.org/abs/1703.07968)
- Reniers, J. M., Mulder, G., & Howey, D. A. (2021). Economic MPC of Li-ion battery cyclic
  aging via online rainflow analysis. *Journal of Energy Storage*.
  [doi:10.1002/est2.228](https://doi.org/10.1002/est2.228)

**GB BESS market context**

- Modo Energy. (2024). *GB Battery Storage Report*.
  [modoenergy.com](https://modoenergy.com/research/future-of-battery-energy-storage-buildout-in-great-britain)
- Timera Energy. (2023). Battery investors confront revenue shift in 2023.
  [timera-energy.com](https://timera-energy.com/blog/battery-investors-confront-revenue-shift-in-2023/)
