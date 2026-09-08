Detecting Unfair Airline Pricing at Scale

A regulatory surveillance framework analysing 76 million U.S. flight fares with XGBoost, Isolation Forest, SQL, and market-concentration analysis, built entirely in Snowflake.



Team project (Hult MBAN, Business Intelligence). I co-developed the seven-model framework and personally built the in-Snowflake ML pipeline. Currently extending it solo with route-level booking-curve analysis (see Extension).



The headline findings

We analysed 76M Expedia ticket offers (235 U.S. routes, 16 airports, Apr–Oct 2022) from a regulator's perspective: which fares can't be explained by any legitimate factor, and do the carriers charging them face real competition?

$312M in unexplained pricing was identified across 7.2M scored flights, concentrated in specific route-carrier combinations. Three findings stood out because pricing ran backwards:







Finding



What we saw



Why it matters





Phantom scarcity



Flights with 3–5 seats left priced higher ($372) than flights with 1–2 seats left ($318)



Scarcity pricing that doesn't follow scarcity, consistent with algorithmic manipulation





The nonstop paradox



Nonstop (better product) at $251 vs one-stop (worse product) at $373 on the same corridors



72% of the market exposed to interline pricing with no competitive discipline





Refundability desert



1,332 refundable offers out of 82M (0.002%), at a 279% premium when shown



Consumers can't compare what they're never shown

Concrete example a regulator can act on: OAK→CLT at $649 (1.91× the fair-price estimate) on a route where market concentration (HHI 3,218) means consumers have nowhere else to go.

How the framework works

Being expensive isn't a regulatory concern; being expensive without justification on a route without competition is. The framework operationalises that in three steps:





What should this flight cost? An XGBoost fair-price model (R² = 0.76) predicts each fare from 33 features. Three "justification" models then test whether demand, supply, or product factors alone can explain the price. What no model can explain becomes the unexplained premium.



Is it structurally anomalous? An Isolation Forest catches internally inconsistent offers, and an SQL window-function percentile ranks each fare against its direct peers (same route, same cabin, median 6,855 peers).



Does the carrier have market power? Route-level HHI (the DOJ/FTC merger standard) both scores and amplifies the composite: the same suspicious price matters more on a near-monopoly route.

Signals combine into a weighted Unfair Pricing Score; flights reach Tier 1 (high-confidence unfair) only if the score, concentration, and peer-percentile gates all pass simultaneously. Result: 2,803 Tier 1 flags out of 7.2M, a reviewable queue (~50/day), not an alarm siren.

Validation: synthetic fraud injection (83.6% recall at Tier 3), placebo test on competitive routes (0.00% false Tier 1 rate), and temporal stability across the pilot window.

My contribution: the in-Snowflake ML pipeline

I built the anomaly-detection pipeline that runs end-to-end inside Snowflake; no data ever leaves the warehouse:





19 engineered features via Snowpark, including SQL window functions for route price benchmarks, carrier-route competition counts, and inventory scarcity buckets



XGBoost trained on Snowflake ML compute (500K-row sample, deterministic hash-based 80/20 split), with ordinal encoding handled in-database



Residual z-score anomaly flagging (±2.5σ) computed as Snowpark column operations



Only sub-100-row aggregates leave Snowflake for visualisation, the pattern a production system would use

Code: [src/snowflake_xgboost_pipeline.py](src/snowflake_xgboost_pipeline.py)

Extension: route-level booking curves (in progress)

Expert feedback on the original project flagged a gap: LEAD_TIME was used as a global model input but never analysed per route, yet routes have wildly different pricing curves (some lock prices for 45 days then spike 60%; some charge 125% last-minute premiums; some get cheaper toward departure).

I'm rebuilding a subset of the analysis locally (the dataset is public on Kaggle) to:





Map price-vs-lead-time curves for the top 50 routes



Cluster routes by curve shape and identify algorithmic threshold behaviour (day-14 / day-6 repricing)



Translate each curve type into a plain-English consumer guidance rule



Honest limitations





"Unfair" is operationally defined as unexplained by measurable factors: analytically strong, legally still debatable



Listed prices only (Expedia channel), not confirmed bookings; corporate fares invisible



Relative anomalies: industry-wide overpricing would not be flagged



Correlation, not causation. The framework finds patterns for investigators; it doesn't convict



Stack

Snowflake · Snowpark · Snowflake ML (XGBoost, OrdinalEncoder) · SQL window functions · Python · matplotlib · Streamlit dashboard



Data: Expedia flight offers, Apr–Oct 2022 (Kaggle: dilwong/flightprices). Course: DAT-8564 Business Intelligence, Hult International Business School.
