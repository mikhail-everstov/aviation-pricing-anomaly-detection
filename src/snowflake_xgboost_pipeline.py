# ============================================================================
# AVIATION PRICING ANOMALY DETECTION USING XGBOOST
# 100% IN SNOWFLAKE - DOT/FAA Regulatory Analysis - Team 5
# ============================================================================

from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F
from snowflake.snowpark import Window
from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.ml.modeling.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
session = get_active_session()

# Set database and schema context so Snowflake ML can create its internal temp tables.
# Without this, OrdinalEncoder.fit() fails with "Cannot perform CREATE TEMPTABLE.
# This session does not have a current database."
session.use_database("FLIGHTS_TEAM_5")
session.use_schema("PUBLIC")

Airplane_dataset = session.table("FLIGHTS_TEAM_5.PUBLIC.FLIGHT_PRICES_RAW")
print("=" * 80)
print("STEP 1: DATA LOADED")
print(f"Total records: {Airplane_dataset.count():,}")
print("=" * 80)

# ============================================================================
# STEP 2: FEATURE ENGINEERING (ALL IN SNOWFLAKE)
# ============================================================================
print("\nSTEP 2: ENGINEERING FEATURES IN SNOWFLAKE...")

features_df = Airplane_dataset.select(
F.col("LEGID"),
F.col("SEARCHDATE"),
F.col("FLIGHTDATE"),
F.col("TOTALFARE"),
F.col("BASEFARE"),

# FEATURE 1: LEAD TIME
(F.datediff('day', F.col("SEARCHDATE"), F.col("FLIGHTDATE"))).alias("LEAD_TIME"),
(F.when(F.col("SEARCHDATE") == F.col("FLIGHTDATE"), 1).otherwise(0)).alias("IS_SAME_DAY"),

# FEATURE 2: FLIGHT DAY OF WEEK
F.dayname(F.col("FLIGHTDATE")).alias("FLIGHT_DAY_OF_WEEK"),

# FEATURE 3: FLIGHT MONTH
F.month(F.col("FLIGHTDATE")).alias("FLIGHT_MONTH"),

# FEATURE 4: ROUTE VARIABLES
F.col("STARTINGAIRPORT").alias("ORIGIN_AIRPORT"),
F.col("DESTINATIONAIRPORT").alias("DESTINATION_AIRPORT"),
F.concat(F.col("STARTINGAIRPORT"), F.lit('-'), F.col("DESTINATIONAIRPORT")).alias("ROUTE"),

# FEATURE 5: AIRLINE
F.col("SEGMENTSAIRLINENAME").alias("AIRLINE_NAME"),
F.col("SEGMENTSAIRLINECODE").alias("AIRLINE_CODE"),

# FEATURE 6: SEAT INVENTORY
F.col("SEATSREMAINING").alias("SEATS_AVAILABLE"),
F.when(F.col("SEATSREMAINING") == 0, F.lit("Sold Out"))
.when(F.col("SEATSREMAINING").between(1, 3), F.lit("Critical"))
.when(F.col("SEATSREMAINING").between(4, 10), F.lit("Low"))
.when(F.col("SEATSREMAINING").between(11, 30), F.lit("Medium"))
.when(F.col("SEATSREMAINING") > 30, F.lit("High"))
.alias("INVENTORY_LEVEL"),

# FEATURE 7: FARE STRUCTURE (cast booleans to integers)
F.iff(F.col("ISBASICECONOMY"), F.lit(1), F.lit(0)).alias("IS_BASIC_ECONOMY"),
F.iff(F.col("ISREFUNDABLE"), F.lit(1), F.lit(0)).alias("IS_REFUNDABLE"),

# FEATURE 8: FLIGHT TYPE
F.iff(F.col("ISNONSTOP"), F.lit(1), F.lit(0)).alias("IS_NONSTOP"),

# FEATURE 9: DISTANCE
F.col("TOTALTRAVELDISTANCE").alias("FLIGHT_DISTANCE_MILES"),
F.when(F.col("TOTALTRAVELDISTANCE") <= 500, F.lit("Short"))
.when(F.col("TOTALTRAVELDISTANCE") <= 1000, F.lit("Medium"))
.when(F.col("TOTALTRAVELDISTANCE") <= 1500, F.lit("Long"))
.when(F.col("TOTALTRAVELDISTANCE") > 1500, F.lit("Very Long"))
.alias("DISTANCE_CATEGORY"),

# FEATURE 10: DEPARTURE TIME
F.hour(F.to_timestamp(F.split_part(F.col("SEGMENTSDEPARTURETIMERAW"), F.lit('||'), F.lit(1)))).alias("DEPARTURE_HOUR"),
F.when(F.hour(F.to_timestamp(F.split_part(F.col("SEGMENTSDEPARTURETIMERAW"), F.lit('||'), F.lit(1)))).between(0, 5), F.lit("Red-Eye"))
.when(F.hour(F.to_timestamp(F.split_part(F.col("SEGMENTSDEPARTURETIMERAW"), F.lit('||'), F.lit(1)))).between(6, 11), F.lit("Morning"))
.when(F.hour(F.to_timestamp(F.split_part(F.col("SEGMENTSDEPARTURETIMERAW"), F.lit('||'), F.lit(1)))).between(12, 17), F.lit("Afternoon"))
.when(F.hour(F.to_timestamp(F.split_part(F.col("SEGMENTSDEPARTURETIMERAW"), F.lit('||'), F.lit(1)))).between(18, 23), F.lit("Evening"))
.alias("DEPARTURE_TIME_PERIOD"),

# FEATURE 11: SEARCH DATE
F.dayname(F.col("SEARCHDATE")).alias("SEARCH_DAY_OF_WEEK"),
F.month(F.col("SEARCHDATE")).alias("SEARCH_MONTH"),

# FEATURE 12: ROUTE BENCHMARKS
# LIMITATION: these window aggregates are computed over the full dataset
# (train + test) and are derived from the target (TOTALFARE), which leaks
# target information into the features and makes test metrics optimistic.
# A production version would compute route benchmarks on training data only.
# Kept as-run for reproducibility; see README limitations.
F.avg(F.col("TOTALFARE")).over(
Window.partition_by(F.concat(F.col("STARTINGAIRPORT"), F.lit('-'), F.col("DESTINATIONAIRPORT")))
).alias("ROUTE_AVG_PRICE"),
F.stddev(F.col("TOTALFARE")).over(
Window.partition_by(F.concat(F.col("STARTINGAIRPORT"), F.lit('-'), F.col("DESTINATIONAIRPORT")))
).alias("ROUTE_PRICE_STDDEV"),

# FEATURE 13: AIRLINE-ROUTE COMPETITION
F.avg(F.col("TOTALFARE")).over(
Window.partition_by(
F.concat(F.col("STARTINGAIRPORT"), F.lit('-'), F.col("DESTINATIONAIRPORT")),
F.col("SEGMENTSAIRLINECODE")
)
).alias("AIRLINE_ROUTE_AVG_PRICE"),
F.count_distinct(F.col("SEGMENTSAIRLINECODE")).over(
Window.partition_by(F.concat(F.col("STARTINGAIRPORT"), F.lit('-'), F.col("DESTINATIONAIRPORT")))
).alias("ROUTE_COMPETITOR_COUNT"),

# FEATURE 14: LEAD TIME URGENCY
F.when(F.datediff('day', F.col("SEARCHDATE"), F.col("FLIGHTDATE")) <= 7, F.lit("Last Minute"))
.when(F.datediff('day', F.col("SEARCHDATE"), F.col("FLIGHTDATE")) <= 14, F.lit("Short Notice"))
.when(F.datediff('day', F.col("SEARCHDATE"), F.col("FLIGHTDATE")) <= 30, F.lit("Moderate"))
.when(F.datediff('day', F.col("SEARCHDATE"), F.col("FLIGHTDATE")) > 30, F.lit("Advance"))
.alias("LEAD_TIME_URGENCY"),

# FEATURE 15: AIRCRAFT CATEGORY
F.when(F.col("SEGMENTSEQUIPMENTDESCRIPTION").like('%CRJ%'), F.lit("Regional Jet"))
.when(F.col("SEGMENTSEQUIPMENTDESCRIPTION").like('%ERJ%'), F.lit("Regional Jet"))
.when(F.col("SEGMENTSEQUIPMENTDESCRIPTION").like('%737%'), F.lit("Narrow-body"))
.when(F.col("SEGMENTSEQUIPMENTDESCRIPTION").like('%A320%'), F.lit("Narrow-body"))
.when(F.col("SEGMENTSEQUIPMENTDESCRIPTION").like('%767%'), F.lit("Wide-body"))
.when(F.col("SEGMENTSEQUIPMENTDESCRIPTION").like('%787%'), F.lit("Wide-body"))
.when(F.col("SEGMENTSEQUIPMENTDESCRIPTION").like('%A330%'), F.lit("Wide-body"))
.otherwise(F.lit("Other"))
.alias("AIRCRAFT_CATEGORY"),

# FEATURE 16: AIRPORT HUB STATUS
F.when(F.col("STARTINGAIRPORT").isin(['ATL', 'ORD', 'DFW', 'LAX', 'DEN']), 1).otherwise(0).alias("IS_ORIGIN_MAJOR_HUB"),
F.when(F.col("DESTINATIONAIRPORT").isin(['ATL', 'ORD', 'DFW', 'LAX', 'DEN']), 1).otherwise(0).alias("IS_DESTINATION_MAJOR_HUB"),

# FEATURE 17: DAYS TO NEAREST HOLIDAY
F.when(
(F.month(F.col("FLIGHTDATE")) == 7) & (F.day(F.col("FLIGHTDATE")).between(1, 10)),
F.abs(F.day(F.col("FLIGHTDATE")) - 4)
).when(
(F.month(F.col("FLIGHTDATE")) == 5) & (F.day(F.col("FLIGHTDATE")).between(20, 31)),
F.abs(F.day(F.col("FLIGHTDATE")) - 27)
).when(
(F.month(F.col("FLIGHTDATE")) == 9) & (F.day(F.col("FLIGHTDATE")).between(1, 10)),
F.abs(F.day(F.col("FLIGHTDATE")) - 6)
).otherwise(365)
.alias("DAYS_TO_NEAREST_HOLIDAY"),

# FEATURE 18: DAY OF MONTH
F.day(F.col("FLIGHTDATE")).alias("FLIGHT_DAY_OF_MONTH"),
F.when(F.day(F.col("FLIGHTDATE")).between(1, 5), F.lit("Month Start"))
.when(F.day(F.col("FLIGHTDATE")).between(6, 24), F.lit("Mid Month"))
.when(F.day(F.col("FLIGHTDATE")) >= 25, F.lit("Month End"))
.alias("FLIGHT_DAY_OF_MONTH_PERIOD"),

# FEATURE 19: ROUTE SATURATION
F.count(F.col("LEGID")).over(
Window.partition_by(
F.concat(F.col("STARTINGAIRPORT"), F.lit('-'), F.col("DESTINATIONAIRPORT")),
F.col("FLIGHTDATE")
)
).alias("FLIGHTS_ON_ROUTE_TODAY"),
)

print("Features engineered!")

# ============================================================================
# STEP 3: SAMPLE + CLEAN + ENCODE (ALL IN SNOWFLAKE)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: SAMPLING & ENCODING IN SNOWFLAKE")
print("=" * 80)

# Sample 500K rows IN SNOWFLAKE (no download)
sample_df = features_df.filter(
F.col("TOTALFARE") > 0
).filter(
F.col("TOTALFARE") < 5000
).filter(
F.col("FLIGHT_DISTANCE_MILES").is_not_null()
).sample(n=500000)

print(f"Sample size: {sample_df.count():,} rows (stays in Snowflake)")

# Define column types
categorical_cols = [
'FLIGHT_DAY_OF_WEEK', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
'AIRLINE_CODE', 'INVENTORY_LEVEL', 'DISTANCE_CATEGORY',
'DEPARTURE_TIME_PERIOD', 'SEARCH_DAY_OF_WEEK', 'LEAD_TIME_URGENCY',
'AIRCRAFT_CATEGORY', 'FLIGHT_DAY_OF_MONTH_PERIOD'
]

encoded_cols = [f"{c}_ENC" for c in categorical_cols]

numeric_cols = [
'LEAD_TIME', 'IS_SAME_DAY', 'FLIGHT_MONTH', 'SEATS_AVAILABLE',
'FLIGHT_DISTANCE_MILES', 'DEPARTURE_HOUR', 'SEARCH_MONTH',
'ROUTE_AVG_PRICE', 'ROUTE_PRICE_STDDEV', 'AIRLINE_ROUTE_AVG_PRICE',
'ROUTE_COMPETITOR_COUNT', 'IS_ORIGIN_MAJOR_HUB', 'IS_DESTINATION_MAJOR_HUB',
'DAYS_TO_NEAREST_HOLIDAY', 'FLIGHT_DAY_OF_MONTH', 'FLIGHTS_ON_ROUTE_TODAY',
'IS_BASIC_ECONOMY', 'IS_REFUNDABLE', 'IS_NONSTOP'
]

# All features for the model
all_model_features = numeric_cols + encoded_cols

# Fill NULLs in categorical columns before encoding (OrdinalEncoder chokes on NULLs)
sample_df = sample_df.fillna("UNKNOWN", subset=categorical_cols)

# Encode categoricals using Snowpark ML (IN SNOWFLAKE)
# NOTE: Snowflake ML's OrdinalEncoder has NO fit_transform — must call fit() then transform()
print("Encoding categorical features in Snowflake...")
encoder = OrdinalEncoder(
input_cols=categorical_cols,
output_cols=encoded_cols,
handle_unknown="use_encoded_value",
unknown_value=-1,
)
encoder.fit(sample_df)
encoded_df = encoder.transform(sample_df)
print(f"Encoded {len(categorical_cols)} categorical features")

# ============================================================================
# STEP 4: TRAIN/TEST SPLIT (IN SNOWFLAKE)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: TRAIN/TEST SPLIT IN SNOWFLAKE")
print("=" * 80)

# Deterministic split using hash of LEGID (80/20)
split_df = encoded_df.with_column(
"IS_TRAIN",
F.when(F.abs(F.hash(F.col("LEGID"))) % 10 < 8, F.lit(1)).otherwise(F.lit(0))
)

train_df = split_df.filter(F.col("IS_TRAIN") == 1)
test_df = split_df.filter(F.col("IS_TRAIN") == 0)

train_count = train_df.count()
test_count = test_df.count()
print(f"Training set: {train_count:,} rows (in Snowflake)")
print(f"Test set: {test_count:,} rows (in Snowflake)")

# ============================================================================
# STEP 5: TRAIN XGBOOST (IN SNOWFLAKE)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: TRAINING XGBOOST MODEL IN SNOWFLAKE")
print("Model predicts FAIR PRICE based on all features")
print("Anomaly = actual price far from predicted fair price")
print("=" * 80)

model = XGBRegressor(
input_cols=all_model_features,
label_cols=["TOTALFARE"],
output_cols=["PREDICTED_FARE"],
n_estimators=300,
max_depth=7,
learning_rate=0.05,
subsample=0.8,
colsample_bytree=0.8,
random_state=42
)

print("Training XGBoost... (this runs on Snowflake compute)")
model.fit(train_df)
print("Model trained!")

# ============================================================================
# STEP 6: PREDICT + DETECT ANOMALIES (IN SNOWFLAKE)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: PREDICTING & DETECTING ANOMALIES IN SNOWFLAKE")
print("=" * 80)

# Predict on full sample (IN SNOWFLAKE)
predictions_df = model.predict(encoded_df)

# Calculate residuals IN SNOWFLAKE
results_df = predictions_df.with_columns(
["RESIDUAL", "RESIDUAL_PERCENT"],
[
F.col("TOTALFARE") - F.col("PREDICTED_FARE"),
((F.col("TOTALFARE") - F.col("PREDICTED_FARE")) / F.col("PREDICTED_FARE")) * 100
]
)

# Compute residual mean and stddev IN SNOWFLAKE
residual_stats = results_df.select(
F.avg(F.col("RESIDUAL")).alias("RESID_MEAN"),
F.stddev(F.col("RESIDUAL")).alias("RESID_STD")
).collect()

resid_mean = float(residual_stats[0]["RESID_MEAN"])
resid_std = float(residual_stats[0]["RESID_STD"])

print(f"Residual Mean: ${resid_mean:.2f}")
print(f"Residual Std: ${resid_std:.2f}")

# Add z-scores and anomaly flags IN SNOWFLAKE
anomaly_df = results_df.with_columns(
["RESIDUAL_ZSCORE", "IS_OVERPRICED", "IS_UNDERPRICED", "IS_ANOMALY"],
[
(F.col("RESIDUAL") - F.lit(resid_mean)) / F.lit(resid_std),
F.when((F.col("RESIDUAL") - F.lit(resid_mean)) / F.lit(resid_std) > 2.5, 1).otherwise(0),
F.when((F.col("RESIDUAL") - F.lit(resid_mean)) / F.lit(resid_std) < -2.5, 1).otherwise(0),
F.when(F.abs((F.col("RESIDUAL") - F.lit(resid_mean)) / F.lit(resid_std)) > 2.5, 1).otherwise(0),
]
)

# ============================================================================
# STEP 7: MODEL METRICS (COMPUTED IN SNOWFLAKE)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: MODEL PERFORMANCE (computed in Snowflake)")
print("=" * 80)

# Split predictions for train/test metrics
train_preds = model.predict(train_df)
test_preds = model.predict(test_df)

train_metrics = train_preds.select(
F.avg(F.pow(F.col("TOTALFARE") - F.col("PREDICTED_FARE"), 2)).alias("MSE"),
F.avg(F.abs(F.col("TOTALFARE") - F.col("PREDICTED_FARE"))).alias("MAE"),
F.corr(F.col("TOTALFARE"), F.col("PREDICTED_FARE")).alias("CORRELATION")
).collect()

test_metrics = test_preds.select(
F.avg(F.pow(F.col("TOTALFARE") - F.col("PREDICTED_FARE"), 2)).alias("MSE"),
F.avg(F.abs(F.col("TOTALFARE") - F.col("PREDICTED_FARE"))).alias("MAE"),
F.corr(F.col("TOTALFARE"), F.col("PREDICTED_FARE")).alias("CORRELATION")
).collect()

train_rmse = float(train_metrics[0]["MSE"]) ** 0.5
test_rmse = float(test_metrics[0]["MSE"]) ** 0.5
test_mae = float(test_metrics[0]["MAE"])
test_corr = float(test_metrics[0]["CORRELATION"])
# NOTE: this is the squared Pearson correlation, not true R-squared.
# For a regression model they differ; squared correlation is an optimistic
# approximation. Kept as-run for reproducibility; see README limitations.
r2_approx = test_corr ** 2

print(f"\n Train RMSE: ${train_rmse:.2f}")
print(f" Test RMSE: ${test_rmse:.2f}")
print(f" Test MAE: ${test_mae:.2f}")
print(f" Test R² (approx): {r2_approx:.4f}")
print(f" Explained: {r2_approx*100:.1f}% of price variation")

# Anomaly summary
anomaly_summary = anomaly_df.select(
F.count(F.col("LEGID")).alias("TOTAL"),
F.sum(F.col("IS_OVERPRICED")).alias("OVERPRICED"),
F.sum(F.col("IS_UNDERPRICED")).alias("UNDERPRICED"),
F.sum(F.col("IS_ANOMALY")).alias("TOTAL_ANOMALIES")
).collect()

total = int(anomaly_summary[0]["TOTAL"])
overpriced = int(anomaly_summary[0]["OVERPRICED"])
underpriced = int(anomaly_summary[0]["UNDERPRICED"])
total_anomalies = int(anomaly_summary[0]["TOTAL_ANOMALIES"])

print(f"\n ANOMALY DETECTION RESULTS:")
print(f" Total flights: {total:,}")
print(f" OVERPRICED: {overpriced:,} ({overpriced/total*100:.2f}%)")
print(f" UNDERPRICED: {underpriced:,} ({underpriced/total*100:.2f}%)")
print(f" Total anomalies: {total_anomalies:,} ({total_anomalies/total*100:.2f}%)")
print(f" Fair price: {total - total_anomalies:,} ({(total-total_anomalies)/total*100:.1f}%)")

# ============================================================================
# STEP 8: AGGREGATE IN SNOWFLAKE FOR CHARTS (tiny data for matplotlib)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: AGGREGATING IN SNOWFLAKE FOR VISUALIZATIONS")
print("(Only tiny summary tables leave Snowflake for plotting)")
print("=" * 80)

# 8a. Top 20 overpriced flights
top20 = anomaly_df.filter(F.col("IS_OVERPRICED") == 1).sort(F.col("RESIDUAL_ZSCORE").desc()).select(
F.col("LEGID"), F.col("ROUTE"), F.col("AIRLINE_CODE"),
F.col("TOTALFARE"), F.col("PREDICTED_FARE"), F.col("RESIDUAL"),
F.col("RESIDUAL_ZSCORE"), F.col("LEAD_TIME"), F.col("SEATS_AVAILABLE"),
F.col("INVENTORY_LEVEL")
).limit(20)
print("\nTop 20 Most Overpriced Flights:")
top20.show()

# 8b. Anomalies by airline (aggregated in Snowflake)
agg_airline = anomaly_df.filter(F.col("IS_OVERPRICED") == 1).group_by(F.col("AIRLINE_CODE")).agg(
F.count(F.col("LEGID")).alias("ANOMALY_COUNT"),
F.avg(F.col("RESIDUAL")).alias("AVG_OVERPRICING"),
F.avg(F.col("RESIDUAL_ZSCORE")).alias("AVG_ZSCORE"),
F.avg(F.col("TOTALFARE")).alias("AVG_FARE")
).sort(F.col("ANOMALY_COUNT").desc())

# 8c. Anomaly RATE by airline
agg_airline_total = anomaly_df.group_by(F.col("AIRLINE_CODE")).agg(
F.count(F.col("LEGID")).alias("TOTAL_FLIGHTS"),
F.sum(F.col("IS_OVERPRICED")).alias("OVERPRICED_COUNT")
)
agg_airline_rate = agg_airline_total.with_column(
"ANOMALY_RATE",
F.col("OVERPRICED_COUNT") / F.col("TOTAL_FLIGHTS") * 100
).sort(F.col("ANOMALY_RATE").desc())

# 8d. Anomalies by route
agg_route = anomaly_df.filter(F.col("IS_OVERPRICED") == 1).group_by(F.col("ROUTE")).agg(
F.count(F.col("LEGID")).alias("ANOMALY_COUNT"),
F.avg(F.col("RESIDUAL")).alias("AVG_OVERPRICING"),
F.avg(F.col("RESIDUAL_ZSCORE")).alias("AVG_ZSCORE")
).sort(F.col("ANOMALY_COUNT").desc()).limit(20)

# 8e. Anomalies by inventory level
agg_inventory = anomaly_df.filter(F.col("IS_OVERPRICED") == 1).group_by(F.col("INVENTORY_LEVEL")).agg(
F.count(F.col("LEGID")).alias("ANOMALY_COUNT"),
F.avg(F.col("RESIDUAL")).alias("AVG_OVERPRICING"),
F.avg(F.col("RESIDUAL_ZSCORE")).alias("AVG_ZSCORE")
).sort(F.col("ANOMALY_COUNT").desc())

# 8f. Anomalies by lead time urgency
agg_urgency = anomaly_df.filter(F.col("IS_OVERPRICED") == 1).group_by(F.col("LEAD_TIME_URGENCY")).agg(
F.count(F.col("LEGID")).alias("ANOMALY_COUNT"),
F.avg(F.col("RESIDUAL")).alias("AVG_OVERPRICING"),
F.avg(F.col("TOTALFARE")).alias("AVG_FARE")
).sort(F.col("ANOMALY_COUNT").desc())

# 8g. Anomalies by day of week
agg_dow = anomaly_df.filter(F.col("IS_OVERPRICED") == 1).group_by(F.col("FLIGHT_DAY_OF_WEEK")).agg(
F.count(F.col("LEGID")).alias("ANOMALY_COUNT"),
F.avg(F.col("RESIDUAL")).alias("AVG_OVERPRICING")
).sort(F.col("ANOMALY_COUNT").desc())

# 8h. Anomalies by month
agg_month = anomaly_df.filter(F.col("IS_OVERPRICED") == 1).group_by(F.col("FLIGHT_MONTH")).agg(
F.count(F.col("LEGID")).alias("ANOMALY_COUNT"),
F.avg(F.col("RESIDUAL")).alias("AVG_OVERPRICING")
).sort(F.col("FLIGHT_MONTH"))

# 8i. Anomalies by departure time period
agg_depart = anomaly_df.filter(F.col("IS_OVERPRICED") == 1).group_by(F.col("DEPARTURE_TIME_PERIOD")).agg(
F.count(F.col("LEGID")).alias("ANOMALY_COUNT"),
F.avg(F.col("RESIDUAL")).alias("AVG_OVERPRICING")
).sort(F.col("ANOMALY_COUNT").desc())

# 8j. Anomalies by distance category
agg_distance = anomaly_df.filter(F.col("IS_OVERPRICED") == 1).group_by(F.col("DISTANCE_CATEGORY")).agg(
F.count(F.col("LEGID")).alias("ANOMALY_COUNT"),
F.avg(F.col("RESIDUAL")).alias("AVG_OVERPRICING")
).sort(F.col("ANOMALY_COUNT").desc())

# 8k. Anomalies by aircraft category
agg_aircraft = anomaly_df.filter(F.col("IS_OVERPRICED") == 1).group_by(F.col("AIRCRAFT_CATEGORY")).agg(
F.count(F.col("LEGID")).alias("ANOMALY_COUNT"),
F.avg(F.col("RESIDUAL")).alias("AVG_OVERPRICING")
).sort(F.col("ANOMALY_COUNT").desc())

# 8l. Small scatter sample for actual vs predicted chart (2000 rows only)
scatter_sample = anomaly_df.select(
F.col("TOTALFARE"), F.col("PREDICTED_FARE"),
F.col("RESIDUAL"), F.col("RESIDUAL_ZSCORE"), F.col("IS_ANOMALY")
).sample(n=2000)

# 8m. Residual distribution (binned in Snowflake)
resid_bins = anomaly_df.select(
F.round(F.col("RESIDUAL_ZSCORE"), 0).alias("ZSCORE_BIN")
).group_by(F.col("ZSCORE_BIN")).agg(
F.count(F.col("ZSCORE_BIN")).alias("BIN_COUNT")
).sort(F.col("ZSCORE_BIN"))

# 8n. Feature importance from model
try:
    xgb_model = model.to_xgboost()
    importance_values = xgb_model.feature_importances_
    importance_features = all_model_features
except Exception:
    importance_values = np.ones(len(all_model_features)) / len(all_model_features)
    importance_features = all_model_features

# ============================================================================
# STEP 9: CONVERT ONLY TINY AGGREGATIONS TO PANDAS FOR MATPLOTLIB
# ============================================================================
print("\nConverting tiny aggregations for charts (NOT raw data)...")

pd_airline = agg_airline.to_pandas()
pd_airline_rate = agg_airline_rate.to_pandas()
pd_route = agg_route.to_pandas()
pd_inventory = agg_inventory.to_pandas()
pd_urgency = agg_urgency.to_pandas()
pd_dow = agg_dow.to_pandas()
pd_month = agg_month.to_pandas()
pd_depart = agg_depart.to_pandas()
pd_distance = agg_distance.to_pandas()
pd_aircraft = agg_aircraft.to_pandas()
pd_scatter = scatter_sample.to_pandas()
pd_resid_bins = resid_bins.to_pandas()

print("Aggregations ready for plotting!")

# ============================================================================
# STEP 10: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: GENERATING VISUALIZATIONS")
print("=" * 80)

# =============================================
# CHART 1: MODEL PERFORMANCE (3 plots)
# =============================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('CHART 1: XGBoost Model Performance', fontsize=14, fontweight='bold')

# 1a. Actual vs Predicted
ax = axes[0]
normal = pd_scatter[pd_scatter['IS_ANOMALY'] == 0]
anomalies = pd_scatter[pd_scatter['IS_ANOMALY'] == 1]
ax.scatter(normal['TOTALFARE'], normal['PREDICTED_FARE'], alpha=0.4, s=8, color='#4CAF50', label='Fair Price')
ax.scatter(anomalies['TOTALFARE'], anomalies['PREDICTED_FARE'], alpha=0.6, s=20, color='#F44336', label='Anomaly')
max_val = pd_scatter['TOTALFARE'].max()
ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='Perfect Prediction')
ax.set_xlabel('Actual Fare ($)')
ax.set_ylabel('Predicted Fair Fare ($)')
ax.set_title(f'Actual vs Predicted (R² ≈ {r2_approx:.3f})')
ax.legend(fontsize=8)

# 1b. Residual Z-Score Distribution
ax = axes[1]
pd_resid_clean = pd_resid_bins[(pd_resid_bins['ZSCORE_BIN'] >= -6) & (pd_resid_bins['ZSCORE_BIN'] <= 8)]
ax.bar(pd_resid_clean['ZSCORE_BIN'], pd_resid_clean['BIN_COUNT'], color='#1f77b4', width=0.8)
ax.axvline(x=2.5, color='red', linewidth=2, linestyle='--', label='Overpriced (>2.5σ)')
ax.axvline(x=-2.5, color='blue', linewidth=2, linestyle='--', label='Underpriced (<-2.5σ)')
ax.set_xlabel('Residual Z-Score')
ax.set_ylabel('Number of Flights')
ax.set_title('Residual Z-Score Distribution')
ax.legend(fontsize=8)

# 1c. Model Summary
ax = axes[2]
ax.axis('off')
summary_text = (
f"XGBoost Model Summary\n"
f"{'='*35}\n\n"
f"Train RMSE: ${train_rmse:.2f}\n"
f"Test RMSE: ${test_rmse:.2f}\n"
f"Test MAE: ${test_mae:.2f}\n"
f"Test R² (approx): {r2_approx:.4f}\n\n"
f"{'='*35}\n"
f"Anomaly Detection\n"
f"{'='*35}\n\n"
f"Total Flights: {total:,}\n"
f"Overpriced: {overpriced:,} ({overpriced/total*100:.1f}%)\n"
f"Underpriced: {underpriced:,} ({underpriced/total*100:.1f}%)\n"
f"Fair Price: {total-total_anomalies:,} ({(total-total_anomalies)/total*100:.1f}%)\n\n"
f"Residual Mean: ${resid_mean:.2f}\n"
f"Residual Std: ${resid_std:.2f}"
)
ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
verticalalignment='top', fontfamily='monospace',
bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('Model & Anomaly Summary', fontweight='bold')

plt.tight_layout()
plt.show()

# =============================================
# CHART 2: FEATURE IMPORTANCE (2 plots)
# =============================================
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('CHART 2: What Drives Fair Flight Prices? (Feature Importance)', fontsize=14, fontweight='bold')

# 2a. Top 20 features bar chart
ax = axes[0]
feat_imp = sorted(zip(importance_features, importance_values), key=lambda x: x[1])
top20_feat = feat_imp[-20:]
feat_names = [f[0] for f in top20_feat]
feat_vals = [f[1] for f in top20_feat]
colors_fi = ['#d62728' if v == max(feat_vals) else '#1f77b4' for v in feat_vals]
ax.barh(feat_names, feat_vals, color=colors_fi)
ax.set_xlabel('Importance (Gain)')
ax.set_title('Top 20 Pricing Drivers')

# 2b. Top 10 pie chart
ax = axes[1]
top10_feat = feat_imp[-10:]
t10_names = [f[0] for f in top10_feat]
t10_vals = [f[1] for f in top10_feat]
other = sum(importance_values) - sum(t10_vals)
pie_labels = t10_names + ['All Others']
pie_vals = t10_vals + [other]
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(pie_vals)))
ax.pie(pie_vals, labels=pie_labels, autopct='%1.1f%%', colors=colors_pie, startangle=90, textprops={'fontsize': 8})
ax.set_title('Top 10 Feature Importance Share')

plt.tight_layout()
plt.show()

# =============================================
# CHART 3: ANOMALIES BY AIRLINE (3 plots)
# =============================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('CHART 3: Which Airlines Overprice? (Airline Analysis)', fontsize=14, fontweight='bold')

# 3a. Anomaly count
ax = axes[0]
top_al = pd_airline.head(15)
ax.barh(top_al['AIRLINE_CODE'], top_al['ANOMALY_COUNT'], color='#d62728')
ax.set_xlabel('Overpriced Flights Count')
ax.set_title('Top 15 Airlines: Overpriced Count')
ax.invert_yaxis()

# 3b. Average overpricing amount
ax = axes[1]
ax.barh(top_al['AIRLINE_CODE'], top_al['AVG_OVERPRICING'], color='#ff7f0e')
ax.set_xlabel('Avg Overpricing ($)')
ax.set_title('Avg Overpricing Severity by Airline')
ax.invert_yaxis()

# 3c. Anomaly rate (% of flights overpriced)
ax = axes[2]
top_rate = pd_airline_rate.head(15)
ax.barh(top_rate['AIRLINE_CODE'], top_rate['ANOMALY_RATE'], color='#9467bd')
ax.set_xlabel('Anomaly Rate (%)')
ax.set_title('% of Flights Overpriced by Airline')
ax.invert_yaxis()

plt.tight_layout()
plt.show()

# =============================================
# CHART 4: ANOMALIES BY ROUTE (2 plots)
# =============================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('CHART 4: Which Routes Are Most Overpriced?', fontsize=14, fontweight='bold')

# 4a. Top 20 routes by anomaly count
ax = axes[0]
ax.barh(pd_route['ROUTE'], pd_route['ANOMALY_COUNT'], color='#ff7f0e')
ax.set_xlabel('Overpriced Flights Count')
ax.set_title('Top 20 Routes: Overpriced Count')
ax.invert_yaxis()

# 4b. Top 20 routes by avg overpricing
ax = axes[1]
pd_route_sorted = pd_route.sort_values('AVG_OVERPRICING', ascending=True)
ax.barh(pd_route_sorted['ROUTE'], pd_route_sorted['AVG_OVERPRICING'], color='#2ca02c')
ax.set_xlabel('Avg Overpricing Amount ($)')
ax.set_title('Top 20 Routes: Avg Overpricing Severity')
ax.invert_yaxis()

plt.tight_layout()
plt.show()

# =============================================
# CHART 5: ANOMALIES BY INVENTORY (2 plots)
# =============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CHART 5: Does Seat Scarcity Drive Overpricing?', fontsize=14, fontweight='bold')

# 5a. Count by inventory
ax = axes[0]
inv_colors = {'Critical': '#d62728', 'Low': '#ff7f0e', 'Medium': '#ffbb78', 'High': '#2ca02c', 'Sold Out': '#7f7f7f'}
bar_colors = [inv_colors.get(x, '#1f77b4') for x in pd_inventory['INVENTORY_LEVEL']]
bars = ax.bar(pd_inventory['INVENTORY_LEVEL'], pd_inventory['ANOMALY_COUNT'], color=bar_colors)
ax.set_xlabel('Inventory Level')
ax.set_ylabel('Overpriced Flights')
ax.set_title('Overpriced Flights by Seat Availability')
for bar, val in zip(bars, pd_inventory['ANOMALY_COUNT']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50, f'{int(val):,}', ha='center', fontweight='bold', fontsize=9)

# 5b. Avg overpricing by inventory
ax = axes[1]
ax.bar(pd_inventory['INVENTORY_LEVEL'], pd_inventory['AVG_OVERPRICING'], color=bar_colors)
ax.set_xlabel('Inventory Level')
ax.set_ylabel('Avg Overpricing ($)')
ax.set_title('Avg Overpricing Amount by Inventory')
for i, val in enumerate(pd_inventory['AVG_OVERPRICING']):
    ax.text(i, val + 5, f'${val:.0f}', ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.show()

# =============================================
# CHART 6: ANOMALIES BY LEAD TIME (2 plots)
# =============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CHART 6: Does Booking Urgency Drive Overpricing?', fontsize=14, fontweight='bold')

# 6a. Count by urgency
ax = axes[0]
urgency_colors = ['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c']
bars = ax.bar(range(len(pd_urgency)), pd_urgency['ANOMALY_COUNT'], color=urgency_colors[:len(pd_urgency)])
ax.set_xticks(range(len(pd_urgency)))
ax.set_xticklabels(pd_urgency['LEAD_TIME_URGENCY'], rotation=15, fontsize=9)
ax.set_ylabel('Overpriced Flights')
ax.set_title('Overpriced Flights by Booking Window')
for bar, val in zip(bars, pd_urgency['ANOMALY_COUNT']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50, f'{int(val):,}', ha='center', fontweight='bold', fontsize=9)

# 6b. Avg overpricing by urgency
ax = axes[1]
ax.bar(range(len(pd_urgency)), pd_urgency['AVG_OVERPRICING'], color=urgency_colors[:len(pd_urgency)])
ax.set_xticks(range(len(pd_urgency)))
ax.set_xticklabels(pd_urgency['LEAD_TIME_URGENCY'], rotation=15, fontsize=9)
ax.set_ylabel('Avg Overpricing ($)')
ax.set_title('Avg Overpricing by Booking Window')

plt.tight_layout()
plt.show()

# =============================================
# CHART 7: ANOMALIES BY DAY & MONTH (2 plots)
# =============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CHART 7: When Does Overpricing Happen?', fontsize=14, fontweight='bold')

# 7a. By day of week
ax = axes[0]
ax.bar(pd_dow['FLIGHT_DAY_OF_WEEK'], pd_dow['ANOMALY_COUNT'], color='#1f77b4')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Overpriced Flights')
ax.set_title('Overpriced Flights by Day of Week')
ax.tick_params(axis='x', rotation=45)

# 7b. By month
ax = axes[1]
month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
# Cast to int (Snowflake returns Decimal) and drop any NaN rows so matplotlib
# doesn't see a mix of strings and floats on the categorical x-axis.
pd_month_plot = pd_month.dropna(subset=['FLIGHT_MONTH']).copy()
pd_month_plot['FLIGHT_MONTH'] = pd_month_plot['FLIGHT_MONTH'].astype(int)
pd_month_plot['MONTH_NAME'] = pd_month_plot['FLIGHT_MONTH'].map(month_names)
pd_month_plot = pd_month_plot.dropna(subset=['MONTH_NAME'])
ax.bar(pd_month_plot['MONTH_NAME'], pd_month_plot['ANOMALY_COUNT'], color='#ff7f0e')
ax.set_xlabel('Month')
ax.set_ylabel('Overpriced Flights')
ax.set_title('Overpriced Flights by Month')

plt.tight_layout()
plt.show()

# =============================================
# CHART 8: ANOMALIES BY DEPARTURE TIME & DISTANCE (2 plots)
# =============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CHART 8: Overpricing by Time of Day & Distance', fontsize=14, fontweight='bold')

# 8a. By departure time period
ax = axes[0]
depart_colors = {'Red-Eye': '#2196F3', 'Morning': '#FFC107', 'Afternoon': '#FF9800', 'Evening': '#9C27B0'}
bar_colors_dep = [depart_colors.get(x, '#1f77b4') for x in pd_depart['DEPARTURE_TIME_PERIOD']]
ax.bar(pd_depart['DEPARTURE_TIME_PERIOD'], pd_depart['ANOMALY_COUNT'], color=bar_colors_dep)
ax.set_xlabel('Departure Time')
ax.set_ylabel('Overpriced Flights')
ax.set_title('Overpriced Flights by Departure Time')

# 8b. By distance category
ax = axes[1]
ax.bar(pd_distance['DISTANCE_CATEGORY'], pd_distance['ANOMALY_COUNT'], color='#2ca02c')
ax.set_xlabel('Distance Category')
ax.set_ylabel('Overpriced Flights')
ax.set_title('Overpriced Flights by Flight Distance')
ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()

# =============================================
# CHART 9: ANOMALIES BY AIRCRAFT & OVERPRICING SCATTER (2 plots)
# =============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CHART 9: Aircraft Type & Overpricing Pattern', fontsize=14, fontweight='bold')

# 9a. By aircraft category
ax = axes[0]
ax.bar(pd_aircraft['AIRCRAFT_CATEGORY'], pd_aircraft['ANOMALY_COUNT'], color='#8c564b')
ax.set_xlabel('Aircraft Category')
ax.set_ylabel('Overpriced Flights')
ax.set_title('Overpriced Flights by Aircraft Type')

# 9b. Overpricing amount vs fare (scatter)
ax = axes[1]
overpriced_scatter = pd_scatter[pd_scatter['IS_ANOMALY'] == 1]
fair_scatter = pd_scatter[pd_scatter['IS_ANOMALY'] == 0]
ax.scatter(fair_scatter['TOTALFARE'], fair_scatter['RESIDUAL'], alpha=0.3, s=8, color='#4CAF50', label='Fair')
ax.scatter(overpriced_scatter['TOTALFARE'], overpriced_scatter['RESIDUAL'], alpha=0.6, s=20, color='#F44336', label='Anomaly')
ax.axhline(y=0, color='black', linewidth=1, linestyle='-')
ax.set_xlabel('Total Fare ($)')
ax.set_ylabel('Overpricing Amount ($)')
ax.set_title('Overpricing Amount vs Ticket Price')
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

# =============================================
# CHART 10: FINAL SUMMARY DASHBOARD
# =============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CHART 10: DOT/FAA Executive Summary Dashboard', fontsize=16, fontweight='bold')

# 10a. Pie chart - Fair vs Anomaly
ax = axes[0, 0]
pie_vals = [total - total_anomalies, overpriced, underpriced]
pie_labels = [f'Fair Price\n({total-total_anomalies:,})', f'Overpriced\n({overpriced:,})', f'Underpriced\n({underpriced:,})']
pie_colors = ['#4CAF50', '#F44336', '#2196F3']
ax.pie(pie_vals, labels=pie_labels, autopct='%1.1f%%', colors=pie_colors, startangle=90, textprops={'fontsize': 10})
ax.set_title('Overall Price Fairness', fontweight='bold')

# 10b. Top 5 worst airlines (by anomaly rate)
ax = axes[0, 1]
top5_rate = pd_airline_rate.head(5)
ax.barh(top5_rate['AIRLINE_CODE'], top5_rate['ANOMALY_RATE'], color='#d62728')
ax.set_xlabel('% Flights Overpriced')
ax.set_title('Top 5 Airlines by Overpricing Rate', fontweight='bold')
ax.invert_yaxis()
for i, v in enumerate(top5_rate['ANOMALY_RATE']):
    ax.text(v + 0.1, i, f'{v:.1f}%', va='center', fontweight='bold')

# 10c. Top 5 worst routes
ax = axes[1, 0]
top5_route = pd_route.head(5)
ax.barh(top5_route['ROUTE'], top5_route['ANOMALY_COUNT'], color='#ff7f0e')
ax.set_xlabel('Overpriced Flights')
ax.set_title('Top 5 Routes by Overpricing Count', fontweight='bold')
ax.invert_yaxis()
for i, v in enumerate(top5_route['ANOMALY_COUNT']):
    ax.text(v + 10, i, f'{int(v):,}', va='center', fontweight='bold')

# 10d. Key metrics text
ax = axes[1, 1]
ax.axis('off')
dashboard_text = (
f"KEY FINDINGS FOR DOT/FAA\n"
f"{'='*40}\n\n"
f"Model Accuracy (R²): {r2_approx:.1%}\n"
f"Test RMSE: ${test_rmse:.2f}\n\n"
f"{'='*40}\n"
f"ANOMALY DETECTION\n"
f"{'='*40}\n\n"
f"Flights Analyzed: {total:,}\n"
f"Overpriced: {overpriced:,} ({overpriced/total*100:.1f}%)\n"
f"Underpriced: {underpriced:,} ({underpriced/total*100:.1f}%)\n\n"
f"{'='*40}\n"
f"WORST OFFENDERS\n"
f"{'='*40}\n\n"
f"Worst Airline: {pd_airline_rate.iloc[0]['AIRLINE_CODE']}\n"
f" Rate: {pd_airline_rate.iloc[0]['ANOMALY_RATE']:.1f}% overpriced\n\n"
f"Worst Route: {pd_route.iloc[0]['ROUTE']}\n"
f" Count: {int(pd_route.iloc[0]['ANOMALY_COUNT']):,} overpriced\n"
f" Avg Markup: ${pd_route.iloc[0]['AVG_OVERPRICING']:.0f}"
)
ax.text(0.05, 0.95, dashboard_text, transform=ax.transAxes, fontsize=11,
verticalalignment='top', fontfamily='monospace',
bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.show()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅✅✅ XGBOOST ANOMALY DETECTION COMPLETE ✅✅✅")
print("=" * 80)
print(f"\n📊 MODEL: XGBoost Regressor (300 trees, depth 7)")
print(f" R² ≈ {r2_approx:.4f} | RMSE = ${test_rmse:.2f} | MAE = ${test_mae:.2f}")
print(f"\n🔍 ANOMALIES DETECTED:")
print(f" {overpriced:,} overpriced flights ({overpriced/total*100:.1f}%)")
print(f" {underpriced:,} underpriced flights ({underpriced/total*100:.1f}%)")
print(f"\n📈 10 CHARTS GENERATED:")
print(f" 1. Model Performance (Actual vs Predicted + Residuals)")
print(f" 2. Feature Importance (Top 20 + Pie)")
print(f" 3. Airline Analysis (Count + Severity + Rate)")
print(f" 4. Route Analysis (Count + Severity)")
print(f" 5. Inventory Level Analysis")
print(f" 6. Booking Lead Time Analysis")
print(f" 7. Day of Week + Month Analysis")
print(f" 8. Departure Time + Distance Analysis")
print(f" 9. Aircraft Type + Scatter")
print(f" 10. Executive Summary Dashboard")
print(f"\n🏛️ ALL COMPUTATION RAN IN SNOWFLAKE")
print(f" Only tiny aggregations (< 100 rows each) used for charts")
print(f"\n📋 DataFrames available:")
print(f" - features_df: 39 engineered features (82M rows in Snowflake)")
print(f" - anomaly_df: predictions + anomaly flags (500K in Snowflake)")
