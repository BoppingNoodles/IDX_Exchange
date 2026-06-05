import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# If True, writes cleaned outputs to disk (overwrites existing files).
WRITE_OUTPUTS = True


def load_mortgage_monthly_rates() -> pd.DataFrame:
    """
    Load 30-year fixed mortgage rates and aggregate to monthly averages.

    Uses local `MORTGAGE30US.csv` only to avoid a FRED network/API dependency.
    """
    local_path = Path("MORTGAGE30US.csv")
    if local_path.exists():
        mortgage = pd.read_csv(local_path, parse_dates=["observation_date"])
        # Local schema: observation_date, MORTGAGE30US
        value_col = "MORTGAGE30US" if "MORTGAGE30US" in mortgage.columns else mortgage.columns[-1]
        mortgage = mortgage.rename(columns={"observation_date": "date", value_col: "rate_30yr_fixed"})
    else:
        # FRED API/network fallback intentionally disabled.
        # url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
        # mortgage = pd.read_csv(url, parse_dates=["observation_date"])
        # mortgage.columns = ["date", "rate_30yr_fixed"]
        return pd.DataFrame(columns=["year_month", "rate_30yr_fixed"])

    mortgage["year_month"] = mortgage["date"].dt.to_period("M")
    return mortgage.groupby("year_month")["rate_30yr_fixed"].mean().reset_index()


def get_lower_and_upper(df: pd.DataFrame, col: str):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


# ==========================================
# PART 1: CRMLS SOLD DATA PIPELINE
# ==========================================

# 1. Load pre-aggregated raw data
sold_df = pd.read_csv("AggSold.csv", encoding="ISO-8859-1", low_memory=False)
print(f"AggSold.csv: {len(sold_df)}")
print(f"Rows after loading aggregated sold data: {len(sold_df)}")

# 2. Filter to Residential
if not sold_df.empty and "PropertyType" in sold_df.columns:
    print(f"Rows before Residential Filter: {len(sold_df)}")
    sold_df = sold_df[sold_df["PropertyType"] == "Residential"]
    print(f"Rows after Residential Filter: {len(sold_df)}")

# 3. Drop columns with >90% missing values
if not sold_df.empty:
    print(f"Number of columns before dropping: {sold_df.shape[1]}")
    threshold = 0.1 * len(sold_df)
    sold_df.dropna(thresh=threshold, axis=1, inplace=True)
    print(f"Number of columns after dropping: {sold_df.shape[1]}")

# 4. Outlier & Distribution Graphs (EDA)
core_numeric_sold = ["ClosePrice", "ListPrice", "DaysOnMarket"]
if not sold_df.empty:
    _ = sold_df[core_numeric_sold].describe()
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 40))
    for i, col in enumerate(core_numeric_sold):
        if col in sold_df.columns:
            sns.histplot(sold_df[col].dropna(), bins=50, kde=True, ax=axes[i, 0])
            axes[i, 0].set_title(f"{col} Distribution")
            sns.boxplot(x=sold_df[col].dropna(), ax=axes[i, 1])
            axes[i, 1].set_title(f"{col} Outliers")
    plt.tight_layout()
    plt.show()

# 5. Fetch and Merge mortgage data (notebook step)
mortgage_monthly = load_mortgage_monthly_rates()
if not sold_df.empty and "CloseDate" in sold_df.columns:
    sold_df["year_month"] = pd.to_datetime(sold_df["CloseDate"]).dt.to_period("M")
    sold_with_rates_df = sold_df.merge(mortgage_monthly, on="year_month", how="left")
else:
    sold_with_rates_df = sold_df.copy()

if not sold_with_rates_df.empty and "rate_30yr_fixed" in sold_with_rates_df.columns:
    print(f"Null mortgage rates: {sold_with_rates_df['rate_30yr_fixed'].isnull().sum()}")
    preview_cols = [c for c in ["CloseDate", "year_month", "ClosePrice", "rate_30yr_fixed"] if c in sold_with_rates_df.columns]
    if preview_cols:
        print("\nPreview of the enriched dataset:")
        print(sold_with_rates_df[preview_cols].head())

# 6. Date conversions & deduplication (notebook step)
date_cols = ["CloseDate", "ListingContractDate", "PurchaseContractDate", "ContractStatusChangeDate"]
for col in date_cols:
    if col in sold_with_rates_df.columns:
        sold_with_rates_df[col] = pd.to_datetime(sold_with_rates_df[col])

if "ListingKey" in sold_with_rates_df.columns:
    sold_with_rates_df = sold_with_rates_df.drop_duplicates(subset=["ListingKey"], keep="last")

# 7. Drop critical missing rows (notebook step)
must_columns_sold = ["ListingKey", "ClosePrice", "CloseDate", "City"]
existing_must_cols = [c for c in must_columns_sold if c in sold_with_rates_df.columns]
if existing_must_cols:
    sold_with_rates_df = sold_with_rates_df.dropna(subset=existing_must_cols)

# 8. Impute missing values (notebook step)
zero_fill_cols = ["GarageSpaces", "ParkingTotal", "FireplacesTotal"]
for col in zero_fill_cols:
    if col in sold_with_rates_df.columns:
        sold_with_rates_df[col] = sold_with_rates_df[col].fillna(0)

fill_no_cols = ["PoolPrivateYN", "ViewYN", "CoolingYN", "HeatingYN"]
for col in fill_no_cols:
    if col in sold_with_rates_df.columns:
        sold_with_rates_df[col] = sold_with_rates_df[col].fillna(False)

cat_fill_cols = ["ArchitectureStyle", "Heating", "Cooling", "WaterSource"]
for col in cat_fill_cols:
    if col in sold_with_rates_df.columns:
        sold_with_rates_df[col] = sold_with_rates_df[col].fillna("Unknown")

num_fill_cols = ["LotSizeArea", "YearBuilt"]
for col in num_fill_cols:
    if col in sold_with_rates_df.columns:
        sold_with_rates_df[col] = sold_with_rates_df[col].fillna(sold_with_rates_df[col].median())

# 9. Drop missing Living Area (notebook step)
if "LivingArea" in sold_with_rates_df.columns:
    sold_with_rates_df = sold_with_rates_df.dropna(subset=["LivingArea"])

# 10. Drop agent cols + clean full name (notebook step)
agent_drop_cols = ["ListAgentFirstName", "ListAgentLastName"]
existing_agent_drop_cols = [c for c in agent_drop_cols if c in sold_with_rates_df.columns]
if existing_agent_drop_cols:
    sold_with_rates_df = sold_with_rates_df.drop(columns=existing_agent_drop_cols)

if "ListAgentFullName" in sold_with_rates_df.columns:
    sold_with_rates_df["ListAgentFullName"] = (
        sold_with_rates_df["ListAgentFullName"]
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", pd.NA)
    )

# 11. Remove invalid numeric values (notebook step)
numeric_required = ["ClosePrice", "LivingArea", "DaysOnMarket", "BedroomsTotal", "BathroomsTotalInteger"]
if all(c in sold_with_rates_df.columns for c in numeric_required):
    sold_with_rates_df = sold_with_rates_df[
        (sold_with_rates_df["ClosePrice"] > 0)
        & (sold_with_rates_df["LivingArea"] > 0)
        & (sold_with_rates_df["DaysOnMarket"] >= 0)
        & (sold_with_rates_df["BedroomsTotal"] >= 0)
        & (sold_with_rates_df["BathroomsTotalInteger"] >= 0)
    ]

# 12. Data consistency flags (match SoldAnalysis.ipynb)
if all(c in sold_with_rates_df.columns for c in ["ListingContractDate", "CloseDate"]):
    sold_with_rates_df["listing_after_close_flag"] = sold_with_rates_df["ListingContractDate"] > sold_with_rates_df["CloseDate"]
else:
    sold_with_rates_df["listing_after_close_flag"] = False

if all(c in sold_with_rates_df.columns for c in ["PurchaseContractDate", "CloseDate"]):
    sold_with_rates_df["purchase_after_close_flag"] = sold_with_rates_df["PurchaseContractDate"] > sold_with_rates_df["CloseDate"]
else:
    sold_with_rates_df["purchase_after_close_flag"] = False

if "DaysOnMarket" in sold_with_rates_df.columns:
    sold_with_rates_df["negative_timeline_flag"] = sold_with_rates_df["DaysOnMarket"] < 0
else:
    sold_with_rates_df["negative_timeline_flag"] = False

print(f"Number of rows before cleaning {sold_with_rates_df.shape[0]}")

# 13. Geographic filters (notebook step)
for c in ["Latitude", "Longitude", "PostalCode", "StateOrProvince"]:
    if c not in sold_with_rates_df.columns:
        sold_with_rates_df[c] = pd.NA

sold_with_rates_df["missing_coords"] = sold_with_rates_df["Latitude"].isnull() | sold_with_rates_df["Longitude"].isnull()
sold_with_rates_df["sentinel_coords"] = (sold_with_rates_df["Latitude"] == 0) | (sold_with_rates_df["Longitude"] == 0)
sold_with_rates_df["cal_coords"] = sold_with_rates_df["Longitude"] > 0
sold_with_rates_df["PostalCode"] = sold_with_rates_df["PostalCode"].astype(str)
sold_with_rates_df["is_california"] = (sold_with_rates_df["StateOrProvince"].isin(["CA", "California"])) | (
    sold_with_rates_df["PostalCode"].str.startswith("9")
)

sold_with_rates_df = sold_with_rates_df[
    (sold_with_rates_df["listing_after_close_flag"] == False)
    & (sold_with_rates_df["purchase_after_close_flag"] == False)
    & (sold_with_rates_df["negative_timeline_flag"] == False)
    & (sold_with_rates_df["missing_coords"] == False)
    & (sold_with_rates_df["sentinel_coords"] == False)
    & (sold_with_rates_df["cal_coords"] == False)
    & (sold_with_rates_df["is_california"] == True)
]
print(f"Number of rows after cleaning: {sold_with_rates_df.shape[0]}")

# 14. Feature creation (notebook step)
if all(c in sold_with_rates_df.columns for c in ["ClosePrice", "LivingArea"]):
    sold_with_rates_df["PricePerSqFt"] = sold_with_rates_df["ClosePrice"] / sold_with_rates_df["LivingArea"]
if "CloseDate" in sold_with_rates_df.columns:
    sold_with_rates_df["Year"] = sold_with_rates_df["CloseDate"].dt.year
    sold_with_rates_df["Month"] = sold_with_rates_df["CloseDate"].dt.month
    sold_with_rates_df["YrMo"] = sold_with_rates_df["CloseDate"].dt.to_period("M").astype(str)
if "OriginalListPrice" in sold_with_rates_df.columns:
    sold_with_rates_df["PriceRatio"] = sold_with_rates_df["ClosePrice"] / sold_with_rates_df["OriginalListPrice"]
    sold_with_rates_df["CloseToOriginalListRatio"] = sold_with_rates_df["ClosePrice"] / sold_with_rates_df["OriginalListPrice"]
if all(c in sold_with_rates_df.columns for c in ["PurchaseContractDate", "ListingContractDate"]):
    sold_with_rates_df["ListingToContractDays"] = sold_with_rates_df["PurchaseContractDate"] - sold_with_rates_df["ListingContractDate"]
if "PurchaseContractDate" in sold_with_rates_df.columns:
    sold_with_rates_df["ContractToCloseDays"] = sold_with_rates_df["CloseDate"] - sold_with_rates_df["PurchaseContractDate"]

# 15. IQR outlier flagging + filtered dataset (notebook step)
sold_rates_filtered_df = sold_with_rates_df.copy()
if all(c in sold_with_rates_df.columns for c in ["ClosePrice", "LivingArea", "DaysOnMarket"]):
    close_price_lower, close_price_upper = get_lower_and_upper(sold_with_rates_df, "ClosePrice")
    living_area_lower, living_area_upper = get_lower_and_upper(sold_with_rates_df, "LivingArea")
    days_lower, days_upper = get_lower_and_upper(sold_with_rates_df, "DaysOnMarket")

    sold_with_rates_df["closeprice_outlier_flag"] = ~sold_with_rates_df["ClosePrice"].between(close_price_lower, close_price_upper)
    sold_with_rates_df["livingarea_outlier_flag"] = ~sold_with_rates_df["LivingArea"].between(living_area_lower, living_area_upper)
    sold_with_rates_df["dom_outlier_flag"] = ~sold_with_rates_df["DaysOnMarket"].between(days_lower, days_upper)

    sold_rates_filtered_df = sold_with_rates_df[
        (sold_with_rates_df["closeprice_outlier_flag"] == False)
        & (sold_with_rates_df["livingarea_outlier_flag"] == False)
        & (sold_with_rates_df["dom_outlier_flag"] == False)
    ]

print(f"Full SOLD dataset:     {len(sold_with_rates_df):,} rows")
print(f"Filtered SOLD dataset: {len(sold_rates_filtered_df):,} rows")
print(f"Removed SOLD:          {len(sold_with_rates_df) - len(sold_rates_filtered_df):,} rows")

if WRITE_OUTPUTS:
    sold_rates_filtered_df.to_csv("Final_Sold.csv", index=False)


# ==========================================
# PART 2: CRMLS LISTING DATA PIPELINE
# ==========================================

# 1. Load pre-aggregated raw data
listing_df = pd.read_csv("AggListing.csv", encoding="ISO-8859-1", low_memory=False)
print(f"AggListing.csv: {len(listing_df)}")
print(f"Rows after loading aggregated listing data: {len(listing_df)}")

# 2. Filter to Residential
if not listing_df.empty and "PropertyType" in listing_df.columns:
    print(f"Rows before Residential Filter: {len(listing_df)}")
    listing_df = listing_df[listing_df["PropertyType"] == "Residential"]
    print(f"Rows after Residential Filter: {len(listing_df)}")

# 3. Drop columns with >90% missing values
if not listing_df.empty:
    print(f"Number of columns before dropping: {listing_df.shape[1]}")
    threshold = 0.1 * len(listing_df)
    listing_df.dropna(thresh=threshold, axis=1, inplace=True)
    print(f"Number of columns after dropping: {listing_df.shape[1]}")

# 4. Outlier & Distribution Graphs (EDA)
core_numeric_listing = ["ClosePrice", "LivingArea", "DaysOnMarket"]
if not listing_df.empty:
    _ = listing_df[core_numeric_listing].describe()
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 40))
    for i, col in enumerate(core_numeric_listing):
        if col in listing_df.columns:
            sns.histplot(listing_df[col].dropna(), bins=50, kde=True, ax=axes[i, 0])
            axes[i, 0].set_title(f"{col} Distribution")
            sns.boxplot(x=listing_df[col].dropna(), ax=axes[i, 1])
            axes[i, 1].set_title(f"{col} Outliers")
    plt.tight_layout()
    plt.show()

# 5. Merge mortgage data (notebook step)
if not listing_df.empty and "ListingContractDate" in listing_df.columns:
    listing_df["year_month"] = pd.to_datetime(listing_df["ListingContractDate"]).dt.to_period("M")
    listing_with_rates_df = listing_df.merge(mortgage_monthly, on="year_month", how="left")
else:
    listing_with_rates_df = listing_df.copy()

if not listing_with_rates_df.empty and "rate_30yr_fixed" in listing_with_rates_df.columns:
    print(f"Null mortgage rates: {listing_with_rates_df['rate_30yr_fixed'].isnull().sum()}")
    preview_cols = [c for c in ["ListingContractDate", "year_month", "ListPrice", "rate_30yr_fixed"] if c in listing_with_rates_df.columns]
    if preview_cols:
        print("\nPreview of the enriched dataset:")
        print(listing_with_rates_df[preview_cols].head())

# 6. Date conversions & deduplication (notebook step)
for col in date_cols:
    if col in listing_with_rates_df.columns:
        listing_with_rates_df[col] = pd.to_datetime(listing_with_rates_df[col])

if "ListingKey" in listing_with_rates_df.columns:
    listing_with_rates_df = listing_with_rates_df.drop_duplicates(subset=["ListingKey"], keep="last")

listing_with_rates_df = listing_with_rates_df.drop(columns=[c for c in listing_with_rates_df.columns if c.endswith(".1")])

# 7. Drop critical missing rows (notebook step)
must_columns_listing = ["ListingKey", "ListPrice", "City"]
existing_must_cols = [c for c in must_columns_listing if c in listing_with_rates_df.columns]
if existing_must_cols:
    listing_with_rates_df = listing_with_rates_df.dropna(subset=existing_must_cols)

# 8. Impute missing values (notebook step)
for col in zero_fill_cols:
    if col in listing_with_rates_df.columns:
        listing_with_rates_df[col] = listing_with_rates_df[col].fillna(0)
for col in fill_no_cols:
    if col in listing_with_rates_df.columns:
        listing_with_rates_df[col] = listing_with_rates_df[col].fillna(False)
for col in cat_fill_cols:
    if col in listing_with_rates_df.columns:
        listing_with_rates_df[col] = listing_with_rates_df[col].fillna("Unknown")
for col in num_fill_cols:
    if col in listing_with_rates_df.columns:
        listing_with_rates_df[col] = listing_with_rates_df[col].fillna(listing_with_rates_df[col].median())

# 9. Drop missing Living Area (notebook step)
if "LivingArea" in listing_with_rates_df.columns:
    listing_with_rates_df = listing_with_rates_df.dropna(subset=["LivingArea"])

# 10. Drop agent cols + clean full name (notebook step)
existing_agent_drop_cols = [c for c in agent_drop_cols if c in listing_with_rates_df.columns]
if existing_agent_drop_cols:
    listing_with_rates_df = listing_with_rates_df.drop(columns=existing_agent_drop_cols)
if "ListAgentFullName" in listing_with_rates_df.columns:
    listing_with_rates_df["ListAgentFullName"] = (
        listing_with_rates_df["ListAgentFullName"]
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", pd.NA)
    )

# 11. Data consistency flags (match ListingAnalysis.ipynb)
if all(c in listing_with_rates_df.columns for c in ["ListingContractDate", "PurchaseContractDate"]):
    invalid_contracts = (listing_with_rates_df["ListingContractDate"] > listing_with_rates_df["PurchaseContractDate"]) & listing_with_rates_df[
        "PurchaseContractDate"
    ].notna()
    listing_with_rates_df["contract_before_listing_flag"] = invalid_contracts
else:
    listing_with_rates_df["contract_before_listing_flag"] = False

if "DaysOnMarket" in listing_with_rates_df.columns:
    listing_with_rates_df["negative_timeline_flag"] = listing_with_rates_df["DaysOnMarket"] < 0
else:
    listing_with_rates_df["negative_timeline_flag"] = False

print(f"Contract Date Errors: {int(listing_with_rates_df['contract_before_listing_flag'].sum())}")
print(f"Negative Timeline Errors: {int(listing_with_rates_df['negative_timeline_flag'].sum())}")

# 12. Geographic filters (notebook step)
print(f"Number of rows before cleaning: {listing_with_rates_df.shape[0]}")
for c in ["Latitude", "Longitude", "PostalCode", "StateOrProvince"]:
    if c not in listing_with_rates_df.columns:
        listing_with_rates_df[c] = pd.NA

listing_with_rates_df["missing_coords"] = listing_with_rates_df["Latitude"].isnull() | listing_with_rates_df["Longitude"].isnull()
listing_with_rates_df["sentinel_coords"] = (listing_with_rates_df["Latitude"] == 0) | (listing_with_rates_df["Longitude"] == 0)
listing_with_rates_df["cal_coords"] = listing_with_rates_df["Longitude"] > 0
listing_with_rates_df["PostalCode"] = listing_with_rates_df["PostalCode"].astype(str)
listing_with_rates_df["is_california"] = (listing_with_rates_df["StateOrProvince"].isin(["CA", "California"])) | (
    listing_with_rates_df["PostalCode"].str.startswith("9")
)

listing_with_rates_df = listing_with_rates_df[
    (listing_with_rates_df["missing_coords"] == False)
    & (listing_with_rates_df["sentinel_coords"] == False)
    & (listing_with_rates_df["cal_coords"] == False)
    & (listing_with_rates_df["negative_timeline_flag"] == False)
    & (listing_with_rates_df["contract_before_listing_flag"] == False)
    & (listing_with_rates_df["is_california"] == True)
]
print(f"Number of rows after cleaning: {listing_with_rates_df.shape[0]}")

# 13. Additional sanity filters (match notebook)
if "LivingArea" in listing_with_rates_df.columns:
    listing_with_rates_df = listing_with_rates_df[
        listing_with_rates_df["LivingArea"].isna() | (listing_with_rates_df["LivingArea"] > 0)
    ]
if "Longitude" in listing_with_rates_df.columns:
    # Notebook logic gates on Latitude.isna() but checks Longitude range.
    listing_with_rates_df = listing_with_rates_df[
        listing_with_rates_df["Latitude"].isna() | listing_with_rates_df["Longitude"].between(-124, -114)
    ]
if "DaysOnMarket" in listing_with_rates_df.columns:
    listing_with_rates_df = listing_with_rates_df[
        listing_with_rates_df["DaysOnMarket"].isna() | (listing_with_rates_df["DaysOnMarket"] >= 0)
    ]

# 14. Feature creation (notebook step)
if "ListingContractDate" in listing_with_rates_df.columns:
    listing_with_rates_df["Year"] = listing_with_rates_df["ListingContractDate"].dt.year
    listing_with_rates_df["Month"] = listing_with_rates_df["ListingContractDate"].dt.month
    listing_with_rates_df["YrMo"] = listing_with_rates_df["ListingContractDate"].dt.to_period("M").astype(str)
if all(c in listing_with_rates_df.columns for c in ["ListPrice", "LivingArea"]):
    listing_with_rates_df["ListPricePerSqFt"] = listing_with_rates_df["ListPrice"] / listing_with_rates_df["LivingArea"]

# 15. IQR outlier flagging + filtered dataset (notebook step)
listing_rates_filtered_df = listing_with_rates_df.copy()
if all(c in listing_with_rates_df.columns for c in ["ClosePrice", "LivingArea", "DaysOnMarket"]):
    close_price_lower, close_price_upper = get_lower_and_upper(listing_with_rates_df, "ClosePrice")
    living_area_lower, living_area_upper = get_lower_and_upper(listing_with_rates_df, "LivingArea")
    days_lower, days_upper = get_lower_and_upper(listing_with_rates_df, "DaysOnMarket")

    listing_with_rates_df["closeprice_outlier_flag"] = ~listing_with_rates_df["ClosePrice"].between(close_price_lower, close_price_upper)
    listing_with_rates_df["livingarea_outlier_flag"] = ~listing_with_rates_df["LivingArea"].between(living_area_lower, living_area_upper)
    listing_with_rates_df["dom_outlier_flag"] = ~listing_with_rates_df["DaysOnMarket"].between(days_lower, days_upper)

    listing_rates_filtered_df = listing_with_rates_df[
        (listing_with_rates_df["closeprice_outlier_flag"] == False)
        & (listing_with_rates_df["livingarea_outlier_flag"] == False)
        & (listing_with_rates_df["dom_outlier_flag"] == False)
    ]

print(f"Full LISTING dataset:     {len(listing_with_rates_df):,} rows")
print(f"Filtered LISTING dataset: {len(listing_rates_filtered_df):,} rows")
print(f"Removed LISTING:          {len(listing_with_rates_df) - len(listing_rates_filtered_df):,} rows")

if WRITE_OUTPUTS:
    listing_rates_filtered_df.to_csv("Final_Listing.csv", index=False)

