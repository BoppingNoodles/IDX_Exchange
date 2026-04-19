import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# PART 1: CRMLS SOLD DATA PIPELINE
# ==========================================

# 1. Data Aggregation
sold_data = glob.glob('**/CRMLSSold*.csv', recursive=True)
df_list = []
for file in sold_data:
    df = pd.read_csv(file, encoding='ISO-8859-1', low_memory=False)
    print(f"{file}: {len(df)}")
    df_list.append(df)
sold_df = pd.concat(df_list, ignore_index=True)

print(f'Rows after aggregation: {len(sold_df)}')

# 2. Filter to Residential
print(f"Rows before Residential Filter: {len(sold_df)}")
sold_df = sold_df[sold_df['PropertyType'] == "Residential"]
print(f"Rows after Residential Filter: {len(sold_df)}")

# 3. Drop columns with >90% missing values
print(f"Number of columns before dropping: {sold_df.shape[1]}")
threshold = 0.1 * len(sold_df)
sold_df.dropna(thresh=threshold, axis=1, inplace=True)
print(f"Number of columns after dropping: {sold_df.shape[1]}")

# 4. Outlier & Distribution Graphs
core_numeric_sold = ['ClosePrice', 'ListPrice', 'DaysOnMarket']
sold_df[core_numeric_sold].describe()

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 40))

for i, col in enumerate(core_numeric_sold):
    if col in sold_df.columns:
        sns.histplot(sold_df[col].dropna(), bins=50, kde=True, ax=axes[i, 0])
        axes[i, 0].set_title(f'{col} Distribution')
        sns.boxplot(x=sold_df[col].dropna(), ax=axes[i, 1])
        axes[i, 1].set_title(f'{col} Outliers')
plt.tight_layout()
plt.show()

# 5. Fetch and Merge FRED Mortgage Data
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
mortgage = pd.read_csv(url, parse_dates=['observation_date'])
mortgage.columns = ['date', 'rate_30yr_fixed']
mortgage['year_month'] = mortgage['date'].dt.to_period('M')
mortgage_monthly = mortgage.groupby('year_month')['rate_30yr_fixed'].mean().reset_index()

sold_df['year_month'] = pd.to_datetime(sold_df['CloseDate']).dt.to_period('M')
sold_with_rates_df = sold_df.merge(mortgage_monthly, on='year_month', how='left')

print(f"Null mortgage rates: {sold_with_rates_df['rate_30yr_fixed'].isnull().sum()}")
print("\nPreview of the enriched dataset:")
print(sold_with_rates_df[['CloseDate', 'year_month', 'ClosePrice', 'rate_30yr_fixed']].head())

# 6. Date Conversions & Deduplication
date_cols = ['CloseDate', 'ListingContractDate', 'PurchaseContractDate', 'ContractStatusChangeDate']
for col in date_cols:
    if col in sold_with_rates_df.columns:
        sold_with_rates_df[col] = pd.to_datetime(sold_with_rates_df[col])

sold_with_rates_df = sold_with_rates_df.drop_duplicates(subset=['ListingKey'], keep='last')

# 7. Drop Critical Missing Rows
must_columns_sold = ['ListingKey', 'ClosePrice', 'CloseDate', 'City']
sold_with_rates_df = sold_with_rates_df.dropna(subset=must_columns_sold)

# 8. Impute Missing Values
zero_fill_cols = ['GarageSpaces', 'ParkingTotal', 'FireplacesTotal']
for col in zero_fill_cols:
    if col in sold_with_rates_df.columns:
        sold_with_rates_df[col] = sold_with_rates_df[col].fillna(0)

fill_no_cols = ['PoolPrivateYN', 'ViewYN', 'CoolingYN', 'HeatingYN']
for col in fill_no_cols:
    if col in sold_with_rates_df.columns:
        sold_with_rates_df[col] = sold_with_rates_df[col].fillna(False)

cat_fill_cols = ['ArchitectureStyle', 'Heating', 'Cooling', 'WaterSource']
for col in cat_fill_cols:
    if col in sold_with_rates_df.columns:
        sold_with_rates_df[col] = sold_with_rates_df[col].fillna('Unknown')

num_fill_cols = ['LotSizeArea', 'YearBuilt']
for col in num_fill_cols:
    if col in sold_with_rates_df.columns:
        sold_with_rates_df[col] = sold_with_rates_df[col].fillna(sold_with_rates_df[col].median())

# 9. Clean Invalid Numeric Data
sold_with_rates_df = sold_with_rates_df.dropna(subset=['LivingArea'])
sold_with_rates_df = sold_with_rates_df[sold_with_rates_df['LivingArea'] > 0]
sold_with_rates_df = sold_with_rates_df[
    (sold_with_rates_df['ClosePrice'] > 0) &
    (sold_with_rates_df['LivingArea'] > 0) &
    (sold_with_rates_df['DaysOnMarket'] >= 0) &
    (sold_with_rates_df['BedroomsTotal'] >= 0) &
    (sold_with_rates_df['BathroomsTotalInteger'] >= 0)
]

# 10. Data Consistency Flags
# FIX: listing_after_close_flag uses > (listing date should not be after contract date)
# FIX: purchase_after_close_flag compares PurchaseContractDate vs CloseDate (not ListingContractDate)
sold_with_rates_df['listing_after_close_flag'] = sold_with_rates_df['ListingContractDate'] > sold_with_rates_df['PurchaseContractDate']
sold_with_rates_df['purchase_after_close_flag'] = sold_with_rates_df['PurchaseContractDate'] > sold_with_rates_df['CloseDate']
sold_with_rates_df['negative_timeline_flag'] = sold_with_rates_df['DaysOnMarket'] < 0

print(f"Number of rows before cleaning {sold_with_rates_df.shape[0]}")

# 11. Geographic Filters
sold_with_rates_df['missing_coords'] = sold_with_rates_df['Latitude'].isnull() | sold_with_rates_df['Longitude'].isnull()
sold_with_rates_df['sentinel_coords'] = (sold_with_rates_df['Latitude'] == 0) | (sold_with_rates_df['Longitude'] == 0)
sold_with_rates_df['cal_coords'] = sold_with_rates_df['Longitude'] > 0
sold_with_rates_df['PostalCode'] = sold_with_rates_df['PostalCode'].astype(str)
sold_with_rates_df['is_california'] = (
    sold_with_rates_df['StateOrProvince'].isin(['CA', 'California'])
) | (
    sold_with_rates_df['PostalCode'].str.startswith('9')
)

sold_with_rates_df = sold_with_rates_df[
    (sold_with_rates_df['listing_after_close_flag'] == False) &
    (sold_with_rates_df['purchase_after_close_flag'] == False) &
    (sold_with_rates_df['negative_timeline_flag'] == False) &
    (sold_with_rates_df['missing_coords'] == False) &
    (sold_with_rates_df['sentinel_coords'] == False) &
    (sold_with_rates_df['cal_coords'] == False) &
    (sold_with_rates_df['is_california'] == True)
]
print(f'Number of rows after cleaning: {sold_with_rates_df.shape[0]}')

# 12. Feature Engineering (Sold Dataset)
sold_with_rates_df['PriceRatio'] = sold_with_rates_df['ClosePrice'] / sold_with_rates_df['OriginalListPrice']
sold_with_rates_df['PricePerSqFt'] = sold_with_rates_df['ClosePrice'] / sold_with_rates_df['LivingArea']
sold_with_rates_df['Year'] = sold_with_rates_df['CloseDate'].dt.year
sold_with_rates_df['Month'] = sold_with_rates_df['CloseDate'].dt.month
sold_with_rates_df['YrMo'] = sold_with_rates_df['CloseDate'].dt.to_period('M').astype(str)
sold_with_rates_df['CloseToOriginalListRatio'] = sold_with_rates_df['ClosePrice'] / sold_with_rates_df['OriginalListPrice']
sold_with_rates_df['ListingToContractDays'] = sold_with_rates_df['PurchaseContractDate'] - sold_with_rates_df['ListingContractDate']
sold_with_rates_df['ContractToCloseDays'] = sold_with_rates_df['CloseDate'] - sold_with_rates_df['PurchaseContractDate']

columns = ['PriceRatio', 'PricePerSqFt', 'Year', 'Month', 'YrMo', 'CloseToOriginalListRatio', 'ListingToContractDays', 'ContractToCloseDays']
print(sold_with_rates_df[columns].head())

# 13. Segmented Analysis
residential_analysis = sold_with_rates_df.groupby('PropertySubType').agg({
    'ClosePrice': 'median',
    'PricePerSqFt': 'median',
    'DaysOnMarket': 'mean',
    'ListingKey': 'count'
}).reset_index()

area_analysis = sold_with_rates_df.groupby(['CountyOrParish', 'MLSAreaMajor']).agg({
    'ClosePrice': 'median',
    'PricePerSqFt': 'median',
    'DaysOnMarket': 'mean',
    'ListingKey': 'count'
}).reset_index()

office_analysis = sold_with_rates_df.groupby(['ListOfficeName', 'BuyerOfficeName']).agg({
    'ClosePrice': 'median',
    'PricePerSqFt': 'median',
    'DaysOnMarket': 'mean',
    'ListingKey': 'count'
}).reset_index()

print(f"Property Type\n{'-' * 20}\n{residential_analysis.head()}\n")
print(f"Area Analysis\n{'-' * 20}\n{area_analysis.head()}\n")
print(f"Office Analysis\n{'-' * 20}\n{office_analysis.head()}\n")

# 14. Statistical Outlier Removal (IQR)
Q1 = sold_with_rates_df['ClosePrice'].quantile(0.25)
Q3 = sold_with_rates_df['ClosePrice'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
sold_with_rates_df = sold_with_rates_df[
    (sold_with_rates_df['ClosePrice'] >= lower) &
    (sold_with_rates_df['ClosePrice'] <= upper)
]

# 15. Export Final Sold Dataset
sold_with_rates_df.to_csv('Final_Sold.csv', index=False)


# ==========================================
# PART 2: CRMLS LISTING DATA PIPELINE
# ==========================================

# 1. Data Aggregation
listing_data = glob.glob('**/CRMLSListing*.csv', recursive=True)
df_list = []
for file in listing_data:
    df = pd.read_csv(file, encoding='ISO-8859-1', low_memory=False)
    print(f"{file}: {len(df)}")
    df_list.append(df)
listing_df = pd.concat(df_list, ignore_index=True)

print(f'Rows after aggregation: {len(listing_df)}')

# 2. Filter to Residential
print(f"Rows before Residential Filter: {len(listing_df)}")
listing_df = listing_df[listing_df['PropertyType'] == "Residential"]
print(f"Rows after Residential Filter: {len(listing_df)}")

# 3. Drop columns with >90% missing values
print(f"Number of columns before dropping: {listing_df.shape[1]}")
threshold = 0.1 * len(listing_df)
listing_df.dropna(thresh=threshold, axis=1, inplace=True)
print(f"Number of columns after dropping: {listing_df.shape[1]}")

# 4. Outlier & Distribution Graphs
core_numeric_listing = ['ClosePrice', 'LivingArea', 'DaysOnMarket']
listing_df[core_numeric_listing].describe()

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 40))

for i, col in enumerate(core_numeric_listing):
    if col in listing_df.columns:
        sns.histplot(listing_df[col].dropna(), bins=50, kde=True, ax=axes[i, 0])
        axes[i, 0].set_title(f'{col} Distribution')
        sns.boxplot(x=listing_df[col].dropna(), ax=axes[i, 1])
        axes[i, 1].set_title(f'{col} Outliers')
plt.tight_layout()
plt.show()

# 5. Fetch and Merge FRED Mortgage Data
listing_df['year_month'] = pd.to_datetime(listing_df['ListingContractDate']).dt.to_period('M')
listing_with_rates_df = listing_df.merge(mortgage_monthly, on='year_month', how='left')

print(f"Null mortgage rates: {listing_with_rates_df['rate_30yr_fixed'].isnull().sum()}")
print("\nPreview of the enriched dataset:")
print(listing_with_rates_df[['ListingContractDate', 'year_month', 'ListPrice', 'rate_30yr_fixed']].head())

# 6. Date Conversions & Deduplication
for col in date_cols:
    if col in listing_with_rates_df.columns:
        listing_with_rates_df[col] = pd.to_datetime(listing_with_rates_df[col])

listing_with_rates_df = listing_with_rates_df.drop_duplicates(subset=['ListingKey'], keep='last')
listing_with_rates_df = listing_with_rates_df.drop(columns=[col for col in listing_with_rates_df.columns if col.endswith('.1')])

# 7. Drop Critical Missing Rows
must_columns_listing = ['ListingKey', 'ListPrice', 'City']
listing_with_rates_df = listing_with_rates_df.dropna(subset=must_columns_listing)

# 8. Impute Missing Values
for col in zero_fill_cols:
    if col in listing_with_rates_df.columns:
        listing_with_rates_df[col] = listing_with_rates_df[col].fillna(0)

for col in fill_no_cols:
    if col in listing_with_rates_df.columns:
        listing_with_rates_df[col] = listing_with_rates_df[col].fillna(False)

for col in cat_fill_cols:
    if col in listing_with_rates_df.columns:
        listing_with_rates_df[col] = listing_with_rates_df[col].fillna('Unknown')

for col in num_fill_cols:
    if col in listing_with_rates_df.columns:
        listing_with_rates_df[col] = listing_with_rates_df[col].fillna(listing_with_rates_df[col].median())

# 9. Clean Invalid Numeric Data
listing_with_rates_df = listing_with_rates_df.dropna(subset=['LivingArea'])
listing_with_rates_df = listing_with_rates_df[
    (listing_with_rates_df['ListPrice'] > 0) &
    (listing_with_rates_df['LivingArea'] > 0) &
    (listing_with_rates_df['DaysOnMarket'] >= 0) &
    (listing_with_rates_df['BedroomsTotal'] >= 0) &
    (listing_with_rates_df['BathroomsTotalInteger'] >= 0)
]

# 10. Data Consistency Flags
invalid_contracts = (
    listing_with_rates_df['ListingContractDate'] > listing_with_rates_df['PurchaseContractDate']
) & listing_with_rates_df['PurchaseContractDate'].notna()
listing_with_rates_df['contract_before_listing_flag'] = invalid_contracts
listing_with_rates_df['negative_timeline_flag'] = listing_with_rates_df['DaysOnMarket'] < 0

print(f"Contract Date Errors: {listing_with_rates_df['contract_before_listing_flag'].sum()}")
print(f"Negative Timeline Errors: {listing_with_rates_df['negative_timeline_flag'].sum()}")

# 11. Geographic Filters
print(f"Number of rows before cleaning: {listing_with_rates_df.shape[0]}")

listing_with_rates_df['missing_coords'] = listing_with_rates_df['Latitude'].isnull() | listing_with_rates_df['Longitude'].isnull()
listing_with_rates_df['sentinel_coords'] = (listing_with_rates_df['Latitude'] == 0) | (listing_with_rates_df['Longitude'] == 0)
listing_with_rates_df['cal_coords'] = listing_with_rates_df['Longitude'] > 0
listing_with_rates_df['is_california'] = (
    listing_with_rates_df['StateOrProvince'].isin(['CA', 'California'])
) | (
    listing_with_rates_df['PostalCode'].str.startswith('9')
)

listing_with_rates_df = listing_with_rates_df[
    (listing_with_rates_df['missing_coords'] == False) &
    (listing_with_rates_df['sentinel_coords'] == False) &
    (listing_with_rates_df['cal_coords'] == False) &
    (listing_with_rates_df['negative_timeline_flag'] == False) &
    (listing_with_rates_df['contract_before_listing_flag'] == False) &
    (listing_with_rates_df['is_california'] == True)
]
print(f'Number of rows after cleaning: {listing_with_rates_df.shape[0]}')

listing_with_rates_df = listing_with_rates_df[listing_with_rates_df['LivingArea'].isna() | (listing_with_rates_df['LivingArea'] > 0)]
listing_with_rates_df = listing_with_rates_df[listing_with_rates_df['Latitude'].isna() | listing_with_rates_df['Longitude'].between(-124, -114)]
listing_with_rates_df = listing_with_rates_df[listing_with_rates_df['DaysOnMarket'].isna() | (listing_with_rates_df['DaysOnMarket'] >= 0)]

# 12. Feature Engineering (Listing Dataset)
# NOTE: Sold-only features (ClosePrice-based) are not engineered here since listings
# include active and pending records. Features are derived from listing-stage fields.
listing_with_rates_df['Year'] = listing_with_rates_df['ListingContractDate'].dt.year
listing_with_rates_df['Month'] = listing_with_rates_df['ListingContractDate'].dt.month
listing_with_rates_df['YrMo'] = listing_with_rates_df['ListingContractDate'].dt.to_period('M').astype(str)
listing_with_rates_df['ListPricePerSqFt'] = listing_with_rates_df['ListPrice'] / listing_with_rates_df['LivingArea']

columns = ['Year', 'Month', 'YrMo', 'ListPricePerSqFt']
print(listing_with_rates_df[columns].head())

# 13. Segmented Analysis
listing_property_segment = listing_with_rates_df.groupby('PropertySubType').agg({
    'ListPrice': 'median',
    'ListPricePerSqFt': 'median',
    'DaysOnMarket': 'mean',
    'ListingKey': 'count'
}).rename(columns={'ListingKey': 'Active_Listings'}).reset_index()

print(f"Active Listings by Property Type\n{'*' * 30}\n{listing_property_segment}\n")

listing_county_segment = listing_with_rates_df.groupby('CountyOrParish').agg({
    'ListPrice': 'median',
    'DaysOnMarket': 'mean',
    'ListingKey': 'count'
}).rename(columns={'ListingKey': 'Active_Listings'}).reset_index()

print(f"Active Listings by County\n{'*' * 30}\n{listing_county_segment}\n")

# 14. Export Final Listing Dataset
listing_with_rates_df.to_csv('Final_Listing.csv', index=False)