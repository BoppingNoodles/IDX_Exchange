import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime

def load_and_clean_data():
    print("Loading data...")
    # Get all CRMLS files (Listings and Sold)
    listing_files = glob.glob('**/CRMLSListing*.csv', recursive=True)
    sold_files = glob.glob('**/CRMLSSold*.csv', recursive=True)
    
    df_list = []
    
    # Load Listing Data
    for file in listing_files:
        try:
            df = pd.read_csv(file, encoding='ISO-8859-1', low_memory=False)
            df_list.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            
    # Load Sold Data
    for file in sold_files:
        try:
            df = pd.read_csv(file, encoding='ISO-8859-1', low_memory=False)
            df_list.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            
    if not df_list:
        print("No data files found. Please ensure CRMLS CSV files are in the directory.")
        return None

    df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df)} total rows.")

    # Filter to Residential
    df = df[df['PropertyType'] == 'Residential'].copy()
    print(f"Filtered to {len(df)} residential rows.")

    # Data Cleaning
    print("Cleaning data...")
    
    # Standardize Date Columns
    date_cols = ['CloseDate', 'ListingContractDate', 'PurchaseContractDate', 'ContractStatusChangeDate']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Deduplicate
    df = df.drop_duplicates(subset=['ListingKey'], keep='last')
    
    # Fill missing values
    zero_fill_cols = ['GarageSpaces', 'ParkingTotal', 'FireplacesTotal']
    for col in zero_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Filter invalid numeric values
    df = df[
        (df['ListPrice'] > 0) & 
        (df['LivingArea'] > 0)
    ]
    
    # If ClosePrice exists (for sold records), use it; otherwise, focus on ListPrice
    # But for the requested metrics, we definitely need ClosePrice for some.
    
    return df

def enrich_mortgage_rates(df):
    print("Enriching with mortgage rates...")
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
        mortgage = pd.read_csv(url, parse_dates=['observation_date'])
        mortgage.columns = ['date', 'rate_30yr_fixed']
        
        mortgage['year_month'] = mortgage['date'].dt.to_period('M')
        mortgage_monthly = mortgage.groupby('year_month')['rate_30yr_fixed'].mean().reset_index()
        
        # Use ListingContractDate for enrichment if CloseDate is missing
        df['year_month_key'] = df['ListingContractDate'].dt.to_period('M')
        df = df.merge(mortgage_monthly, left_on='year_month_key', right_on='year_month', how='left')
        df = df.drop(columns=['year_month_key', 'year_month'])
    except Exception as e:
        print(f"Warning: Could not enrich with mortgage rates: {e}")
    
    return df

def engineeer_metrics(df):
    print("Engineering metrics...")
    
    # Price Ratio (negotiation strength)
    # Using ClosePrice if available, else NaN
    if 'ClosePrice' in df.columns and 'OriginalListPrice' in df.columns:
        df['PriceRatio'] = df['ClosePrice'] / df['OriginalListPrice']
        
    # Close-to-Original-List Ratio
    if 'ClosePrice' in df.columns and 'OriginalListPrice' in df.columns:
        df['CloseToOriginalListRatio'] = df['ClosePrice'] / df['OriginalListPrice']

    # PPSF (Price Per Square Foot)
    # Use ClosePrice if available, else ListPrice
    if 'ClosePrice' in df.columns:
        df['PPSF'] = df['ClosePrice'] / df['LivingArea']
    else:
        df['PPSF'] = df['ListPrice'] / df['LivingArea']

    # YrMo (Year-Month)
    # Use CloseDate if available, else ListingContractDate
    if 'CloseDate' in df.columns:
        df['YrMo'] = df['CloseDate'].dt.to_period('M').astype(str)
        # Handle cases where CloseDate is NaT
        mask = df['YrMo'] == 'NaT'
        df.loc[mask, 'YrMo'] = df.loc[mask, 'ListingContractDate'].dt.to_period('M').astype(str)
    else:
        df['YrMo'] = df['ListingContractDate'].dt.to_period('M').astype(str)

    # Listing-to-Contract Days
    if 'PurchaseContractDate' in df.columns and 'ListingContractDate' in df.columns:
        df['ListingToContractDays'] = (df['PurchaseContractDate'] - df['ListingContractDate']).dt.days

    # Contract-to-Close Days
    if 'CloseDate' in df.columns and 'PurchaseContractDate' in df.columns:
        df['ContractToCloseDays'] = (df['CloseDate'] - df['PurchaseContractDate']).dt.days

    return df

def main():
    df = load_and_clean_data()
    if df is None:
        return

    df = enrich_mortgage_rates(df)
    df = engineeer_metrics(df)

    # Filter for valid metrics (some require closed sales)
    # We'll show the top of the dataframe with the new columns
    display_cols = [
        'ListingKey', 'PropertySubType', 'CountyOrParish', 'ListPrice', 'ClosePrice', 
        'PriceRatio', 'CloseToOriginalListRatio', 'PPSF', 'DaysOnMarket', 
        'YrMo', 'ListingToContractDays', 'ContractToCloseDays'
    ]
    
    # Only select columns that actually exist
    display_cols = [col for col in display_cols if col in df.columns]
    
    print("\n--- Sample Output Table (First 10 Rows) ---")
    print(df[display_cols].head(10).to_string())

    print("\n--- Segmented Summary Table: Grouped by PropertySubType ---")
    if 'PropertySubType' in df.columns:
        summary = df.groupby('PropertySubType').agg({
            'ListPrice': 'median',
            'PPSF': 'median',
            'DaysOnMarket': 'mean'
        }).reset_index()
        print(summary)

    print("\n--- Segmented Summary Table: Grouped by CountyOrParish ---")
    if 'CountyOrParish' in df.columns:
        summary_county = df.groupby('CountyOrParish').agg({
            'ListPrice': 'median',
            'PPSF': 'median',
            'DaysOnMarket': 'mean'
        }).reset_index()
        print(summary_county)

if __name__ == "__main__":
    main()
