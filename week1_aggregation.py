import pandas as pd
import numpy as np
import glob as glob

# Get all CRMLSSold and CRMlSListing files
sold_data = glob.glob('**/CRMLSSold*.csv',recursive=True)
listing_data = glob.glob('**/CRMLSListing*.csv', recursive=True)

sold_df_list = []
for file in sold_data:
    df = pd.read_csv(file, encoding='ISO-8859-1',low_memory=True)
    sold_df_list.append(df)
sold_df = pd.concat(sold_df_list, ignore_index=True)

listing_df_list = []
for file in listing_data:
    df = pd.read_csv(file, encoding='ISO-8859-1', low_memory=True)
    listing_df_list.append(df)
listing_df = pd.concat(listing_df_list, ignore_index=True)

# Filter to residential
print(f"Number of rows before filtering: {len(sold_df)}")
sold_df = sold_df[sold_df['PropertyType'] == 'Residential']
print(f"Number of rows after filtering: {len(sold_df)}")

print(f"Number of rows before filtering: {len(listing_df)}")
listing_df = listing_df[listing_df['PropertyType'] == 'Residential']
print(f"Number of rows after filtering: {len(listing_df)}")

# Save to csv 
sold_df.to_csv('CRMLS_Sold_Residential.csv', index=False)
listing_df.to_csv('CRMLS_Listing_Residential.csv', index=False)