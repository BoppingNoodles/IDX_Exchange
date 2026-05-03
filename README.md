
---

## 🔄 Data Pipeline

### 1. Data Extraction
- Pull MLS listing & sold data via API  
- Export monthly CSV files  

### 2. Data Aggregation
- Combine monthly datasets into unified tables  
- Filter to Residential properties only  

### 3. Data Validation & EDA
- Missing value analysis  
- Distribution analysis (price, DOM, etc.)  
- Dataset integrity checks  

### 4. Data Enrichment
- Merge 30-year mortgage rates from FRED  
- Convert weekly → monthly time series  

### 5. Data Cleaning
- Fix data types and formatting  
- Remove invalid values (e.g., negative prices)  
- Validate date consistency  
- Flag geographic anomalies  

---

## 📊 Feature Engineering

Key metrics created:

- Price Ratio = ClosePrice / OriginalListPrice  
- Price per Sq Ft = ClosePrice / LivingArea  
- Days on Market (DOM)  
- Listing → Contract Days  
- Contract → Close Days  
- Time-based features (Year, Month, YrMo)  

---

## 📈 Outlier Detection

- Implemented Interquartile Range (IQR) method  
- Flagged extreme values instead of deleting raw data  
- Created:
  - Full dataset with outlier flags  
  - Clean filtered dataset for analysis  

---

## 📊 Tableau Dashboards

### Market Analysis Dashboard
- Median home price trends  
- Days on market  
- Price ratios  
- New listings vs. closed sales  

### Competitive Analysis Dashboard
- Top agents & brokerages  
- Sales volume & transaction counts  
- Geographic heatmaps (ZIP-level)  

---

## 📝 Final Deliverables

- Tableau Dashboards (published)  
- 1-page Market Intelligence Report  
- 5-minute Presentation  

---

## 🔍 Example Insights

- Pricing trends across counties and time  
- Market competitiveness (above/below list price)  
- Supply vs. demand dynamics  
- Agent and brokerage performance  

---
