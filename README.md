# 🏡 IDX Exchange MLS Analytics Project

A full-stack data analytics pipeline and dashboard project built during the IDX Exchange Data Analyst Internship Program. This project transforms raw MLS transaction data into actionable housing market insights using Python and Tableau.

---

## 📌 Overview

This project follows a 12-week structured analytics pipeline covering:

- Data extraction from MLS APIs  
- Data cleaning and validation  
- Feature engineering and market metric creation  
- Outlier detection and statistical filtering  
- Interactive dashboard development in Tableau  
- Final market intelligence reporting  

The end result is a set of professional-grade real estate analytics dashboards and a data-driven market report.

---

## 🎯 Project Objectives

- Understand how real estate datasets are produced and structured  
- Build end-to-end data pipelines using Python (Pandas)  
- Develop key housing market indicators  
- Create interactive Tableau dashboards  
- Communicate insights through data storytelling  

---

## 🛠️ Tech Stack

- Python (Pandas) – data processing & analysis  
- Tableau Public – dashboard visualization  
- CoreLogic Trestle API – MLS data source  
- FRED API – mortgage rate enrichment  

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
