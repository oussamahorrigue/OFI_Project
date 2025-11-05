# 🧮 Data Analyst Case Study – Olam Food Ingredients (OFI)

## 🚚 Logistics Optimization & Cost Efficiency Analysis

### 📌 Author: _Oussama Horrigue_

### 📅 Duration: 48-hour Assessment Submission

---

## 🏢 Company Context

Olam Food Ingredients (OFI) is a global supplier of dairy ingredients, serving manufacturers, retailers, and distributors across Europe.  
Its logistics division manages deliveries from **central warehouses (Germany, France, Poland)** to customers in multiple European countries.

In recent months, OFI’s logistics operations have expanded significantly, raising concerns about **rising transport costs, route inefficiencies, and delivery delays**.

---

## 🎯 Business Challenge

OFI’s management team seeks **data-driven insights** to:

1. Identify cost-saving opportunities
2. Improve delivery performance and service reliability
3. Optimize route planning and warehouse allocation
4. Support strategic decision-making for client servicing

---

## 📊 Dataset Description

**File:** `ofi_deliveries.csv`

| Column                                    | Description                         |
| ----------------------------------------- | ----------------------------------- |
| `delivery_id`                             | Unique delivery identifier          |
| `date`, `time`                            | Delivery timestamp                  |
| `origin_warehouse`                        | Source warehouse (DE, FR, PL)       |
| `destination_city`, `destination_country` | Destination information             |
| `client_name`                             | Customer ID (Client_1–Client_35)    |
| `product_type`                            | Dairy product (SMP, Butter, Cheese) |
| `product_grade`                           | Product quality grade (A–C)         |
| `volume_kg`                               | Weight delivered                    |
| `distance_km`                             | Distance covered                    |
| `delivery_time_hours`                     | Delivery duration                   |
| `transport_cost_eur`                      | Total transport cost                |
| `delivery_status`                         | On-time / Delayed                   |
| `delay_hours`                             | Delay duration (hours)              |

⚠️ **Data Quality Notes:**  
The dataset includes missing values, duplicates, date/time inconsistencies, and potential cost outliers.  
Handling these appropriately was a key focus of the analysis.

---

## 🧠 Project Objectives

### 1. Data Preparation & Exploration

- Identify missing, duplicated, and inconsistent records
- Clean and prepare data for analysis
- Provide descriptive statistics and data quality documentation

### 2. Cost Efficiency Analysis

- Analyze transport costs by **country, client, product type, and route**
- Compute **cost per km** and **cost per kg** metrics
- Detect high-cost routes and anomalies

### 3. Delivery Performance Analysis

- Evaluate on-time delivery performance
- Identify delay patterns by region, product, distance, and warehouse
- Quantify relationships between **delay_hours** and key variables

### 4. Geographic & Routing Analysis

- Assess route coverage and distance efficiency
- Identify clustering opportunities for cost reduction
- Evaluate warehouse-to-destination optimization

### 5. Client Segmentation

- Group clients by **volume, frequency, and profitability**
- Distinguish high-value vs. high-cost clients
- Support differentiated service strategies

### 6. Strategic Recommendations

- Deliver 5–7 actionable insights with potential cost or efficiency impacts

---

## 🧩 Repository Structure

```
OFI_Project/
│
├── ofi_deliveries.csv                # Raw dataset
├── ofi_logistics_analysis.py         # Main Python analysis script
├── ofi_analysis_outputs/             # Cleaned data, visualizations, summaries
│   ├── ofi_deliveries_cleaned.csv
│   ├── cost_by_country.csv
│   ├── client_segments.csv
│   ├── delay_vs_*.png                (correlation plots)
│   ├── summary_tables.csv
│   └── other result files
└── README.md                         # Project documentation
```

### Setup & Execution

#### Prerequisites

- Python 3.8+
- Required libraries:  
  `pip install pandas numpy matplotlib seaborn scikit-learn scipy`

#### How to Run

### 1. Clone the repository:

`git clone https://github.com/https://github.com/oussamahorrigue/OFI_Project.git`

`cd OFI_Project`

### 2. Run the main analysis script:

`python ofi_logistics_analysis.py`

### 3. All generated outputs (cleaned dataset, visuals, and reports) will be stored in:

`/ofi_analysis_outputs/`

### Insights Summary

![Insights Summary Table](/Insights_summary_table.png)

### Key Findings

#### 1. Delays are trending upward over time (correlation between delay_hours and month was about 0.31) — likely linked to growing delivery volume or capacity strain.

#### 2. Distance and country moderately affect delays (delay_hours vs distance_km → correlation = 0.12) especially on longer cross-border routes.

#### 3. Warehouse and city-level differences suggest local process inefficiencies.

#### 4. Cost per km and per kg reveal clear outliers — potential route optimization opportunities.

#### 5. Client segmentation shows a few clients drive a large portion of costs — high-priority for contract renegotiation.

### Recommendations

![Recommendation Table](/recommendation_table.png)

### Methodology Summary

#### 1. Data Cleaning:

- Removed duplicates and handled missing values using median/mode imputation.
- Fixed inconsistent date/time formats and numeric conversions.
- Normalized categorical fields and filtered out invalid entries.

#### 2. Exploratory Data Analysis:

- Generated descriptive statistics and correlation matrices.
- Assessed performance across countries, clients, and routes.

#### 3. Visualization:

- Created multiple scatter, box, and bar plots.
- Added correlation coefficients to highlight relationships.

#### 4. Segmentation & Optimization:

- Applied K-Means clustering to identify client and city patterns.
- Flagged high-cost routes and outliers for optimization.

### Deliverables

Python analysis script: `ofi_logistics_analysis.py`

Cleaned dataset: `ofi_deliveries_cleaned.csv`

Output reports and visuals in `/ofi_analysis_outputs/`

This `README.md` — documentation for reproducibility

## Conclusion

This project demonstrates a structured, analytical approach to logistics optimization, balancing technical accuracy with business insight.
It provides actionable findings to help reduce cost, improve delivery reliability, and enhance strategic decision-making at OFI.
