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

````
```plaintext
OFI_Project/
│
├── ofi_deliveries.csv                        # Raw dataset
├── ofi_logistics_analysis.py                 # Main Python analysis script
├── recommendation_table.png                  # Recommendation summary visualization
├── Insights_summary_table.png                # Key insights summary table
│
├── ofi_analysis_outputs/                     # Cleaned data, visualizations, summaries
│   ├── avg_cost_per_km_by_country.png
│   ├── city_clusters.csv
│   ├── client_segments.csv
│   ├── corr_delay_vs_date.png
│   ├── corr_delay_vs_destination_city.png
│   ├── corr_delay_vs_destination_country.png
│   ├── corr_delay_vs_distance.png
│   ├── corr_delay_vs_month.png
│   ├── corr_delay_vs_origin_warehouse.png
│   ├── corr_delay_vs_product_type.png
│   ├── corr_delay_vs_transport_cost.png
│   ├── corr_delay_vs_volume.png
│   ├── cost_by_client.csv
│   ├── cost_by_country.csv
│   ├── delay_correlations_summary.csv
│   ├── delivery_performance_by_country.csv
│   ├── distance_vs_delay_scatter.png
│   ├── high_cost_routes.csv
│   ├── ofi_deliveries_cleaned.csv
│   ├── top15_clients_total_cost.png
│
└── README.md                                 # Project documentation

````

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

The first analysis i made focused on examining whether transportation costs vary by destination country. To explore this, I created a chart showing the average transport cost per kilometer by country, as illustrated in the figure below.

![Average cost per km by country](/ofi_analysis_outputs/avg_cost_per_km_by_country.png)

### Interpretation

- Cost Efficiency:
  Average cost per km is quite consistent across countries (≈ €1.03–€1.05/km), suggesting that transport rates are standardized across routes and logistics providers.

- Delay Impact:
  Countries with longer distances (e.g., Spain and Italy) show higher average delays (~2 hours), which is expected due to route length and border crossings.

- Operational Insight:
  Countries like Germany and the Netherlands show lower average delay hours and strong delivery volumes, indicating efficient domestic logistics and possibly better road networks or warehouse proximity.

### Conclusion

- The cost efficiency per km is stable, meaning pricing strategies are likely optimized.

- Delay issues are more related to distance and route complexity than cost per km — this indicates operational (not financial) inefficiencies cause delays.

- Further route-level or warehouse-level analysis could reveal where bottlenecks occur.

The second analysis i made aimed to understand how much each client spends on transportation. I wanted to identify whether certain clients account for disproportionately high logistics costs, which could help pinpoint potential inefficiencies or cost concentration. To explore this, I created a chart showing the top 15 clients by total transport cost, as illustrated in the figure below.

![top 15 clients total cost](/ofi_analysis_outputs/top15_clients_total_cost.png)

### Interpretation

- Cost Drivers:
  The highest-spending clients (Client_21, 18, 25, 24) correspond to high-volume, frequent deliveries, suggesting bulk clients or key accounts driving cost concentration.

- Delays:
  Average delay hours for top clients hover around 1.2–1.7 hours, with no major outliers — meaning delays are evenly distributed across clients, not client-specific.

- Operational Focus:
  Since volume and delay don’t show a strong link, client-specific issues are unlikely the main delay cause.
  Instead, geographic or route-level inefficiencies appear more relevant.

**Further analysis was necessary, as neither cost per kilometer nor client behavior adequately explained the delivery delays. Therefore, I conducted a correlation analysis between delivery delays and multiple operational parameters to identify which factors have the greatest impact on delay performance as shown in the table below, which includes variables such as transport_cost_eur, volume_kg, product_type, destination_country, distance_km, origin_warehouse, destination_city, date, and month, along with their corresponding correlation coefficients. These coefficients indicate the strength and direction of the relationship between each variable and delivery delays where a value close to 0 signifies no relationship, and a value close to 1 (or -1) represents a strong positive (or negative) relationship.**

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
