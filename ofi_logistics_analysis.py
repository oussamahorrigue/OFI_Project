# loading the ofi_deliveries.csv dataset CSV and performing cleaning + key analyses.
# Outputs: cleaned CSV, summary tables, and several charts saved under /ofi_analysis_outputs/
# Note: This code runs in the notebook environment and will display key tables and figures to you.

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from IPython.display import display
# Helper to display DataFrames to the user (provided in this environment)

IN = 'ofi_deliveries.csv'
OUT_DIR = 'ofi_analysis_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Load data
df_raw = pd.read_csv(IN, dtype=str)
print("Raw shape:", df_raw.shape)
display(df_raw.head())

# 2) Initial data quality report
df = df_raw.copy()
df.columns = [c.strip() for c in df.columns]

# Count duplicates (exact row duplicates)
dup_count = df.duplicated().sum()
print(f"Exact duplicate rows: {dup_count}")

# Basic null counts
null_report = pd.DataFrame({
    'column': df.columns,
    'non_null_count': df.notnull().sum().values,
    'null_count': df.isnull().sum().values,
    'null_pct': (df.isnull().mean().values * 100)
}).sort_values('null_pct', ascending=False)
print("Data Quality - Nulls by Column:")
print(null_report)


# 3) Coerce numeric columns (replace comma decimal, strip)
num_cols = ['volume_kg', 'distance_km', 'delivery_time_hours', 'transport_cost_eur', 'delay_hours']
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c].str.replace(',', '.').str.strip(), errors='coerce')

# 4) Parse dates and times
if 'date' in df.columns:
    df['date_raw'] = df['date']
    # try common formats
    df['date'] = pd.to_datetime(df['date'], dayfirst=False, errors='coerce')

if 'time' in df.columns:
    df['time_raw'] = df['time']
    # try to parse times; keep as string if fails
    def try_time(x):
        try:
            return pd.to_datetime(x).time()
        except:
            return pd.NaT
    df['time_parsed'] = df['time'].apply(lambda x: try_time(x) if pd.notnull(x) else pd.NaT)

# 5) Drop exact duplicates (keep first)
before = df.shape[0]
df = df.drop_duplicates(keep='first').reset_index(drop=True)
after = df.shape[0]
print(f"Dropped {before-after} exact duplicate rows. New shape: {df.shape}")

# 6) Fix impossible numeric values (<=0)
for c in ['volume_kg', 'distance_km', 'delivery_time_hours', 'transport_cost_eur']:
    if c in df.columns:
        df.loc[df[c] <= 0, c] = np.nan

# 7) Normalize categorical fields
if 'delivery_status' in df.columns:
    df['delivery_status'] = df['delivery_status'].str.strip().str.title()
    df.loc[~df['delivery_status'].isin(['On-Time','Delayed']), 'delivery_status'] = np.nan

if 'product_grade' in df.columns:
    df['product_grade'] = df['product_grade'].str.strip().str.upper()
    df.loc[~df['product_grade'].isin(['A','B','C']), 'product_grade'] = df['product_grade']  # leave NaNs as-is

# 8) Flag missing important fields
for c in ['distance_km','transport_cost_eur','volume_kg','delivery_time_hours']:
    if c in df.columns:
        df[c + '_was_missing'] = df[c].isnull()

# 9) Imputation strategy (conservative medians)
# distance_km: median by origin_warehouse + destination_city, fallback overall median
if 'distance_km' in df.columns:
    med_route = df.groupby(['origin_warehouse','destination_city'])['distance_km'].median()
    def impute_distance(row):
        if pd.notnull(row['distance_km']):
            return row['distance_km']
        key = (row.get('origin_warehouse'), row.get('destination_city'))
        if key in med_route.index and not np.isnan(med_route.loc[key]):
            return med_route.loc[key]
        return df['distance_km'].median()
    df['distance_km_imp'] = df.apply(impute_distance, axis=1)
    df['distance_km_imputed_flag'] = df['distance_km'].isnull()

# transport_cost_eur: derive cost_per_km median by origin_warehouse + destination_country
if 'transport_cost_eur' in df.columns and 'distance_km_imp' in df.columns:
    df['cost_per_km'] = df['transport_cost_eur'] / df['distance_km_imp']
    # where transport_cost_eur is null, cost_per_km is NaN; compute median per route group
    median_costpkm = df.groupby(['origin_warehouse','destination_country'])['cost_per_km'].median()
    overall_median_costpkm = df['cost_per_km'].median()
    def impute_cost(row):
        if pd.notnull(row['transport_cost_eur']):
            return row['transport_cost_eur']
        key = (row.get('origin_warehouse'), row.get('destination_country'))
        if key in median_costpkm.index and not np.isnan(median_costpkm.loc[key]):
            return median_costpkm.loc[key] * row['distance_km_imp']
        return overall_median_costpkm * row['distance_km_imp']
    df['transport_cost_eur_imp'] = df.apply(impute_cost, axis=1)
    df['transport_cost_imputed_flag'] = df['transport_cost_eur'].isnull()

# volume_kg: median per client_name + product_type
if 'volume_kg' in df.columns:
    med_vol = df.groupby(['client_name','product_type'])['volume_kg'].median()
    def impute_vol(row):
        if pd.notnull(row['volume_kg']):
            return row['volume_kg']
        key = (row.get('client_name'), row.get('product_type'))
        if key in med_vol.index and not np.isnan(med_vol.loc[key]):
            return med_vol.loc[key]
        return df['volume_kg'].median()
    df['volume_kg_imp'] = df.apply(impute_vol, axis=1)
    df['volume_kg_imputed_flag'] = df['volume_kg'].isnull()

# delivery_time_hours: median by distance band
if 'delivery_time_hours' in df.columns and 'distance_km_imp' in df.columns:
    bins = [0,50,150,300,600,1200,10000]
    df['dist_band'] = pd.cut(df['distance_km_imp'], bins=bins)
    med_time = df.groupby('dist_band')['delivery_time_hours'].median()
    def impute_time(row):
        if pd.notnull(row['delivery_time_hours']):
            return row['delivery_time_hours']
        band = row['dist_band']
        if band in med_time.index and not np.isnan(med_time.loc[band]):
            return med_time.loc[band]
        return df['delivery_time_hours'].median()
    df['delivery_time_imp'] = df.apply(impute_time, axis=1)
    df['delivery_time_imputed_flag'] = df['delivery_time_hours'].isnull()

# 10) Derived metrics: cost per km and cost per kg (using imputed columns)
df['cost_per_km_imp'] = df['transport_cost_eur_imp'] / df['distance_km_imp']
df['cost_per_kg_imp'] = df['transport_cost_eur_imp'] / df['volume_kg_imp']

# 11) Save cleaned dataset
clean_path = os.path.join(OUT_DIR, 'ofi_deliveries_cleaned.csv')
df.to_csv(clean_path, index=False)
print("Cleaned dataset saved to:", clean_path)

# 12) Summary statistics (overall)
num_summary = df[['volume_kg_imp','distance_km_imp','delivery_time_imp','transport_cost_eur_imp','cost_per_km_imp','cost_per_kg_imp']].describe().transpose().reset_index()
num_summary.rename(columns={'index':'metric'}, inplace=True)
print("Numeric Summary (Imputed):")
print(num_summary)


# 13) Cost efficiency - by destination_country
if 'destination_country' in df.columns:
    cost_country = df.groupby('destination_country').agg(
        deliveries=('delivery_id','count'),
        total_cost=('transport_cost_eur_imp','sum'),
        avg_cost_per_km=('cost_per_km_imp','mean'),
        avg_cost_per_kg=('cost_per_kg_imp','mean'),
        avg_distance=('distance_km_imp','mean')
    ).reset_index().sort_values('avg_cost_per_km', ascending=False)
    cost_country_path = os.path.join(OUT_DIR, 'cost_by_country.csv')
    cost_country.to_csv(cost_country_path, index=False)
    print("Cost by Destination Country:")
    print(cost_country)
    print("Saved:", cost_country_path)

# 14) Top clients by total transport cost
if 'client_name' in df.columns:
    cost_client = df.groupby('client_name').agg(
        deliveries=('delivery_id','count'),
        total_cost=('transport_cost_eur_imp','sum'),
        total_volume=('volume_kg_imp','sum'),
        avg_cost_per_kg=('cost_per_kg_imp','mean')
    ).reset_index().sort_values('total_cost', ascending=False)
    client_path = os.path.join(OUT_DIR, 'cost_by_client.csv')
    cost_client.to_csv(client_path, index=False)
    print("Top Clients by Total Transport Cost (sorted):")
    print(cost_client.head(20))
    print("Saved:", client_path)

# 15) Delivery performance by country (delays)
if 'delivery_status' in df.columns and 'delay_hours' in df.columns:
    df['is_delayed'] = df['delivery_status'] == 'Delayed'
    perf_country = df.groupby('destination_country').agg(
        deliveries=('delivery_id','count'),
        delayed_count=('is_delayed','sum'),
        avg_delay_hours=('delay_hours','mean'),
        avg_delivery_time=('delivery_time_imp','mean')
    ).reset_index()
    perf_country['pct_delayed'] = perf_country['delayed_count'] / perf_country['deliveries'] * 100
    perf_country_path = os.path.join(OUT_DIR, 'delivery_performance_by_country.csv')
    perf_country.to_csv(perf_country_path, index=False)
    print("Delivery Performance by Country (sorted by % delayed):")
    print(perf_country.sort_values('pct_delayed', ascending=False))
    print("Saved:", perf_country_path)
else:
    print("delivery_status or delay_hours not available for full delay analysis. Partial results only.")

# 16) Identify high cost routes (top 5% cost_per_km)
route_cost = df.groupby(['origin_warehouse','destination_city']).agg(
    avg_cost_per_km=('cost_per_km_imp','mean'),
    deliveries=('delivery_id','count'),
    avg_distance=('distance_km_imp','mean')
).reset_index()
threshold = route_cost['avg_cost_per_km'].quantile(0.95)
high_cost_routes = route_cost[route_cost['avg_cost_per_km'] > threshold].sort_values('avg_cost_per_km', ascending=False)
high_cost_routes_path = os.path.join(OUT_DIR,'high_cost_routes.csv')
high_cost_routes.to_csv(high_cost_routes_path, index=False)
print("High-cost Routes (top 5% by avg cost/km):")
print(high_cost_routes)
print("Saved:", high_cost_routes_path)

# 17) Simple client segmentation (frequency/volume/cost quartiles)
if 'client_name' in df.columns:
    client = cost_client.copy()
    client['freq_q'] = pd.qcut(client['deliveries'].rank(method='first'), 4, labels=[1,2,3,4]).astype(int)
    client['vol_q'] = pd.qcut(client['total_volume'].rank(method='first'), 4, labels=[1,2,3,4]).astype(int)
    client['cost_q'] = pd.qcut(client['total_cost'].rank(method='first'), 4, labels=[1,2,3,4]).astype(int)
    client['segment_score'] = client['freq_q'] + client['vol_q'] + client['cost_q']
    def seg_label(s):
        if s >= 10: return 'High-Value'
        if s >=7: return 'Medium-Value'
        return 'Low-Value'
    client['segment'] = client['segment_score'].apply(seg_label)
    client_seg_path = os.path.join(OUT_DIR,'client_segments.csv')
    client.to_csv(client_seg_path, index=False)
    print("Client Segmentation (sample):")
    print(client.head(20))
    print("Saved:", client_seg_path)

# 18) Clustering destination cities (deliveries, avg_distance, total_volume)
if 'destination_city' in df.columns:
    city_agg = df.groupby(['destination_country','destination_city']).agg(
        deliveries=('delivery_id','count'),
        avg_distance=('distance_km_imp','mean'),
        total_volume=('volume_kg_imp','sum')
    ).reset_index().fillna(0)
    # scale and cluster (k=4)
    scaler = StandardScaler()
    X = scaler.fit_transform(city_agg[['deliveries','avg_distance','total_volume']])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)
    city_agg['cluster'] = kmeans.labels_
    city_clusters_path = os.path.join(OUT_DIR,'city_clusters.csv')
    city_agg.to_csv(city_clusters_path, index=False)
    print("City Clusters (sample):")
    print(city_agg.head(20))
    print("Saved:", city_clusters_path)

# 19) Simple visualizations (matplotlib) - Save PNGs
# Avg cost per km by country
if 'destination_country' in df.columns:
    try:
        plt.figure(figsize=(8,4))
        order = cost_country.sort_values('avg_cost_per_km', ascending=False)
        plt.bar(order['destination_country'], order['avg_cost_per_km'])
        plt.title('Avg Cost per km by Country')
        plt.ylabel('EUR per km')
        plt.xlabel('Destination Country')
        plt.tight_layout()
        fig1 = os.path.join(OUT_DIR,'avg_cost_per_km_by_country.png')
        plt.savefig(fig1)
        plt.close()
        print("Saved figure:", fig1)
    except Exception as e:
        print("Error saving avg cost figure:", str(e))

# Pct delayed by country
if 'pct_delayed' in locals():
    try:
        plt.figure(figsize=(8,4))
        order = perf_country.sort_values('pct_delayed', ascending=False)
        plt.bar(order['destination_country'], order['pct_delayed'])
        plt.title('Pct Delayed by Country (%)')
        plt.ylabel('% delayed')
        plt.xlabel('Destination Country')
        plt.tight_layout()
        fig2 = os.path.join(OUT_DIR,'pct_delayed_by_country.png')
        plt.savefig(fig2)
        plt.close()
        print("Saved figure:", fig2)
    except Exception as e:
        print("Error saving delay figure:", str(e))

# Top clients by total cost (bar)
try:
    topc = cost_client.head(15)
    plt.figure(figsize=(10,4))
    plt.bar(topc['client_name'], topc['total_cost'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 15 Clients by Total Transport Cost')
    plt.ylabel('Total Cost (EUR)')
    plt.tight_layout()
    fig3 = os.path.join(OUT_DIR,'top15_clients_total_cost.png')
    plt.savefig(fig3)
    plt.close()
    print("Saved figure:", fig3)
except Exception as e:
    print("Error saving top clients figure:", str(e))

# Scatter: distance vs delay_hours (if delay_hours present)
if 'delay_hours' in df.columns and df['delay_hours'].notnull().sum() > 0:
    try:
        s = df.dropna(subset=['distance_km_imp','delay_hours'])
        plt.figure(figsize=(6,4))
        plt.scatter(s['distance_km_imp'], s['delay_hours'], alpha=0.6)
        plt.title('Distance vs Delay Hours (scatter)')
        plt.xlabel('Distance (km)')
        plt.ylabel('Delay Hours')
        plt.tight_layout()
        fig4 = os.path.join(OUT_DIR,'distance_vs_delay_scatter.png')
        plt.savefig(fig4)
        plt.close()
        print("Saved figure:", fig4)
    except Exception as e:
        print("Error saving scatter:", str(e))
else:
    print("Not enough delay_hours data for scatter plot.")

print("\nAll outputs saved to:", OUT_DIR)
print("You can download key files (cleaned dataset and summaries):")
print(" - Cleaned dataset:", clean_path)
print(" - Cost by country CSV:", cost_country_path if 'cost_country_path' in locals() else 'N/A')
print(" - Delivery perf CSV:", perf_country_path if 'perf_country_path' in locals() else 'N/A')
print(" - High-cost routes CSV:", high_cost_routes_path if 'high_cost_routes_path' in locals() else 'N/A')


# Display a short executive summary as a DataFrame for quick viewing
exec_summary = pd.DataFrame([
    {"item":"Total deliveries (rows)","value":df.shape[0]},
    {"item":"Unique clients","value":df['client_name'].nunique() if 'client_name' in df.columns else 'N/A'},
    {"item":"Countries covered","value":df['destination_country'].nunique() if 'destination_country' in df.columns else 'N/A'},
    {"item":"Pct rows with any imputation","value": round((df[['distance_km_imputed_flag' if 'distance_km_imputed_flag' in df.columns else 'distance_km_imputed_flag']].any(axis=1).mean()*100) if 'distance_km_imputed_flag' in df.columns else 0,2)}
])
print("Quick Executive Summary:")
print(exec_summary)
# ============================================================
# 20) Correlation Visualizations for delay_hours vs key features
# ============================================================

import seaborn as sns

corr_outputs = []

def save_plot(fig, name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path)
    plt.close()
    print("Saved figure:", path)

# Helper to compute correlation safely
def safe_corr(a, b):
    if a.isnull().all() or b.isnull().all():
        return np.nan
    return a.corr(b)

# 1️⃣ delay_hours vs Transport Cost
if 'delay_hours' in df.columns and 'transport_cost_eur_imp' in df.columns:
    s = df.dropna(subset=['delay_hours','transport_cost_eur_imp'])
    r = safe_corr(s['delay_hours'], s['transport_cost_eur_imp'])
    plt.figure(figsize=(6,4))
    plt.scatter(s['transport_cost_eur_imp'], s['delay_hours'], alpha=0.6)
    z = np.polyfit(s['transport_cost_eur_imp'], s['delay_hours'], 1)
    p = np.poly1d(z)
    plt.plot(s['transport_cost_eur_imp'], p(s['transport_cost_eur_imp']), color='red')
    plt.title(f"Delay vs Transport Cost (r={r:.2f})")
    plt.xlabel("Transport Cost (€)")
    plt.ylabel("Delay (hours)")
    plt.tight_layout()
    save_plot(plt, "corr_delay_vs_transport_cost.png")
    corr_outputs.append(('transport_cost_eur', r))

# 2️⃣ delay_hours vs Volume
if 'volume_kg_imp' in df.columns:
    s = df.dropna(subset=['delay_hours','volume_kg_imp'])
    r = safe_corr(s['delay_hours'], s['volume_kg_imp'])
    plt.figure(figsize=(6,4))
    plt.scatter(s['volume_kg_imp'], s['delay_hours'], alpha=0.6)
    z = np.polyfit(s['volume_kg_imp'], s['delay_hours'], 1)
    p = np.poly1d(z)
    plt.plot(s['volume_kg_imp'], p(s['volume_kg_imp']), color='red')
    plt.title(f"Delay vs Volume (r={r:.2f})")
    plt.xlabel("Volume (kg)")
    plt.ylabel("Delay (hours)")
    plt.tight_layout()
    save_plot(plt, "corr_delay_vs_volume.png")
    corr_outputs.append(('volume_kg', r))

# 3️⃣ delay_hours vs Product Type
if 'product_type' in df.columns:
    s = df.dropna(subset=['delay_hours','product_type'])
    mean_enc = s.groupby('product_type')['delay_hours'].mean()
    encoded = s['product_type'].map(mean_enc)
    r = safe_corr(s['delay_hours'], encoded)
    plt.figure(figsize=(6,4))
    sns.boxplot(data=s, x='product_type', y='delay_hours')
    plt.title(f"Delay vs Product Type (r={r:.2f})")
    plt.tight_layout()
    save_plot(plt, "corr_delay_vs_product_type.png")
    corr_outputs.append(('product_type', r))

# 4️⃣ delay_hours vs Destination Country
if 'destination_country' in df.columns:
    s = df.dropna(subset=['delay_hours','destination_country'])
    mean_enc = s.groupby('destination_country')['delay_hours'].mean()
    encoded = s['destination_country'].map(mean_enc)
    r = safe_corr(s['delay_hours'], encoded)
    plt.figure(figsize=(7,4))
    sns.boxplot(data=s, x='destination_country', y='delay_hours')
    plt.title(f"Delay vs Destination Country (r={r:.2f})")
    plt.tight_layout()
    save_plot(plt, "corr_delay_vs_destination_country.png")
    corr_outputs.append(('destination_country', r))

# 5️⃣ delay_hours vs Distance
if 'distance_km_imp' in df.columns:
    s = df.dropna(subset=['delay_hours','distance_km_imp'])
    r = safe_corr(s['delay_hours'], s['distance_km_imp'])
    plt.figure(figsize=(6,4))
    plt.scatter(s['distance_km_imp'], s['delay_hours'], alpha=0.6)
    z = np.polyfit(s['distance_km_imp'], s['delay_hours'], 1)
    p = np.poly1d(z)
    plt.plot(s['distance_km_imp'], p(s['distance_km_imp']), color='red')
    plt.title(f"Delay vs Distance (r={r:.2f})")
    plt.xlabel("Distance (km)")
    plt.ylabel("Delay (hours)")
    plt.tight_layout()
    save_plot(plt, "corr_delay_vs_distance.png")
    corr_outputs.append(('distance_km', r))

# 6️⃣ delay_hours vs Origin Warehouse
if 'origin_warehouse' in df.columns:
    s = df.dropna(subset=['delay_hours','origin_warehouse'])
    mean_enc = s.groupby('origin_warehouse')['delay_hours'].mean()
    encoded = s['origin_warehouse'].map(mean_enc)
    r = safe_corr(s['delay_hours'], encoded)
    plt.figure(figsize=(8,4))
    sns.boxplot(data=s, x='origin_warehouse', y='delay_hours')
    plt.title(f"Delay vs Origin Warehouse (r={r:.2f})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(plt, "corr_delay_vs_origin_warehouse.png")
    corr_outputs.append(('origin_warehouse', r))

# 7️⃣ delay_hours vs Destination City
if 'destination_city' in df.columns:
    s = df.dropna(subset=['delay_hours','destination_city'])
    mean_enc = s.groupby('destination_city')['delay_hours'].mean()
    encoded = s['destination_city'].map(mean_enc)
    r = safe_corr(s['delay_hours'], encoded)
    plt.figure(figsize=(10,5))
    sns.boxplot(data=s, x='destination_city', y='delay_hours')
    plt.title(f"Delay vs Destination City (r={r:.2f})")
    plt.xticks(rotation=90)
    plt.tight_layout()
    save_plot(plt, "corr_delay_vs_destination_city.png")
    corr_outputs.append(('destination_city', r))

# 8️⃣ delay_hours vs Date
if 'date' in df.columns:
    s = df.dropna(subset=['delay_hours','date'])
    s['date'] = pd.to_datetime(s['date'], errors='coerce')
    s = s.dropna(subset=['date'])
    s['date_ordinal'] = s['date'].map(pd.Timestamp.toordinal)
    r = safe_corr(s['delay_hours'], s['date_ordinal'])
    avg_delay_by_date = s.groupby('date')['delay_hours'].mean()
    plt.figure(figsize=(8,4))
    plt.plot(avg_delay_by_date.index, avg_delay_by_date.values, marker='o')
    plt.title(f"Average Delay by Date (r={r:.2f})")
    plt.xlabel("Date")
    plt.ylabel("Average Delay (hours)")
    plt.tight_layout()
    save_plot(plt, "corr_delay_vs_date.png")
    corr_outputs.append(('date', r))

# 9️⃣ delay_hours vs Month (trend)
if 'date' in df.columns:
    s = df.dropna(subset=['delay_hours','date'])
    s['date'] = pd.to_datetime(s['date'], errors='coerce')
    s['year_month'] = s['date'].dt.to_period('M')
    avg_month = s.groupby('year_month')['delay_hours'].mean().reset_index()
    avg_month['year_month'] = avg_month['year_month'].dt.to_timestamp()
    r = safe_corr(avg_month['year_month'].map(pd.Timestamp.toordinal), avg_month['delay_hours'])
    plt.figure(figsize=(8,4))
    plt.plot(avg_month['year_month'], avg_month['delay_hours'], marker='o')
    plt.title(f"Average Monthly Delay (r={r:.2f})")
    plt.xlabel("Month")
    plt.ylabel("Average Delay (hours)")
    plt.tight_layout()
    save_plot(plt, "corr_delay_vs_month.png")
    corr_outputs.append(('month', r))

# 10️⃣ Summary of correlations
corr_summary = pd.DataFrame(corr_outputs, columns=['Variable','Correlation_r']).sort_values('Correlation_r', ascending=False)
corr_summary_path = os.path.join(OUT_DIR, 'delay_correlations_summary.csv')
corr_summary.to_csv(corr_summary_path, index=False)
print("\nCorrelation Summary:")
print(corr_summary)
print("Saved:", corr_summary_path)

