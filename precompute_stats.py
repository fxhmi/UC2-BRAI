import pandas as pd
import json

print("📊 Pre-computing route statistics...")

# Load the big CSV
df = pd.read_csv('data/ridership_prepared_hourly.csv')
df['route_no'] = df['route_no'].astype(str)

# Compute all needed statistics
stats = {}

for route in df['route_no'].unique():
    route_data = df[df['route_no'] == route]
    
    stats[route] = {
        'avg': float(route_data['ridership_total'].mean()),
        'std': float(route_data['ridership_total'].std()),
        'max': float(route_data['ridership_total'].max()),
        'min': float(route_data['ridership_total'].min()),
        'depot': int(route_data['depot'].mode()[0]),
        'hourly': {},
        'dow': {}
    }
    
    # Hourly averages
    hourly = route_data.groupby('hour')['ridership_total'].mean()
    for hour, avg in hourly.items():
        stats[route]['hourly'][str(hour)] = float(avg)
    
    # Day of week averages
    dow = route_data.groupby('day_of_week')['ridership_total'].mean()
    for day, avg in dow.items():
        stats[route]['dow'][str(day)] = float(avg)

# Save as JSON (much smaller!)
with open('data/route_statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"✅ Saved statistics for {len(stats)} routes")

import os
json_size = os.path.getsize('data/route_statistics.json') / (1024 * 1024)
print(f"📦 JSON size: {json_size:.2f} MB")

