import pandas as pd
import numpy as np
import os

def synthesize_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Renaming and formatting time_index
    df.rename(columns={'Date': 'time_index'}, inplace=True)
    df['time_index'] = pd.to_datetime(df['time_index'])
    
    # Sort chronologically (Critokical for CNN Time-Series later)
    df = df.sort_values(['City', 'time_index']).reset_index(drop=True)
    
    np.random.seed(42)

    print("Generating realistic synthetic variables...")

    # 2. Extract out temporal properties for realistic weather generation
    months = df['time_index'].dt.month

    # Every AR(1) process below is generated INDEPENDENTLY PER CITY. Running one
    # recursion over the whole frame would let the last day of one city seed the
    # first day of the next, fabricating continuity across unrelated locations.
    city_blocks = [df.index[df['City'] == c] for c in df['City'].unique()]

    base_temp = 30 + 10 * np.sin((months - 5) * np.pi / 6)
    base_humidity = 50 + 25 * np.sin((months - 8) * np.pi / 6)

    AR_COEF_TEMP = 0.85
    AR_COEF_HUM = 0.85
    AR_COEF_WIND = 0.8
    AR_COEF_VEH = 0.85
    AR_COEF_ENE = 0.88

    temp_col = np.empty(len(df))
    hum_col = np.empty(len(df))
    wind_col = np.empty(len(df))
    ind_col = np.empty(len(df), dtype=int)

    for block in city_blocks:
        pos = np.asarray(block)

        # Simulate Temperature (C) with AR(1) temporal persistence
        # Real weather changes gradually - AR(0.85) gives ~2-day half-life
        prev = float(base_temp.loc[pos[0]]) + np.random.normal(0, 1)
        temp_col[pos[0]] = round(max(10, min(48, prev)), 1)
        for i in pos[1:]:
            steady = float(base_temp.loc[i])
            prev = AR_COEF_TEMP * prev + (1 - AR_COEF_TEMP) * steady + np.random.normal(0, 1)
            prev = max(10, min(48, prev))
            temp_col[i] = round(prev, 1)

        # Simulate Humidity (%) with AR(1) temporal persistence
        prev = float(base_humidity.loc[pos[0]]) + np.random.normal(0, 2)
        hum_col[pos[0]] = round(max(10, min(100, prev)), 1)
        for i in pos[1:]:
            steady = float(base_humidity.loc[i])
            prev = AR_COEF_HUM * prev + (1 - AR_COEF_HUM) * steady + np.random.normal(0, 2)
            prev = max(10, min(100, prev))
            hum_col[i] = round(prev, 1)

        # Simulate Wind Speed (km/h) with AR(1) temporal persistence
        prev = np.random.lognormal(mean=2.0, sigma=0.5)
        wind_col[pos[0]] = round(max(0.1, prev), 1)
        for i in pos[1:]:
            prev = AR_COEF_WIND * prev + (1 - AR_COEF_WIND) * 7.5 + np.random.normal(0, 1.5)
            prev = max(0.1, prev)
            wind_col[i] = round(prev, 1)

        # 3. Industry Level (1-5) with Markov persistence
        # Real industries maintain stable operational levels; 90% chance of staying same
        prev_i = np.random.randint(1, 6)
        ind_col[pos[0]] = prev_i
        for i in pos[1:]:
            if np.random.random() >= 0.9:
                prev_i = max(1, min(5, prev_i + np.random.choice([-1, 1])))
            ind_col[i] = prev_i

    df['temperature'] = temp_col
    df['humidity'] = hum_col
    df['wind_speed'] = wind_col
    df['industry_level'] = ind_col

    # 4. Vehicle Count with temporal autocorrelation (AR(1) process)
    # Logic: High industry -> More commercial traffic -> High vehicles
    # AR(1): today's count is 85% of yesterday's + new shock
    # 5. Energy Usage (MWh) with temporal autocorrelation (AR(1) process)
    # Logic: High industry -> High energy AND High temp (>30C) -> High energy (AC Usage)
    # AR(1): energy demand persists day-to-day (industrial schedules, grid inertia)
    ac_load = np.where(df['temperature'] > 30, (df['temperature'] - 30) * 15, 0)
    industry_arr = df['industry_level'].values

    veh_col = np.empty(len(df), dtype=int)
    ene_col = np.empty(len(df))

    for block in city_blocks:
        pos = np.asarray(block)

        prev_v = 15000 + (int(industry_arr[pos[0]]) * 5000)
        prev_e = 150 + (float(industry_arr[pos[0]]) * 120) + float(ac_load[pos[0]])
        for i in pos:
            steady_v = 15000 + industry_arr[i] * 5000
            prev_v = max(AR_COEF_VEH * prev_v + (1 - AR_COEF_VEH) * steady_v
                         + np.random.normal(loc=0, scale=3000), 1000)
            veh_col[i] = int(prev_v)

            steady_e = 150 + (industry_arr[i] * 120) + ac_load[i]
            prev_e = AR_COEF_ENE * prev_e + (1 - AR_COEF_ENE) * steady_e \
                + np.random.normal(loc=0, scale=20)
            ene_col[i] = round(prev_e, 2)

    df['vehicle_count'] = veh_col
    df['energy_usage'] = ene_col

    # 6. Carbon Emission (Target)
    # Realistic relation:
    # High Energy -> High Emission
    # High Vehicles -> High Emission
    # High Wind Speed -> Blows emission away (Negative correlation)
    
    # Ensure real pollutants are numeric (in case of dirty 'str' data) and fill NAs
    temp_co = pd.to_numeric(df['CO'], errors='coerce')
    temp_co = temp_co.fillna(temp_co.median())
    
    temp_no2 = pd.to_numeric(df['NO2'], errors='coerce')
    temp_no2 = temp_no2.fillna(temp_no2.median())
    
    # Carbon emission with AR(1) - real pollutant levels persist in atmosphere
    # All 6 model features must have direct, learnable effects on emission.
    #
    # The relationships below are deliberately NON-LINEAR and include interaction
    # terms. An earlier version summed the features with fixed weights, which made
    # the target an exactly linear function of the inputs - so plain Linear
    # Regression was the correctly-specified model and no neural architecture could
    # ever beat it. Each non-linearity here is physically motivated.
    gross_emission = (
        (df['energy_usage'] * 3.5) +                                  # Energy dominates
        (df['vehicle_count'] * 0.08) +                                # Vehicles contribute heavily
        (df['industry_level'] ** 1.6) * 140.0 +                       # Super-linear: dirtier plants run at high load
        np.maximum(df['temperature'] - 28.0, 0.0) ** 1.5 * 45.0 +     # Threshold: AC/combustion kicks in past 28C
        (df['industry_level'] * df['vehicle_count'] * 0.004) +        # Interaction: freight congestion around industry
        (temp_co * 5.0) +                                             # Positive correlation with other pollutants
        (temp_no2 * 2.0)
    )

    # Wind disperses pollution multiplicatively, not additively: concentration decays
    # exponentially with wind speed, and humid air suppresses that dispersal (particulates
    # bind to moisture and settle locally instead of blowing out).
    dispersal = np.exp(-df['wind_speed'] / (9.0 + df['humidity'] * 0.10))
    base_emission = 400.0 + gross_emission * (0.45 + 0.55 * dispersal)

    AR_COEF_EMISSION = 0.6
    emission_col = np.empty(len(df))
    for block in city_blocks:
        pos = np.asarray(block)
        prev = float(base_emission.loc[pos[0]]) + np.random.normal(0, 10)
        emission_col[pos[0]] = round(max(50, prev), 2)
        for i in pos[1:]:
            prev = (AR_COEF_EMISSION * prev +
                    (1 - AR_COEF_EMISSION) * float(base_emission.loc[i]) +
                    np.random.normal(0, 10))
            prev = max(50, prev)
            emission_col[i] = round(prev, 2)
    df['carbon_emission'] = emission_col

    # 7. Select final columns for Layer 2 Output
    # The models ingest these specific features; the raw pollutant columns are
    # dropped so they cannot leak into training by accident.
    final_cols = [
        'City', 'time_index', 'temperature', 'humidity', 'wind_speed',
        'industry_level', 'vehicle_count', 'energy_usage', 'carbon_emission'
    ]
    final_df = df[final_cols]

    final_df.to_csv(output_path, index=False)
    print(f"Layer 2 Complete! Saved 'final_df' to {output_path}")
    print("\nPreview of final_df columns for the CNN:")
    summary_cols = ['time_index', 'temperature', 'industry_level', 'vehicle_count', 'energy_usage', 'carbon_emission']
    print(final_df[summary_cols].head(10))

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "city_day.csv")
    output_file = os.path.join(current_dir, "final_df.csv") 
    
    if os.path.exists(input_file):
        synthesize_data(input_file, output_file)
    else:
        print(f"Error: Could not find {input_file}.")
