import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import os

def run_eda(input_path):
    print(f"Loading data from {input_path} for EDA...")
    df = pd.read_csv(input_path)
    df['time_index'] = pd.to_datetime(df['time_index'])

    # Plot the SAME city the model is actually trained on (Layer 4 selects the city
    # with the most records). Picking unique()[0] here could describe a different city
    # than the one the reported metrics come from.
    city_name = df['City'].value_counts().idxmax()
    city_df = df[df['City'] == city_name].sort_values('time_index').head(365)  # First year

    # Set up the plot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Layer 3: Exploratory Data Analysis (CNN Features)', fontsize=16)
    
    # Plot 1: Time Series of Carbon Emissions
    axes[0, 0].plot(city_df['time_index'], city_df['carbon_emission'], color='red')
    axes[0, 0].set_title(f'1-Year Carbon Emission Trend in {city_name}')
    axes[0, 0].set_ylabel('Carbon Emission')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: AC Load Simulation (Temperature vs Energy)
    axes[0, 1].scatter(df['temperature'], df['energy_usage'], alpha=0.3, color='orange')
    axes[0, 1].set_title('Temperature vs Energy Usage (Notice the AC Spike > 30C)')
    axes[0, 1].set_xlabel('Temperature (C)')
    axes[0, 1].set_ylabel('Energy Usage (MWh)')
    
    # Plot 3: Industry Level vs Vehicles
    axes[1, 0].scatter(df['industry_level'], df['vehicle_count'], alpha=0.3, color='blue')
    axes[1, 0].set_title('Industry Level vs Vehicle Count')
    axes[1, 0].set_xlabel('Industry Level (1-5)')
    axes[1, 0].set_ylabel('Number of Vehicles')
    
    # Plot 4: Wind Speed vs Carbon Emission
    axes[1, 1].scatter(df['wind_speed'], df['carbon_emission'], alpha=0.3, color='green')
    axes[1, 1].set_title('Wind Speed vs Carbon Emission (Dispersal Effect)')
    axes[1, 1].set_xlabel('Wind Speed (km/h)')
    axes[1, 1].set_ylabel('Carbon Emission')
    
    plt.tight_layout()
    
    # Save the plot explicitly so we can view it
    output_png = os.path.join(os.path.dirname(input_path), "eda_results.png")
    plt.savefig(output_png)
    print(f"EDA Visualization successfully saved to: {output_png}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "final_df.csv")
    
    if os.path.exists(input_file):
        run_eda(input_file)
    else:
        print(f"Error: Could not find {input_file}. Please run Layer 2 first.")
