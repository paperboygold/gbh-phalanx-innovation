import matplotlib
matplotlib.use('GTK3Agg')  # You can also try 'Qt5Agg' or 'GTK3Agg'
import matplotlib.pyplot as plt
import pandas as pd
import os

def read_and_preprocess(directories):
    forecast_spot_prices = {'SA1': [], 'NSW1': [], 'QLD1': [], 'TAS1': [], 'VIC1': []}
    scheduled_demand_predispatch = {'SA1': [], 'NSW1': [], 'QLD1': [], 'TAS1': [], 'VIC1': []}
    cleared_supply_dispatch = {'SA1': [], 'NSW1': [], 'QLD1': [], 'TAS1': [], 'VIC1': []}
    trading_price_rrp = {'SA1': [], 'NSW1': [], 'QLD1': [], 'TAS1': [], 'VIC1': []}
    timestamps_forecast = {'SA1': [], 'NSW1': [], 'QLD1': [], 'TAS1': [], 'VIC1': []}
    timestamps_demand = {'SA1': [], 'NSW1': [], 'QLD1': [], 'TAS1': [], 'VIC1': []}
    timestamps_supply = {'SA1': [], 'NSW1': [], 'QLD1': [], 'TAS1': [], 'VIC1': []}
    timestamps_trading = {'SA1': [], 'NSW1': [], 'QLD1': [], 'TAS1': [], 'VIC1': []}

    for directory_path in directories:
        for filename in os.listdir(directory_path):
            if filename.endswith('.CSV'):
                file_path = os.path.join(directory_path, filename)
                with open(file_path, 'r') as file:
                    for line in file:
                        parts = line.strip().split(',')
                        if line.startswith('D,'):
                            if directory_path.endswith('PredispatchIS_Reports'):
                                if parts[1] == 'PREDISPATCH' and parts[2] == 'REGION_PRICES':
                                    region = parts[6].strip('"')  # Extract region from REGION_PRICES
                                    timestamp = parts[28].strip('"')  # Extract timestamp for REGION_PRICES
                                    try:
                                        forecast_spot_prices[region].append(float(parts[9]))
                                        timestamps_forecast[region].append(timestamp)
                                    except (ValueError, IndexError, KeyError):
                                        print(f"Skipping line due to error in file {filename}: {line}")
                                elif parts[1] == 'PREDISPATCH' and parts[2] == 'REGION_SOLUTION':
                                    region = parts[6].strip('"')  # Extract region from REGION_SOLUTION
                                    timestamp = parts[66].strip('"')  # Extract timestamp for REGION_SOLUTION
                                    try:
                                        scheduled_demand_predispatch[region].append(float(parts[68]))
                                        timestamps_demand[region].append(timestamp)
                                    except (ValueError, IndexError, KeyError):
                                        print(f"Skipping line due to error in file {filename}: {line}")
                            elif directory_path.endswith('DispatchIS_Reports'):
                                region = parts[6].strip('"')  # Extract region from DISPATCH reports
                                timestamp = parts[4].strip('"')  # Extract timestamp for DISPATCH reports
                                if parts[1] == 'DISPATCH' and parts[2] == 'REGIONSUM':
                                    try:
                                        cleared_supply_dispatch[region].append(float(parts[69]))
                                        timestamps_supply[region].append(timestamp)
                                    except (ValueError, IndexError, KeyError):
                                        print(f"Skipping line due to error in file {filename}: {line}")
                            elif directory_path.endswith('TradingIS_Reports'):
                                region = parts[6].strip('"')  # Extract region from TRADING reports
                                timestamp = parts[4].strip('"')  # Extract timestamp for TRADING reports
                                if parts[1] == 'TRADING' and parts[2] == 'PRICE':
                                    try:
                                        trading_price_rrp[region].append(float(parts[8]))
                                        timestamps_trading[region].append(timestamp)
                                    except (ValueError, IndexError, KeyError):
                                        print(f"Skipping line due to error in file {filename}: {line}")

    # Convert lists to DataFrames and sort them
    forecast_spot_prices_df = {}
    scheduled_demand_predispatch_df = {}
    cleared_supply_dispatch_df = {}
    trading_price_rrp_df = {}

    for region in ['SA1', 'NSW1', 'QLD1', 'TAS1', 'VIC1']:
        forecast_spot_prices_df[region] = pd.DataFrame({'Timestamp': timestamps_forecast[region], 'Forecast Spot Price': forecast_spot_prices[region]}).sort_values(by='Timestamp')
        scheduled_demand_predispatch_df[region] = pd.DataFrame({'Timestamp': timestamps_demand[region], 'Scheduled Demand Pre-dispatch': scheduled_demand_predispatch[region]}).sort_values(by='Timestamp')
        cleared_supply_dispatch_df[region] = pd.DataFrame({'Timestamp': timestamps_supply[region], 'Cleared Supply Dispatch': cleared_supply_dispatch[region]}).sort_values(by='Timestamp')
        trading_price_rrp_df[region] = pd.DataFrame({'Timestamp': timestamps_trading[region], 'Trading Price RRP': trading_price_rrp[region]}).sort_values(by='Timestamp')

    return forecast_spot_prices_df, scheduled_demand_predispatch_df, cleared_supply_dispatch_df, trading_price_rrp_df

def plot_data(forecast_spot_prices_df, scheduled_demand_predispatch_df, cleared_supply_dispatch_df, trading_price_rrp_df):
    for region in ['SA1', 'NSW1', 'QLD1', 'TAS1', 'VIC1']:
        # Convert 'Timestamp' columns to datetime for better plotting
        forecast_spot_prices_df[region]['Timestamp'] = pd.to_datetime(forecast_spot_prices_df[region]['Timestamp'])
        scheduled_demand_predispatch_df[region]['Timestamp'] = pd.to_datetime(scheduled_demand_predispatch_df[region]['Timestamp'])
        cleared_supply_dispatch_df[region]['Timestamp'] = pd.to_datetime(cleared_supply_dispatch_df[region]['Timestamp'])
        trading_price_rrp_df[region]['Timestamp'] = pd.to_datetime(trading_price_rrp_df[region]['Timestamp'])

        # Apply exponential moving average
        span = 20  # Adjust span as needed for smoothing
        forecast_spot_prices_df[region].loc[:, 'Smoothed Forecast Spot Price'] = forecast_spot_prices_df[region]['Forecast Spot Price'].ewm(span=span).mean()
        scheduled_demand_predispatch_df[region].loc[:, 'Smoothed Scheduled Demand Pre-dispatch'] = scheduled_demand_predispatch_df[region]['Scheduled Demand Pre-dispatch'].ewm(span=span).mean()
        cleared_supply_dispatch_df[region].loc[:, 'Smoothed Cleared Supply Dispatch'] = cleared_supply_dispatch_df[region]['Cleared Supply Dispatch'].ewm(span=span).mean()
        trading_price_rrp_df[region].loc[:, 'Smoothed Trading Price RRP'] = trading_price_rrp_df[region]['Trading Price RRP'].ewm(span=span).mean()

        # Plotting
        plt.figure(figsize=(14, 8))

        # Plot each DataFrame with smoothed data
        plt.plot(forecast_spot_prices_df[region]['Timestamp'], forecast_spot_prices_df[region]['Smoothed Forecast Spot Price'], label='Forecast Spot Price')
        plt.plot(scheduled_demand_predispatch_df[region]['Timestamp'], scheduled_demand_predispatch_df[region]['Smoothed Scheduled Demand Pre-dispatch'], label='Scheduled Demand Pre-dispatch')
        plt.plot(cleared_supply_dispatch_df[region]['Timestamp'], cleared_supply_dispatch_df[region]['Smoothed Cleared Supply Dispatch'], label='Cleared Supply Dispatch')
        plt.plot(trading_price_rrp_df[region]['Timestamp'], trading_price_rrp_df[region]['Smoothed Trading Price RRP'], label='Trading Price RRP')

        plt.title(f"Energy Market Data for {region}")
        plt.xlabel("Timestamp")
        plt.ylabel("Values")
        plt.legend()
        plt.grid(True)
        plt.show()

# Example Usage
directories = ['data/PredispatchIS_Reports', 'data/DispatchIS_Reports', 'data/TradingIS_Reports']
forecast_spot_prices_df, scheduled_demand_predispatch_df, cleared_supply_dispatch_df, trading_price_rrp_df = read_and_preprocess(directories)

# Print the .head() of each DataFrame for each region
for region in ['SA1', 'NSW1', 'QLD1', 'TAS1', 'VIC1']:
    print(f"Current region: {region}")
    print(f"\n{region} - Trading Spot Price 5 min RRP DataFrame:")
    print(trading_price_rrp_df[region].head(10))
    print(f"{region} - Forecast Spot Prices DataFrame:")
    print(forecast_spot_prices_df[region].head(10))
    print(f"\n{region} - Scheduled Demand Pre-dispatch DataFrame:")
    print(scheduled_demand_predispatch_df[region].head(10))
    print(f"\n{region} - Cleared Supply Dispatch DataFrame:")
    print(cleared_supply_dispatch_df[region].head(10))

# Call the plotting function
plot_data(forecast_spot_prices_df, scheduled_demand_predispatch_df, cleared_supply_dispatch_df, trading_price_rrp_df)