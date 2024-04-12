# analysis.py
import matplotlib.pyplot as plt
import pandas as pd
from preprocess import read_and_preprocess, aggregate_data_across_regions

def smooth_data(df, column_name, window_size=5):
    """
    Applies a moving average to smooth out the given column in the DataFrame.
    
    Parameters:
    - df: pandas DataFrame containing the data.
    - column_name: the name of the column to smooth.
    - window_size: the number of periods to consider for the moving average.
    
    Returns:
    - A new DataFrame with the smoothed column.
    """
    df_smoothed = df.copy()
    df_smoothed[column_name] = df[column_name].rolling(window=window_size, min_periods=1, center=True).mean()
    return df_smoothed

def plot_prices(forecast_spot_prices_df, trading_price_rrp_df, window_size=5):
    """
    Plots the forecast spot prices and trading prices for different regions with smoothing applied.
    
    Parameters:
    - forecast_spot_prices_df: DataFrame containing forecast spot prices.
    - trading_price_rrp_df: DataFrame containing trading prices.
    - window_size: the number of periods used for smoothing.
    """
    for region in ['SA1', 'NSW1', 'QLD1', 'TAS1', 'VIC1']:
        forecast_spot_prices_df[region]['Timestamp'] = pd.to_datetime(forecast_spot_prices_df[region]['Timestamp'])
        trading_price_rrp_df[region]['Timestamp'] = pd.to_datetime(trading_price_rrp_df[region]['Timestamp'])

        plt.figure(figsize=(14, 8))

        # Apply smoothing directly before plotting
        smoothed_forecast = smooth_data(forecast_spot_prices_df[region], 'Forecast Spot Price_latest', window_size)
        smoothed_trading = smooth_data(trading_price_rrp_df[region], 'Trading Price RRP', window_size)

        plt.plot(smoothed_forecast['Timestamp'], smoothed_forecast['Forecast Spot Price_latest'], label='Forecast Spot Price (Latest, Smoothed)')
        plt.plot(smoothed_trading['Timestamp'], smoothed_trading['Trading Price RRP'], label='Trading Price RRP (Smoothed)')

        plt.title(f"Forecast Spot Prices and Trading Prices for {region} (Smoothed)")
        plt.xlabel("Timestamp")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_demand_supply(scheduled_demand_predispatch_df, cleared_supply_dispatch_df, window_size=5):
    """
    Plots the demand and supply data for different regions with smoothing applied.
    
    Parameters:
    - scheduled_demand_predispatch_df: DataFrame containing scheduled demand data.
    - cleared_supply_dispatch_df: DataFrame containing cleared supply data.
    - window_size: the number of periods used for smoothing.
    """
    for region in ['SA1', 'NSW1', 'QLD1', 'TAS1', 'VIC1']:
        scheduled_demand_predispatch_df[region]['Timestamp'] = pd.to_datetime(scheduled_demand_predispatch_df[region]['Timestamp'])
        cleared_supply_dispatch_df[region]['Timestamp'] = pd.to_datetime(cleared_supply_dispatch_df[region]['Timestamp'])

        plt.figure(figsize=(14, 8))

        # Apply smoothing directly before plotting
        smoothed_demand = smooth_data(scheduled_demand_predispatch_df[region], 'Scheduled Demand Pre-dispatch_latest', window_size)
        smoothed_supply = smooth_data(cleared_supply_dispatch_df[region], 'Cleared Supply Dispatch', window_size)

        plt.plot(smoothed_demand['Timestamp'], smoothed_demand['Scheduled Demand Pre-dispatch_latest'], label='Scheduled Demand Pre-dispatch (Latest, Smoothed)')
        plt.plot(smoothed_supply['Timestamp'], smoothed_supply['Cleared Supply Dispatch'], label='Cleared Supply Dispatch (Smoothed)')

        plt.title(f"Scheduled Demand and Cleared Supply for {region} (Smoothed)")
        plt.xlabel("Timestamp")
        plt.ylabel("Volume")
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_aggregated_prices(aggregated_forecast_prices, aggregated_trading_prices, window_size=5):
    """
    Plots aggregated forecast and trading prices with smoothing applied.
    
    Parameters:
    - aggregated_forecast_prices: DataFrame containing aggregated forecast prices.
    - aggregated_trading_prices: DataFrame containing aggregated trading prices.
    - window_size: the number of periods used for smoothing.
    """
    # Smooth the aggregated data
    smoothed_forecast_prices = smooth_data(aggregated_forecast_prices, 'Forecast Spot Price_latest', window_size)
    smoothed_trading_prices = smooth_data(aggregated_trading_prices, 'Trading Price RRP', window_size)

    plt.figure(figsize=(14, 8))
    plt.plot(smoothed_forecast_prices['Timestamp'], smoothed_forecast_prices['Forecast Spot Price_latest'], label='Aggregated Forecast Spot Price (Smoothed)')
    plt.plot(smoothed_trading_prices['Timestamp'], smoothed_trading_prices['Trading Price RRP'], label='Aggregated Trading Price RRP (Smoothed)')

    plt.title("Aggregated Forecast and Trading Prices (Smoothed)")
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_aggregated_demand_supply(aggregated_demand, aggregated_supply, window_size=5):
    """
    Plots aggregated demand and supply data with smoothing applied.
    
    Parameters:
    - aggregated_demand: DataFrame containing aggregated demand data.
    - aggregated_supply: DataFrame containing aggregated supply data.
    - window_size: the number of periods used for smoothing.
    """
    # Smooth the aggregated data
    smoothed_demand = smooth_data(aggregated_demand, 'Scheduled Demand Pre-dispatch_latest', window_size)
    smoothed_supply = smooth_data(aggregated_supply, 'Cleared Supply Dispatch', window_size)

    # Plotting
    plt.figure(figsize=(18, 10))
    plt.plot(smoothed_demand['Timestamp'], smoothed_demand['Scheduled Demand Pre-dispatch_latest'], label='Scheduled Demand (Latest, Smoothed)', color='red')
    plt.plot(smoothed_supply['Timestamp'], smoothed_supply['Cleared Supply Dispatch'], label='Cleared Supply (Smoothed)', color='purple')

    plt.title("Aggregated Demand and Supply (Smoothed)")
    plt.xlabel("Timestamp")
    plt.ylabel("Volume")
    plt.legend()
    plt.grid(True)
    plt.show()
    
# Example usage
directories = ['data/PredispatchIS_Reports', 'data/DispatchIS_Reports', 'data/TradingIS_Reports']
forecast_spot_prices_df, scheduled_demand_predispatch_df, cleared_supply_dispatch_df, trading_price_rrp_df = read_and_preprocess(directories)
aggregated_forecast_prices = aggregate_data_across_regions(forecast_spot_prices_df)
aggregated_trading_prices = aggregate_data_across_regions(trading_price_rrp_df)
aggregated_demand = aggregate_data_across_regions(scheduled_demand_predispatch_df)
aggregated_supply = aggregate_data_across_regions(cleared_supply_dispatch_df)

plot_aggregated_prices(aggregated_forecast_prices, aggregated_trading_prices)
plot_aggregated_demand_supply(aggregated_demand, aggregated_supply)
plot_prices(forecast_spot_prices_df, trading_price_rrp_df)
plot_demand_supply(scheduled_demand_predispatch_df, cleared_supply_dispatch_df)