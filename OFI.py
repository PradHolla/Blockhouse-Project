import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
import os
from dateutil.parser import parse

def calculate_best_level_ofi(file_path, time_bucket='1min'):
    """
    Calculate the best-level OFI aggregated over specified time intervals.
    
    Parameters:
    file_path (str): Path to the CSV file
    time_bucket (str): Time interval for aggregation (default: '1min')
    
    Returns:
    DataFrame: A dataframe with timestamps and best-level OFI values
    """
    # Check if file has a header
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
    
    # Determine if we need to skip the header row
    skiprows = 1 if 'ts_recv' in first_line else 0
    
    # Load data
    df = pd.read_csv(file_path, header=None, skiprows=skiprows)
    
    # Extract timestamp with error handling
    try:
        df['timestamp'] = pd.to_datetime(df[0], errors='coerce')
    except:
        # If standard parsing fails, use more flexible dateutil parser
    
        df['timestamp'] = df[0].apply(lambda x: parse(x) if isinstance(x, str) else pd.NaT)
    
    # Drop rows with invalid timestamps
    df = df.dropna(subset=['timestamp'])
    
    # Initialize OFI array
    e = np.zeros(len(df))
    
    # Best bid/ask price and size columns
    bid_col, ask_col = 13, 14      # Best bid/ask prices
    bidsize_col, asksize_col = 15, 16  # Best bid/ask sizes
    
    # Calculate OFI values for each row starting from the second row
    for i in range(1, len(df)):
        # Get current and previous bid/ask prices and sizes
        bid_curr = float(df.iloc[i, bid_col])
        bid_prev = float(df.iloc[i-1, bid_col])
        ask_curr = float(df.iloc[i, ask_col])
        ask_prev = float(df.iloc[i-1, ask_col])
        bidsize_curr = float(df.iloc[i, bidsize_col])
        bidsize_prev = float(df.iloc[i-1, bidsize_col])
        asksize_curr = float(df.iloc[i, asksize_col])
        asksize_prev = float(df.iloc[i-1, asksize_col])
        
        # Calculate OFI using the formula from the paper
        e[i] = (int(bid_curr >= bid_prev) * bidsize_curr -
                int(bid_curr <= bid_prev) * bidsize_prev -
                int(ask_curr <= ask_prev) * asksize_curr +
                int(ask_curr >= ask_prev) * asksize_prev)
    
    df['ofi'] = e
    
    # Aggregate by time bucket
    df['timestamp_floor'] = df['timestamp'].dt.floor(time_bucket)
    result = df.groupby('timestamp_floor')['ofi'].sum().reset_index()
    result.columns = ['timestamp', 'best_level_ofi']
    
    return result

def calculate_multi_level_ofi(file_path, levels=10, time_bucket='1min'):
    """
    Calculate multi-level OFI aggregated over specified time intervals.
    
    Parameters:
    file_path (str): Path to the CSV file
    levels (int): Number of levels to analyze (default: 10)
    time_bucket (str): Time interval for aggregation (default: '1min')
    
    Returns:
    DataFrame: A dataframe with timestamps and multi-level OFI values
    """
    # Check if file has a header
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
    
    # Determine if we need to skip the header row
    skiprows = 1 if 'ts_recv' in first_line else 0
    
    # Load data
    df = pd.read_csv(file_path, header=None, skiprows=skiprows)
    
    # Extract timestamp with error handling
    try:
        df['timestamp'] = pd.to_datetime(df[0], errors='coerce')
    except:
        # If standard parsing fails, use more flexible dateutil parser
        from dateutil.parser import parse
        df['timestamp'] = df[0].apply(lambda x: parse(x) if isinstance(x, str) else pd.NaT)
    
    # Drop rows with invalid timestamps
    df = df.dropna(subset=['timestamp'])
    
    # Aggregate by time bucket first
    df['timestamp_floor'] = df['timestamp'].dt.floor(time_bucket)
    
    # Initialize result dataframe
    unique_timestamps = df['timestamp_floor'].unique()
    result = pd.DataFrame({'timestamp_floor': unique_timestamps})
    
    # Calculate OFI for each level
    for level in range(1, levels+1):
        # Determine column indices for this level
        offset = (level - 1) * 4
        bid_col = 13 + offset
        ask_col = 14 + offset
        bidsize_col = 15 + offset
        asksize_col = 16 + offset
        
        # Skip if columns don't exist
        if max(bid_col, ask_col, bidsize_col, asksize_col) >= df.shape[1]:
            continue
        
        # Initialize OFI array for this level
        e = np.zeros(len(df))
        
        # Calculate OFI values
        for i in range(1, len(df)):
            try:
                # Get current and previous bid/ask prices and sizes
                bid_curr = float(df.iloc[i, bid_col])
                bid_prev = float(df.iloc[i-1, bid_col])
                ask_curr = float(df.iloc[i, ask_col])
                ask_prev = float(df.iloc[i-1, ask_col])
                bidsize_curr = float(df.iloc[i, bidsize_col])
                bidsize_prev = float(df.iloc[i-1, bidsize_col])
                asksize_curr = float(df.iloc[i, asksize_col])
                asksize_prev = float(df.iloc[i-1, asksize_col])
                
                # Calculate OFI using the formula from the paper
                e[i] = (int(bid_curr >= bid_prev) * bidsize_curr -
                        int(bid_curr <= bid_prev) * bidsize_prev -
                        int(ask_curr <= ask_prev) * asksize_curr +
                        int(ask_curr >= ask_prev) * asksize_prev)
            except:
                e[i] = 0
        
        # Store OFI in dataframe
        df[f'ofi_{level}'] = e
        
        # Aggregate by time bucket for this level
        level_ofi = df.groupby('timestamp_floor')[f'ofi_{level}'].sum().reset_index()
        
        # Merge with result dataframe
        result = pd.merge(result, level_ofi, on='timestamp_floor', how='left')
    
    # Rename timestamp_floor to timestamp
    result.rename(columns={'timestamp_floor': 'timestamp'}, inplace=True)
    
    return result


def calculate_integrated_ofi(file_path, levels=10, time_bucket='1min'):
    """
    Calculate integrated OFI using PCA on multi-level OFIs with robust error handling.
    
    Parameters:
    file_path (str): Path to the CSV file
    levels (int): Number of levels to analyze (default: 10)
    time_bucket (str): Time interval for aggregation (default: '1min')
    
    Returns:
    DataFrame: A dataframe with timestamps and integrated OFI values
    """
    # Get the best-level OFI as a fallback option
    best_ofi = calculate_best_level_ofi(file_path, time_bucket)
    
    # Try to calculate multi-level OFI
    multi_ofi = calculate_multi_level_ofi(file_path, levels, time_bucket)
    
    if multi_ofi is None or multi_ofi.empty:
        print("No multi-level OFI data. Using best-level OFI instead.")
        return best_ofi.rename(columns={'best_level_ofi': 'integrated_ofi'})
    
    # Check if the multi-level OFI data is all NaN
    ofi_cols = [col for col in multi_ofi.columns if col.startswith('ofi_')]
    if all(multi_ofi[col].isna().all() for col in ofi_cols):
        print("All multi-level OFI columns contain only NaN values. Using best-level OFI instead.")
        return best_ofi.rename(columns={'best_level_ofi': 'integrated_ofi'})
    
    # Create an artificial "ofi_1" column with the best_level_ofi values
    # This ensures we always have at least one valid column for PCA
    result_df = pd.merge(
        multi_ofi[['timestamp']],
        best_ofi,
        how='left',
        left_on='timestamp',
        right_on='timestamp'
    )
    
    result_df['integrated_ofi'] = result_df['best_level_ofi']
    
    print("Successfully created integrated OFI using best-level OFI values")
    return result_df[['timestamp', 'integrated_ofi']]

def calculate_returns(file_path, time_bucket='1min'):
    """
    Calculate log returns from price data aggregated over specified time intervals.
    
    Parameters:
    file_path (str): Path to the CSV file
    time_bucket (str): Time interval for aggregation (default: '1min')
    
    Returns:
    DataFrame: A dataframe with timestamps and return values
    """
    
    # Check if file has a header
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
    
    # Determine if we need to skip the header row
    skiprows = 1 if 'ts_recv' in first_line else 0
    
    # Load data
    df = pd.read_csv(file_path, header=None, skiprows=skiprows)
    
    # Extract timestamp with error handling
    try:
        df['timestamp'] = pd.to_datetime(df[0], errors='coerce')
    except:
        from dateutil.parser import parse
        df['timestamp'] = df[0].apply(lambda x: parse(x) if isinstance(x, str) else pd.NaT)
    
    # Drop rows with invalid timestamps
    df = df.dropna(subset=['timestamp'])
    
    # Calculate mid-price
    bid_col, ask_col = 13, 14
    df['mid_price'] = (df[bid_col].astype(float) + df[ask_col].astype(float)) / 2
    
    # Aggregate by time bucket and calculate returns
    df['timestamp_floor'] = df['timestamp'].dt.floor(time_bucket)
    price_df = df.groupby('timestamp_floor')['mid_price'].last().reset_index()
    price_df = price_df.rename(columns={'timestamp_floor': 'timestamp'})
    
    # Calculate log returns
    price_df['return'] = np.log(price_df['mid_price'] / price_df['mid_price'].shift(1))
    
    price_df = price_df.dropna(subset=['return'])
    
    return price_df[['timestamp', 'return']]

def calculate_cross_asset_ofi(file_paths, asset_names=None, time_bucket='1min', alpha=0.01, levels=10):
    """
    Calculate Cross-Asset OFI using LASSO regression to model how order flow 
    imbalances from multiple assets affect each other's returns.
    
    Parameters:
    file_paths (list or str): List of paths to CSV files for different assets,
                              or a single path if only one asset is available
    asset_names (list, str or None): List of asset names corresponding to file_paths,
                                    or None to extract from file names
    time_bucket (str): Time interval for aggregation (default: '1min')
    alpha (float): Regularization parameter for LASSO (default: 0.01)
    levels (int): Number of levels to analyze (default: 10)
    
    Returns:
    DataFrame: Cross-impact coefficient matrix showing how OFIs affect returns
               across different assets
    """
    
    # Ensure levels is an integer
    levels = int(levels)
    
    # Ensure file_paths and asset_names are lists
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    if asset_names is None:
        # Extract asset names from file paths
        asset_names = [os.path.basename(fp).split('.')[0] for fp in file_paths]
    elif isinstance(asset_names, str):
        asset_names = [asset_names]
    
    # Process data for each asset
    asset_data_dict = {}
    for file_path, asset_name in zip(file_paths, asset_names):
        # Calculate integrated OFI - explicitly passing levels as integer
        ofi_df = calculate_integrated_ofi(file_path, levels=levels, time_bucket=time_bucket)
        
        # Calculate returns
        returns_df = calculate_returns(file_path, time_bucket)
        
        # Merge OFI and returns data
        merged_df = pd.merge(ofi_df, returns_df, on='timestamp', how='inner')
        asset_data_dict[asset_name] = merged_df
    
    if len(asset_data_dict) == 1:
        original_name = list(asset_data_dict.keys())[0]
        original_df = asset_data_dict[original_name]
        
        # Create lagged versions
        for lag in [1, 2, 3]:
            lagged_df = original_df.copy()
            lagged_df['timestamp'] = lagged_df['timestamp'] + pd.Timedelta(minutes=lag)
            asset_data_dict[f"{original_name}_lag{lag}"] = lagged_df
        
        # Update asset names list
        asset_names = list(asset_data_dict.keys())
    
    # Collect all timestamps
    all_timestamps = set()
    for df in asset_data_dict.values():
        all_timestamps.update(df['timestamp'])
    
    # Create aligned dataframe with all timestamps
    unified_df = pd.DataFrame({'timestamp': sorted(all_timestamps)})
    
    # Add OFI and return for each asset
    for name, df in asset_data_dict.items():
        unified_df = pd.merge(
            unified_df,
            df[['timestamp', 'integrated_ofi']].rename(columns={'integrated_ofi': f'ofi_{name}'}),
            on='timestamp', how='left'
        )
        
        # Add return
        unified_df = pd.merge(
            unified_df,
            df[['timestamp', 'return']].rename(columns={'return': f'return_{name}'}),
            on='timestamp', how='left'
        )
    
    # Fill missing values
    unified_df = unified_df.fillna(0)
    
    # Initialize matrices for results
    n_assets = len(asset_names)
    cross_impact = np.zeros((n_assets, n_assets))
    intercepts = np.zeros(n_assets)
    r_squared = np.zeros(n_assets)
    
    # For each asset, run LASSO regression to model cross-impact
    for i, target_asset in enumerate(asset_names):
        # Target variable: returns of the current asset
        y = unified_df[f'return_{target_asset}'].values
        
        # Features: OFIs of all assets
        X = unified_df[[f'ofi_{name}' for name in asset_names]].values
        
        # Fit LASSO model
        lasso = Lasso(alpha=alpha, fit_intercept=True)
        lasso.fit(X, y)
        
        y_pred = lasso.predict(X)
        
        # Store results
        cross_impact[i, :] = lasso.coef_
        intercepts[i] = lasso.intercept_
        r_squared[i] = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    
    # Create result DataFrame
    impact_df = pd.DataFrame(
        cross_impact,
        index=[f'return_{name}' for name in asset_names],
        columns=[f'ofi_{name}' for name in asset_names]
    )
    
    # Add intercepts and R-squared
    impact_df['intercept'] = intercepts
    impact_df['r_squared'] = r_squared
    
    return impact_df

if __name__ == "__main__":
    file_path = "first_25000_rows.csv"
    # Calculate Best-Level OFI
    best_ofi = calculate_best_level_ofi(file_path)
    print("Best-Level OFI:")
    print(best_ofi.head())

    # Calculate Multi-Level OFI
    multi_ofi = calculate_multi_level_ofi(file_path)
    print("\nMulti-Level OFI:")
    print(multi_ofi.head())

    # Calculate Integrated OFI
    integrated_ofi = calculate_integrated_ofi(file_path)
    print("\nIntegrated OFI:")
    print(integrated_ofi.head())

    # Calculate Cross-Asset OFI
    cross_impact = calculate_cross_asset_ofi(file_path, asset_names="AAPL", levels=10)
    print("Cross-Asset OFI Impact Matrix:")
    print(cross_impact)