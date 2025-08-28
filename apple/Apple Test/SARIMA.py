import itertools
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Suppress warnings
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)

def find_best_sarima(series, seasonal_period=7, max_order=7, d=None, D=None):
    """
    Find the best SARIMA model using a stepwise approach with AICc as the criterion.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data to model
    seasonal_period : int, default=7
        Seasonal period (S parameter)
    max_order : int, default=7
        Maximum value for p, d, q, P, D, Q parameters
    d : int, optional
        Non-seasonal differencing order. If None, will try values 0-2.
    D : int, optional
        Seasonal differencing order. If None, will try values 0-1.
        
    Returns:
    --------
    tuple
        (best_order, best_seasonal_order, best_model, results)
        where:
        - best_order: tuple (p, d, q)
        - best_seasonal_order: tuple (P, D, Q, S)
        - best_model: Fitted SARIMAX model
        - results: DataFrame with all tested models
    """
    def evaluate_model(series, order, seasonal_order):
        try:
            model = SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = model.fit(disp=0)
            return results.aicc, results
        except:
            return float('inf'), None
    
    # Initialize best parameters
    best_aicc = float('inf')
    best_order = (0, 0, 0)
    best_seasonal_order = (0, 0, 0, seasonal_period)
    best_model = None
    results = []
    
    # Step 1: Find best non-seasonal differencing (d)
    print("Step 1: Finding best non-seasonal differencing (d)")
    if d is None:
        d_range = range(0, min(3, max_order))  # Try d from 0 to 2
    else:
        d_range = [d]
        
    best_d = 0
    for d_test in d_range:
        # Simple AR(1) model to test differencing
        aicc, _ = evaluate_model(
            series,
            order=(1, d_test, 0),
            seasonal_order=(0, 0, 0, 0)  # No seasonal components yet
        )
        print(f"d={d_test} - AICc: {aicc:.2f}")
        if aicc < best_aicc:
            best_aicc = aicc
            best_d = d_test
    
    d = best_d
    print(f"Best d: {d}")
    
    # Step 2: Find best AR order (p)
    print("\nStep 2: Finding best AR order (p)")
    best_p = 0
    for p in range(0, min(7, max_order)):
        aicc, _ = evaluate_model(
            series,
            order=(p, d, 0),
            seasonal_order=(0, 0, 0, 0)  # No seasonal components yet
        )
        print(f"p={p} - AICc: {aicc:.2f}")
        if aicc < best_aicc:
            best_aicc = aicc
            best_p = p
    
    p = best_p
    print(f"Best p: {p}")
    
    # Step 3: Find best MA order (q)
    print("\nStep 3: Finding best MA order (q)")
    best_q = 0
    for q in range(0, min(7, max_order)):
        aicc, model = evaluate_model(
            series,
            order=(p, d, q),
            seasonal_order=(0, 0, 0, 0)  # No seasonal components yet
        )
        print(f"q={q} - AICc: {aicc:.2f}")
        results.append({
            'order': (p, d, q),
            'seasonal_order': (0, 0, 0, 0),
            'aicc': aicc
        })
        if aicc < best_aicc:
            best_aicc = aicc
            best_q = q
            best_model = model
    
    q = best_q
    best_order = (p, d, q)
    print(f"Best q: {q}")
    
    # Step 4: Find best seasonal differencing (D)
    print("\nStep 4: Finding best seasonal differencing (D)")
    if D is None:
        D_range = [0, 1]  # Typically D is 0 or 1
    else:
        D_range = [D]
        
    best_D = 0
    for D_test in D_range:
        aicc, _ = evaluate_model(
            series,
            order=best_order,
            seasonal_order=(0, D_test, 0, seasonal_period)  # Only test D, no P or Q yet
        )
        print(f"D={D_test} - AICc: {aicc:.2f}")
        if aicc < best_aicc:
            best_aicc = aicc
            best_D = D_test
    
    D = best_D
    print(f"Best D: {D}")
    
    # Step 5: Find best seasonal AR order (P)
    print("\nStep 5: Finding best seasonal AR order (P)")
    best_P = 0
    for P in range(0, min(7, max_order)):
        aicc, model = evaluate_model(
            series,
            order=best_order,
            seasonal_order=(P, D, 0, seasonal_period)  # Only test P, keep Q=0
        )
        print(f"P={P} - AICc: {aicc:.2f}")
        results.append({
            'order': best_order,
            'seasonal_order': (P, D, 0, seasonal_period),
            'aicc': aicc
        })
        if aicc < best_aicc:
            best_aicc = aicc
            best_P = P
            best_model = model
    
    P = best_P
    print(f"Best P: {P}")
    
    # Step 6: Find best seasonal MA order (Q)
    print("\nStep 6: Finding best seasonal MA order (Q)")
    best_Q = 0
    for Q in range(0, min(7, max_order)):
        aicc, model = evaluate_model(
            series,
            order=best_order,
            seasonal_order=(P, D, Q, seasonal_period)  # Now test Q with best P and D
        )
        print(f"Q={Q} - AICc: {aicc:.2f}")
        results.append({
            'order': best_order,
            'seasonal_order': (P, D, Q, seasonal_period),
            'aicc': aicc
        })
        if aicc < best_aicc:
            best_aicc = aicc
            best_Q = Q
            best_model = model
    
    Q = best_Q
    best_seasonal_order = (P, D, Q, seasonal_period)
    print(f"Best Q: {Q}")
    
    # Final model with all parameters
    if best_model is None:
        best_model, _ = evaluate_model(series, best_order, best_seasonal_order)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results).sort_values('aicc')
    
    print("\nFinal Model:")
    print(f"SARIMA{best_order}{best_seasonal_order} - AICc: {best_aicc:.2f}")
    
    return best_model, best_order, best_seasonal_order, results_df

def fit_sarima(series, order, seasonal_order, **kwargs):
    """
    Fit a SARIMA model with the given parameters.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    order : tuple
        (p, d, q) order
    seasonal_order : tuple
        (P, D, Q, S) seasonal order
    **kwargs : 
        Additional arguments to pass to SARIMAX
        
    Returns:
    --------
    model : SARIMAX model
        Fitted SARIMAX model
    """
    model = SARIMAX(series,
                   order=order,
                   seasonal_order=seasonal_order,
                   enforce_stationarity=False,
                   enforce_invertibility=False,
                   **kwargs)
    
    return model.fit(disp=0)

def forecast_sarima(model, steps, alpha=0.05):
    """
    Generate forecast and confidence intervals from a fitted SARIMAX model.
    
    Parameters:
    -----------
    model : SARIMAX model
        Fitted SARIMAX model
    steps : int
        Number of steps to forecast
    alpha : float, default=0.05
        Significance level for confidence intervals
        
    Returns:
    --------
    forecast : pd.Series
        Point forecasts
    conf_int : pd.DataFrame
        Confidence intervals for the forecasts
    """
    # Get forecast
    forecast_result = model.get_forecast(steps=steps)
    
    # Get confidence intervals
    conf_int = forecast_result.conf_int(alpha=alpha)
    
    return forecast_result.predicted_mean, conf_int

def plot_forecast(series, model, steps, title="SARIMA Forecast"):
    """
    Plot the original series, fitted values, and forecast with confidence intervals.
    
    Parameters:
    -----------
    series : pd.Series
        Original time series data (must have datetime index)
    model : tuple, SARIMAX model, or SARIMAXResults
        Can be one of:
        - Output from find_best_sarima() (a tuple)
        - A fitted SARIMAXResults object
        - An unfitted SARIMAX model (will be fitted)
    steps : int
        Number of steps to forecast
    title : str, optional
        Title for the plot
        
    Returns:
    --------
    tuple
        (forecast_mean, conf_int, metrics_dict)
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from statsmodels.tsa.statespace.sarimax import SARIMAXResults, SARIMAX
    import numpy as np
    
    # Handle different input types
    if isinstance(model, tuple) and len(model) == 4:
        # Input is from find_best_sarima() - extract the fitted model
        # The return value is (best_model, best_order, best_seasonal_order, results_df)
        model = model[0]  # First element is the fitted model
    elif isinstance(model, SARIMAX):
        # Input is an unfitted SARIMAX model - fit it
        print("Fitting the model...")
        model = model.fit(disp=0)
    elif not isinstance(model, SARIMAXResults):
        raise ValueError("model must be a SARIMAX model, SARIMAXResults, or output from find_best_sarima()")
    
    # If we get here, model should be a fitted SARIMAX model
    
    # Get forecast and confidence intervals
    forecast = model.get_forecast(steps=steps)
    
    # Handle frequency for date range
    freq = getattr(series.index, 'freq', None) or 'D'  # Default to daily if no freq
    forecast_index = pd.date_range(
        series.index[-1], 
        periods=steps+1, 
        freq=freq
    )[1:]
    
    # Calculate metrics
    fitted_values = model.fittedvalues
    mae = np.mean(np.abs(series - fitted_values))
    rmse = np.sqrt(np.mean((series - fitted_values) ** 2))
    aicc = model.aicc
    sawant_score = ((mae + rmse) / 2) * np.sqrt(aicc + 1e-10)
    
    # Create metrics dictionary
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'AICc': aicc,
        'Sawant_Score': sawant_score
    }
    
    # Create figure with larger size to accommodate metrics
    plt.figure(figsize=(14, 8))
    
    # Plot original series (last 100 points if series is long)
    plot_series = series[-100:] if len(series) > 100 else series
    plt.plot(plot_series.index, plot_series, label='Observed', color='#1f77b4')
    
    # Plot fitted values (aligned with observed data)
    try:
        fitted = model.fittedvalues
        # Only plot fitted values that align with our series
        fitted = fitted[fitted.index.isin(series.index)]
        plt.plot(fitted.index, fitted, color='#ff7f0e', label='Fitted', alpha=0.8)
    except:
        print("Could not plot fitted values - using only observed data")
    
    # Plot forecast
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()
    
    plt.plot(forecast_index, forecast_mean, '--', color='#d62728', 
             label=f'Forecast ({steps} steps)')
    
    # Plot confidence intervals
    plt.fill_between(forecast_index,
                    conf_int.iloc[:, 0],
                    conf_int.iloc[:, 1],
                    color='#d62728', alpha=0.15,
                    label='95% Confidence Interval')
    
    # Add vertical line at forecast start
    plt.axvline(x=series.index[-1], color='k', linestyle='--', alpha=0.5)
    
    # Add simple title
    plt.title("SARIMA Forecast")
    plt.xlabel('Date')
    plt.ylabel('Value')
    
    # Create custom legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper left')
    
    # Add metrics box with improved formatting
    metrics_text = (
        f"Model Metrics:\n"
        f"MAE: {mae:.4f}\n"
        f"RMSE: {rmse:.4f}\n"
        f"AICc: {aicc:.2f}\n"
        f"Sawant's Score: {sawant_score:.4f}"
    )
    
    # Position the metrics box in the upper right corner
    plt.gcf().text(
        0.8, 0.95, 
        metrics_text, 
        bbox=dict(
            facecolor='white', 
            alpha=0.9, 
            edgecolor='gray', 
            boxstyle='round,pad=0.5',
            linewidth=0.5
        ),
        fontsize=10, 
        verticalalignment='top',
        horizontalalignment='left',
        fontfamily='monospace'
    )
    
    plt.grid(True, alpha=0.3)
    
    # Format x-axis for dates
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent text overlap
    plt.subplots_adjust(right=0.85)
    
    plt.tight_layout()
    plt.show()
    
    return forecast_mean, conf_int, metrics

def calculate_metrics(model, test_series, train_series=None):
    """
    Calculate various error metrics for a fitted SARIMAX model.
    
    Parameters:
    -----------
    model : SARIMAXResults
        Fitted SARIMAX model
    test_series : pd.Series
        Test dataset for evaluation
    train_series : pd.Series, optional
        Training dataset (required for AICc if model was not fitted with it)
        
    Returns:
    --------
    dict
        Dictionary containing various error metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    
    # Get predictions for the test period
    forecast = model.get_forecast(steps=len(test_series))
    pred = forecast.predicted_mean
    
    # Calculate metrics
    mae = mean_absolute_error(test_series, pred)
    rmse = np.sqrt(mean_squared_error(test_series, pred))
    mape = np.mean(np.abs((test_series - pred) / test_series)) * 100
    
    # Calculate AICc if possible
    try:
        aicc = model.aicc
    except:
        aicc = None
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'AICc': aicc
    }

def hybrid_score(metrics, weights=None):
    """
    Calculate a simple hybrid score as: (avg of RMSE and MAE) * sqrt(AICc)
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing:
        - 'MAE': Mean Absolute Error
        - 'RMSE': Root Mean Square Error
        - 'AICc': Corrected Akaike Information Criterion
    weights : dict, optional
        For backward compatibility, not used in this implementation
        
    Returns:
    --------
    float
        Hybrid score (lower is better)
    """
    # Calculate average of RMSE and MAE
    avg_error = (metrics.get('RMSE', 0) + metrics.get('MAE', 0)) / 2
    
    # Get AICc, default to a large number if not available
    aicc = metrics.get('AICc', 1e6)
    
    # Calculate score: (avg of RMSE and MAE) * sqrt(AICc)
    # Add small constant to avoid sqrt(0) if AICc is 0
    score = avg_error * np.sqrt(aicc + 1e-10)
    
    return score

def evaluate_model(model, train_series, test_series, verbose=True):
    """
    Evaluate a SARIMAX model and return metrics and hybrid score.
    
    Parameters:
    -----------
    model : SARIMAX or SARIMAXResults
        Model to evaluate (will be fitted if not already)
    train_series : pd.Series
        Training data
    test_series : pd.Series
        Test data
    verbose : bool, optional
        Whether to print the metrics
        
    Returns:
    --------
    tuple
        (metrics_dict, hybrid_score)
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAXResults
    
    # Fit the model if it's not already fitted
    if not isinstance(model, SARIMAXResults):
        model = model.fit(disp=0)
    
    # Calculate metrics
    metrics = calculate_metrics(model, test_series, train_series)
    
    # Calculate hybrid score using the simple formula
    score = hybrid_score(metrics)
    
    if verbose:
        print("\nModel Evaluation Metrics:")
        print("-" * 50)
        print(f"MAE: {metrics['MAE']:.4f}")
        print(f"RMSE: {metrics['RMSE']:.4f}")
        if 'MAPE' in metrics:
            print(f"MAPE: {metrics['MAPE']:.2f}%")
        if 'AICc' in metrics and metrics['AICc'] is not None:
            print(f"AICc: {metrics['AICc']:.2f}")
        print(f"Hybrid Score: {score:.6f}")
        print("-" * 50)
    
    return metrics, score

def find_best_sarima_hybrid(series, test_size=0.2, seasonal_period=7, max_order=7, d=None, D=None):
    """
    Find the best SARIMA model using a stepwise approach with hybrid scoring.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data to model
    test_size : float, default=0.2
        Proportion of data to use for testing
    seasonal_period : int, default=7
        Seasonal period (S parameter)
    max_order : int, default=7
        Maximum value for p, d, q, P, D, Q parameters
    d : int, optional
        Non-seasonal differencing order. If None, will try values 0-2.
    D : int, optional
        Seasonal differencing order. If None, will try values 0-1.
        
    Returns:
    --------
    tuple
        (best_order, best_seasonal_order, best_model, results_df)
        where:
        - best_order: tuple (p, d, q)
        - best_seasonal_order: tuple (P, D, Q, S)
        - best_model: Fitted SARIMAX model
        - results_df: DataFrame with all tested models
    """
    def evaluate_model(series, order, seasonal_order):
        try:
            model = SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = model.fit(disp=0)
            return results.aicc, results
        except:
            return float('inf'), None
    
    # Initialize best parameters
    best_aicc = float('inf')
    best_order = (0, 0, 0)
    best_seasonal_order = (0, 0, 0, seasonal_period)
    best_model = None
    results = []
    
    # Split data into training and testing sets
    split_idx = int(len(series) * (1 - test_size))
    train, test = series.iloc[:split_idx], series.iloc[split_idx:]
    
    # Step 1: Find best non-seasonal differencing (d)
    print("Step 1: Finding best non-seasonal differencing (d)")
    if d is None:
        d_range = range(0, min(3, max_order))  # Try d from 0 to 2
    else:
        d_range = [d]
        
    best_d = 0
    for d_test in d_range:
        # Simple AR(1) model to test differencing
        aicc, _ = evaluate_model(
            train,
            order=(1, d_test, 0),
            seasonal_order=(0, 0, 0, 0)  # No seasonal components yet
        )
        print(f"d={d_test} - AICc: {aicc:.2f}")
        if aicc < best_aicc:
            best_aicc = aicc
            best_d = d_test
    
    d = best_d
    print(f"Best d: {d}")
    
    # Step 2: Find best AR order (p)
    print("\nStep 2: Finding best AR order (p)")
    best_p = 0
    for p in range(0, min(7, max_order)):
        aicc, _ = evaluate_model(
            train,
            order=(p, d, 0),
            seasonal_order=(0, 0, 0, 0)  # No seasonal components yet
        )
        print(f"p={p} - AICc: {aicc:.2f}")
        if aicc < best_aicc:
            best_aicc = aicc
            best_p = p
    
    p = best_p
    print(f"Best p: {p}")
    
    # Step 3: Find best MA order (q)
    print("\nStep 3: Finding best MA order (q)")
    best_q = 0
    for q in range(0, min(7, max_order)):
        aicc, model = evaluate_model(
            train,
            order=(p, d, q),
            seasonal_order=(0, 0, 0, 0)  # No seasonal components yet
        )
        print(f"q={q} - AICc: {aicc:.2f}")
        results.append({
            'order': (p, d, q),
            'seasonal_order': (0, 0, 0, 0),
            'aicc': aicc
        })
        if aicc < best_aicc:
            best_aicc = aicc
            best_q = q
            best_model = model
    
    q = best_q
    best_order = (p, d, q)
    print(f"Best q: {q}")
    
    # Step 4: Find best seasonal differencing (D)
    print("\nStep 4: Finding best seasonal differencing (D)")
    if D is None:
        D_range = [0, 1]  # Typically D is 0 or 1
    else:
        D_range = [D]
        
    best_D = 0
    for D_test in D_range:
        aicc, _ = evaluate_model(
            train,
            order=best_order,
            seasonal_order=(0, D_test, 0, seasonal_period)  # Only test D, no P or Q yet
        )
        print(f"D={D_test} - AICc: {aicc:.2f}")
        if aicc < best_aicc:
            best_aicc = aicc
            best_D = D_test
    
    D = best_D
    print(f"Best D: {D}")
    
    # Step 5: Find best seasonal AR order (P)
    print("\nStep 5: Finding best seasonal AR order (P)")
    best_P = 0
    for P in range(0, min(7, max_order)):
        aicc, model = evaluate_model(
            train,
            order=best_order,
            seasonal_order=(P, D, 0, seasonal_period)  # Only test P, keep Q=0
        )
        print(f"P={P} - AICc: {aicc:.2f}")
        results.append({
            'order': best_order,
            'seasonal_order': (P, D, 0, seasonal_period),
            'aicc': aicc
        })
        if aicc < best_aicc:
            best_aicc = aicc
            best_P = P
            best_model = model
    
    P = best_P
    print(f"Best P: {P}")
    
    # Step 6: Find best seasonal MA order (Q)
    print("\nStep 6: Finding best seasonal MA order (Q)")
    best_Q = 0
    for Q in range(0, min(7, max_order)):
        aicc, model = evaluate_model(
            train,
            order=best_order,
            seasonal_order=(P, D, Q, seasonal_period)  # Now test Q with best P and D
        )
        print(f"Q={Q} - AICc: {aicc:.2f}")
        results.append({
            'order': best_order,
            'seasonal_order': (P, D, Q, seasonal_period),
            'aicc': aicc
        })
        if aicc < best_aicc:
            best_aicc = aicc
            best_Q = Q
            best_model = model
    
    Q = best_Q
    best_seasonal_order = (P, D, Q, seasonal_period)
    print(f"Best Q: {Q}")
    
    # Evaluate best model on test set
    print("\nEvaluating best model on test set...")
    metrics, score = evaluate_model(best_model, train, test)
    
    # Add metrics to results
    results_df = pd.DataFrame(results)
    results_df['test_MAE'] = metrics['MAE']
    results_df['test_RMSE'] = metrics['RMSE']
    results_df['test_MAPE'] = metrics['MAPE']
    results_df['hybrid_score'] = score
    
    return best_model, best_order, best_seasonal_order, results_df
