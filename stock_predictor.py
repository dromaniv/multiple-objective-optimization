import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

class StockPredictor:
    def __init__(self, data_folder="Bundle2"):
        self.data_folder = data_folder
        self.asset_names, self.asset_times, self.asset_prices = self.load_data()

    def load_data(self):
        asset_names = []
        asset_times = []
        asset_prices = []
        
        folder_name = os.path.basename(self.data_folder).lower()
        if "bundle" in folder_name and folder_name[-1].isdigit():
            part_suffix = f"Part{folder_name[-1]}.txt"
        else:
            part_suffix = ".txt"
            
        print(f"Looking for files with suffix: {part_suffix}")
        
        txt_files = [f for f in os.listdir(self.data_folder) if f.endswith(part_suffix)]
        
        if not txt_files:
            print(f"No files found with suffix {part_suffix}, looking for any .txt files")
            txt_files = [f for f in os.listdir(self.data_folder) if f.endswith('.txt')]
            
        print(f"Found {len(txt_files)} files to process")
        
        for fname in txt_files:
            path = os.path.join(self.data_folder, fname)
            with open(path, "r") as f:
                asset_name = f.readline().strip()
                N = int(f.readline().strip())
                times = []
                prices = []
                for _ in range(N):
                    line = f.readline().strip()
                    t_str, p_str = line.split()
                    times.append(float(t_str))
                    prices.append(float(p_str))
            asset_names.append(asset_name)
            asset_times.append(times)
            asset_prices.append(prices)
            
        if not asset_names:
            raise ValueError(f"No asset data found in folder: {self.data_folder}")
            
        return asset_names, asset_times, asset_prices

    def analyze_market_cycles(self, times, prices, min_peak_height=0.05, min_peak_distance=10):
        price_range = np.max(prices) - np.min(prices)
        if price_range == 0:
            return {"trend": "flat", "avg_cycle_length": None, "volatility": 0}
            
        prices_norm = (prices - np.min(prices)) / price_range
        
        peak_indices, _ = find_peaks(prices_norm, height=min_peak_height, distance=min_peak_distance)
        trough_indices, _ = find_peaks(-prices_norm, height=min_peak_height, distance=min_peak_distance)
        
        if len(peak_indices) < 2 and len(trough_indices) < 2:
            if len(times) > 1:
                coef = np.polyfit(times, prices, 1)[0]
                trend = "upward" if coef > 0 else "downward" if coef < 0 else "flat"
            else:
                trend = "flat"
                
            if len(prices) > 1:
                volatility = np.std(prices) / np.mean(prices) if np.mean(prices) != 0 else 0
            else:
                volatility = 0
                
            return {"trend": trend, "avg_cycle_length": None, "volatility": volatility}
        
        cycle_lengths = []
        if len(peak_indices) >= 2:
            for i in range(1, len(peak_indices)):
                cycle_lengths.append(times[peak_indices[i]] - times[peak_indices[i-1]])
        
        elif len(trough_indices) >= 2:
            for i in range(1, len(trough_indices)):
                cycle_lengths.append(times[trough_indices[i]] - times[trough_indices[i-1]])
        
        avg_cycle_length = np.mean(cycle_lengths) if cycle_lengths else None
        
        if len(times) > 1:
            coef = np.polyfit(times, prices, 1)[0]
            trend = "upward" if coef > 0 else "downward" if coef < 0 else "flat"
        else:
            trend = "flat"
            
        volatility = np.std(prices) / np.mean(prices) if np.mean(prices) != 0 else 0
        
        last_idx = len(prices) - 1
        
        if peak_indices.size > 0 and trough_indices.size > 0:
            last_peak_idx = peak_indices[-1]
            last_trough_idx = trough_indices[-1]
            
            if last_peak_idx > last_trough_idx:
                cycle_position = "near_peak"
            else:
                cycle_position = "near_trough"
                
            if last_idx - max(last_peak_idx, last_trough_idx) > min_peak_distance:
                if prices[-1] > prices[last_peak_idx] * 0.95:
                    cycle_position = "near_peak"
                elif prices[-1] < prices[last_trough_idx] * 1.05:
                    cycle_position = "near_trough"
                else:
                    recent_trend = "upward" if prices[-1] > prices[-min(10, len(prices))] else "downward"
                    cycle_position = "mid_upward" if recent_trend == "upward" else "mid_downward"
        else:
            cycle_position = "near_peak" if trend == "upward" else "near_trough"
        
        return {
            "trend": trend,
            "avg_cycle_length": avg_cycle_length,
            "volatility": volatility,
            "cycle_position": cycle_position,
            "peak_indices": peak_indices,
            "trough_indices": trough_indices
        }

    def market_aware_curve_fit_forecast(self, max_attempts=7, reversion_strength=0.6):
        import warnings
        
        # def model_linear(x, a, b):
        #     return a * x + b
        
        # def model_quadratic(x, a, b, c):
        #     return a * x**2 + b * x + c
        
        # def model_sin_linear(x, a, b, c, d, e):
        #     return a * np.sin(b * x + c) + d * x + e
        
        # def model_exp(x, a, b, c):
        #     safe_exp = np.exp(np.minimum(b * x, 700))
        #     return a * safe_exp + c
        
        # def model_log(x, a, b, c):
        #     with np.errstate(divide='ignore', invalid='ignore'):
        #         result = a * np.log(np.maximum(b * x, 1e-10)) + c
        #         invalid_mask = ~np.isfinite(result)
        #         if np.any(invalid_mask):
        #             valid_mask = ~invalid_mask
        #             if np.sum(valid_mask) >= 2:
        #                 coeffs = np.polyfit(x[valid_mask], result[valid_mask], 1)
        #                 result[invalid_mask] = np.polyval(coeffs, x[invalid_mask])
        #             else:
        #                 result[invalid_mask] = result[~invalid_mask][0] if np.any(~invalid_mask) else 0
        #         return result
        
        # def model_logistic(x, a, b, c, d):
        #     return a / (1 + b * np.exp(-c * x)) + d
        
        def model_damped_sin(x, a, b, c, d, e, f):
            return a * np.exp(-b * x) * np.sin(c * x + d) + e * x + f
        
        # def model_power(x, a, b, c):
        #     with np.errstate(divide='ignore', invalid='ignore'):
        #         result = a * np.power(np.maximum(x, 1e-10), b) + c
        #         invalid_mask = ~np.isfinite(result)
        #         if np.any(invalid_mask):
        #             valid_mask = ~invalid_mask
        #             if np.sum(valid_mask) >= 2:
        #                 coeffs = np.polyfit(x[valid_mask], result[valid_mask], 1)
        #                 result[invalid_mask] = np.polyval(coeffs, x[invalid_mask])
        #             else:
        #                 result[invalid_mask] = 0
        #         return result
        
        def model_holt_winters(x, level, trend, season1, season2, period=50):
            result = np.zeros_like(x, dtype=float)
            for i, t in enumerate(x):
                cycle = np.sin(2 * np.pi * t / period)
                result[i] = level + trend * t + season1 * cycle + season2 * np.cos(2 * np.pi * t / period)
            return result
        
        # def model_fourier(x, a0, a1, b1, a2, b2, w):
        #     return a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x) + a2 * np.cos(2 * w * x) + b2 * np.sin(2 * w * x)
        
        models = [
            # (model_linear, "Linear"),
            # (model_quadratic, "Quadratic"),
            # (model_sin_linear, "Sin+Linear"),
            # (model_log, "Logarithmic"),
            # (model_logistic, "Logistic"),
            # (model_power, "Power"),
            # (model_fourier, "Fourier"),
            (model_holt_winters, "HoltWinters"),
            (model_damped_sin, "DampedSin"),
            # (model_exp, "Exponential")
        ]
        
        curve_fit_pred = {}
        
        for i, name in enumerate(self.asset_names):
            print(f"Processing asset: {name}")
            times = np.array(self.asset_times[i])
            prices = np.array(self.asset_prices[i])
            training_start = times[0]
            training_end = times[-1]
            forecast_time = training_end + 100
            
            market_info = self.analyze_market_cycles(times, prices)
            print(f"  Market trend: {market_info['trend']}, Position: {market_info.get('cycle_position', 'unknown')}")
            print(f"  Volatility: {market_info['volatility']:.4f}")
            
            if market_info['avg_cycle_length']:
                print(f"  Average cycle length: {market_info['avg_cycle_length']:.2f}")
            
            mask = (times >= training_start) & (times <= training_end)
            t_train = times[mask]
            p_train = prices[mask]
            
            price_mean = np.mean(p_train)
            price_min = np.min(p_train)
            price_max = np.max(p_train)
            price_last = p_train[-1]
            
            best_model = None
            best_model_name = None
            best_params = None
            best_mse = float('inf')
            best_predictions = None
            
            models_to_try = models[:min(max_attempts, len(models))]
            
            for model_func, model_name in models_to_try:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        if model_name == "Exponential" and (price_max / max(price_min, 0.001) > 100):
                            print(f"  Skipping {model_name} due to large price range")
                            continue
                        
                        p0 = np.ones(model_func.__code__.co_argcount - 1)
                        
                        if model_name == "Quadratic":
                            if len(t_train) > 2:
                                trend_coef = np.polyfit(t_train, p_train, 1)[0]
                                p0[0] = -0.0001 if trend_coef < 0 else 0.0001
                        elif model_name == "HoltWinters":
                            p0[0] = price_mean  # level
                            p0[1] = 0  # trend
                            p0[2] = 0.1  # season1
                            p0[3] = 0.1  # season2
                        elif model_name == "Fourier":
                            if market_info['avg_cycle_length']:
                                p0[5] = 2 * np.pi / market_info['avg_cycle_length']  # omega
                            else:
                                p0[5] = 0.1
                        
                        bounds = ([-1000] * len(p0), [1000] * len(p0))
                        
                        if model_name == "Sin+Linear":
                            bounds[0][1] = 0.001
                            bounds[1][1] = 10
                        elif model_name == "DampedSin":
                            bounds[0][1] = 0.0001
                            bounds[1][1] = 0.1
                            bounds[0][2] = 0.001
                            bounds[1][2] = 10
                            p0[1] = 0.01
                            p0[2] = 0.1
                        elif model_name == "Logistic":
                            mid_x = np.mean(t_train)
                            p0[0] = price_max - price_min
                            p0[1] = 1.0
                            p0[2] = 0.1
                            p0[3] = price_min
                            bounds[0][2] = 0.001
                            bounds[1][2] = 1.0
                        elif model_name == "Exponential":
                            p0[1] = 0.001
                            bounds[0][1] = -0.1
                            bounds[1][1] = 0.1
                        elif model_name == "Fourier":
                            bounds[0][5] = 0.001  # min frequency
                            bounds[1][5] = 10.0  # max frequency
                        
                        params, _ = curve_fit(
                            model_func, t_train, p_train,
                            p0=p0,
                            bounds=bounds,
                            maxfev=20000,
                            method='trf'
                        )
                        
                        predictions = model_func(t_train, *params)
                        mse = np.mean((predictions - p_train)**2)
                        
                        if mse < best_mse:
                            best_mse = mse
                            best_model = model_func
                            best_model_name = model_name
                            best_params = params
                            best_predictions = predictions
                    
                    print(f"  Tried {model_name}: MSE = {mse:.4f}")
                    
                except Exception as e:
                    print(f"  {model_name} fitting failed: {str(e)}")
                    continue
            
            if best_model is None:
                print("  All models failed, falling back to linear regression")
                coeffs = np.polyfit(t_train, p_train, 1)
                best_predictions = np.polyval(coeffs, t_train)
                best_model_name = "Linear fallback"
                
                def linear_fallback(x):
                    return np.polyval(coeffs, x)
                
                forecast_times = np.linspace(training_end + 1, forecast_time, int(forecast_time - training_end))
                raw_forecast_values = linear_fallback(forecast_times)
            else:
                print(f"  Selected {best_model_name} model with MSE: {best_mse:.4f}")
                
                forecast_times = np.linspace(training_end + 1, forecast_time, int(forecast_time - training_end))
                raw_forecast_values = best_model(forecast_times, *best_params)
            
            adjusted_forecast_values = raw_forecast_values.copy()
            
            forecast_max = np.max(raw_forecast_values)
            forecast_min = np.min(raw_forecast_values)
            
            if len(raw_forecast_values) >= 2:
                forecast_direction = "upward" if raw_forecast_values[-1] > raw_forecast_values[0] else "downward"
            else:
                forecast_direction = "flat"
            
            price_range = price_max - price_min
            extreme_threshold = 1.5
            
            extreme_high = forecast_max > price_max + extreme_threshold * price_range
            extreme_low = forecast_min < price_min - extreme_threshold * price_range
            
            cycle_position = market_info.get('cycle_position', 'unknown')
            
            if extreme_high or extreme_low or (cycle_position in ['near_peak', 'near_trough']):
                print(f"  Applying market-aware mean reversion (strength={reversion_strength:.2f})")
                
                if cycle_position == 'near_peak' and forecast_direction == 'upward':
                    print("  Near market peak with upward forecast - increasing reversion")
                    effective_reversion = min(reversion_strength * 1.5, 0.9)
                    reversion_target = max(price_mean, price_min + price_range * 0.4)
                    
                elif cycle_position == 'near_trough' and forecast_direction == 'downward':
                    print("  Near market trough with downward forecast - increasing reversion")
                    effective_reversion = min(reversion_strength * 1.5, 0.9)
                    reversion_target = min(price_mean, price_max - price_range * 0.4)
                    
                else:
                    effective_reversion = reversion_strength
                    reversion_target = price_mean
                
                for j in range(len(adjusted_forecast_values)):
                    t_factor = j / len(adjusted_forecast_values)
                    local_reversion = effective_reversion * t_factor
                    adjusted_forecast_values[j] = (1 - local_reversion) * raw_forecast_values[j] + local_reversion * reversion_target
            
            adjusted_forecast_values = np.maximum(adjusted_forecast_values, 0)
            
            actual_last_price = p_train[-1]
            model_price_at_training = best_predictions[-1] if best_predictions is not None else p_train[-1]
            price_at_forecast = adjusted_forecast_values[-1]
            predicted_return = ((price_at_forecast / actual_last_price) - 1) * 100
            
            full_times = np.concatenate([t_train, forecast_times])
            full_predictions = np.concatenate([best_predictions, adjusted_forecast_values])
            
            full_predictions = np.maximum(full_predictions, 0)
            
            curve_fit_pred[name] = {
                "model_predictions": (t_train, best_predictions),
                "full_reconstruction": (full_times, full_predictions),
                "raw_forecast": (forecast_times, raw_forecast_values),
                "price_at_training": model_price_at_training,
                "actual_last_price": actual_last_price,
                "price_at_forecast": price_at_forecast,
                "predicted_return": predicted_return,
                "training_end": training_end,
                "forecast_time": forecast_time,
                "model_name": best_model_name,
                "market_info": market_info
            }
        
        return curve_fit_pred

    def plot_market_aware_predictions(self, training_start=0, training_end=None, forecast_time=None, show_raw=True):
        if training_end is None:
            training_end = self.asset_times[0][-1]
            
        if forecast_time is None:
            forecast_time = training_end + 100
            
        curve_fit_results = self.market_aware_curve_fit_forecast(max_attempts=11)
        
        n_assets = len(self.asset_names)
        n_rows = (n_assets + 3) // 4
        n_cols = min(4, n_assets)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        if n_rows * n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
            
        for i, asset in enumerate(self.asset_names):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            times = np.array(self.asset_times[i])
            prices = np.array(self.asset_prices[i])
            
            mask = (times >= training_start) & (times <= training_end)
            t_train = times[mask]
            p_train = prices[mask]
            ax.plot(t_train, p_train, 'b-', label="Real Training Data", linewidth=2)
            
            if asset in curve_fit_results:
                res = curve_fit_results[asset]
                model_name = res.get("model_name", "CurveFit")
                
                recon_times, recon_values = res["full_reconstruction"]
                ax.plot(recon_times, recon_values, 'r-', label=f"Market-Adjusted {model_name}", linewidth=2)
                
                if show_raw and "raw_forecast" in res:
                    raw_times, raw_values = res["raw_forecast"]
                    t_combined = np.concatenate([t_train, raw_times])
                    p_combined = np.concatenate([res["model_predictions"][1], raw_values])
                    ax.plot(t_combined, p_combined, 'g--', label=f"Raw {model_name}", linewidth=1, alpha=0.6)
                
                ax.scatter([res["training_end"]], [res["price_at_training"]], 
                           marker='o', color='green', s=100, label=f"Model at Training End")
                ax.scatter([res["training_end"]], [res["actual_last_price"]],
                           marker='*', color='blue', s=100, label=f"Actual Last Price")
                ax.scatter([res["forecast_time"]], [res["price_at_forecast"]], 
                           marker='x', color='red', s=100, label=f"Forecast")
                
                market_info = res.get("market_info", {})
                if "peak_indices" in market_info and "trough_indices" in market_info:
                    peak_indices = market_info["peak_indices"]
                    trough_indices = market_info["trough_indices"]
                    
                    if len(peak_indices) > 0:
                        ax.scatter(times[peak_indices], prices[peak_indices], 
                                  marker='^', color='orange', s=80, alpha=0.6, label="Peaks")
                    
                    if len(trough_indices) > 0:
                        ax.scatter(times[trough_indices], prices[trough_indices], 
                                  marker='v', color='purple', s=80, alpha=0.6, label="Troughs")
                
                predicted_return = res["predicted_return"]
                ax.set_title(f"{asset} - {model_name} (Return from actual: {predicted_return:.2f}%)")
            else:
                ax.set_title(asset)
                
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            ax.legend(fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.show()
        
        print("\nSummary of Market-Aware Predictions for Each Asset:")
        print(f"{'Asset':<20} {'Best Model':<15} {'Actual Last':>15} {'Forecast Price':>15} {'Expected Return':>15}")
        print("-" * 80)
        
        for asset in self.asset_names:
            if asset in curve_fit_results:
                res = curve_fit_results[asset]
                model_name = res.get("model_name", "CurveFit")
                actual_price = res["actual_last_price"]
                forecast_price = res["price_at_forecast"]
                expected_return = res["predicted_return"]
                
                print(f"{asset:<20} {model_name:<15} {actual_price:15.4f} {forecast_price:15.4f} {expected_return:15.2f}%")
                
        positive_returns = [res["predicted_return"] for asset, res in curve_fit_results.items() 
                           if res["predicted_return"] > 0]
        negative_returns = [res["predicted_return"] for asset, res in curve_fit_results.items() 
                           if res["predicted_return"] <= 0]
        
        print("\nAggregate Statistics:")
        print(f"Assets with positive returns: {len(positive_returns)} ({len(positive_returns)/len(curve_fit_results)*100:.1f}%)")
        print(f"Assets with negative returns: {len(negative_returns)} ({len(negative_returns)/len(curve_fit_results)*100:.1f}%)")
        
        if positive_returns:
            print(f"Average positive return: {np.mean(positive_returns):.2f}%")
        if negative_returns:
            print(f"Average negative return: {np.mean(negative_returns):.2f}%")
        
        print(f"Overall average expected return: {np.mean([res['predicted_return'] for res in curve_fit_results.values()]):.2f}%")
        
        return curve_fit_results

    def baseline_forecast(self):
        baseline_pred = {}
        for i, name in enumerate(self.asset_names):
            times = np.array(self.asset_times[i])
            prices = np.array(self.asset_prices[i])
            training_start = times[0]
            training_end = times[-1]
            forecast_time = training_end + 100

            mask = (times >= training_start) & (times <= training_end)
            t_used = times[mask]
            p_used = prices[mask]
            coeffs = np.polyfit(t_used, p_used, deg=1)
            model_predictions = np.polyval(coeffs, t_used)
            price_at_training = np.polyval(coeffs, training_end)
            price_at_forecast = np.polyval(coeffs, forecast_time)
            predicted_return = ((price_at_forecast / price_at_training) - 1) * 100
            recon_times = np.linspace(training_start, forecast_time, int(forecast_time - training_start))
            linear_reconstruction = np.polyval(coeffs, recon_times)
            baseline_pred[name] = {
                "model_predictions": (t_used, model_predictions),
                "full_reconstruction": (recon_times, linear_reconstruction),
                "price_at_training": price_at_training,
                "actual_last_price": p_used[-1],
                "price_at_forecast": price_at_forecast,
                "predicted_return": ((price_at_forecast / p_used[-1]) - 1) * 100,
                "training_end": training_end,
                "forecast_time": forecast_time,
                "model_name": "Linear"
            }
        return baseline_pred