import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Machine learning and forecasting libraries
from sklearn.linear_model import Lasso
from scipy.signal import savgol_filter
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN


class StockPredictor:
    def __init__(self, data_folder="Bundle2"):
        """
        Initializes the predictor and loads asset data from the given folder.
        The expected file names must end with "Part2.txt".
        """
        self.data_folder = data_folder
        self.asset_names, self.asset_times, self.asset_prices = self.load_data()

    def load_data(self):
        """
        Loads asset data from text files in self.data_folder.
        Each file should have:
          - A first line with the asset name.
          - A second line with the number of data points.
          - Then each line contains a time and price.
        Returns lists of asset names, times, and prices.
        """
        asset_names = []
        asset_times = []
        asset_prices = []
        txt_files = [f for f in os.listdir(self.data_folder) if f.endswith("Part2.txt")]
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
        return asset_names, asset_times, asset_prices

    def plot_forecasts(self, forecast_results):
        """
        Plots the forecast results for up to 20 assets in a 5x4 grid.
        Historical prices are shown as black scatter points.
        For each asset:
          - A marker is plotted at the training end time.
          - A forecast marker is plotted at the forecast time.
          - If available, the full reconstruction is plotted as a line.
        """
        fig, axes = plt.subplots(5, 4, figsize=(20, 25))
        axes = axes.flatten()
        num_assets = min(len(self.asset_names), 20)

        for i in range(num_assets):
            ax = axes[i]
            asset = self.asset_names[i]
            times = np.array(self.asset_times[i])
            prices = np.array(self.asset_prices[i])
            ax.scatter(times, prices, label="Historical Prices", color="black")

            # Retrieve the forecast dictionary for this asset.
            forecast = forecast_results[asset]
            training_end = forecast.get("training_end")
            forecast_time = forecast.get("forecast_time")
            # Plot training marker at training_end.
            ax.scatter([training_end], [forecast["price_at_training"]],
                       marker="o", color="red", s=200,
                       label=f"Training End @ t={training_end}")
            # Plot forecast marker at forecast_time.
            ax.scatter([forecast_time], [forecast["price_at_forecast"]],
                       marker="x", color="red", s=200,
                       label=f"Forecast @ t={forecast_time}")

            if "full_reconstruction" in forecast:
                recon_times, reconstruction = forecast["full_reconstruction"]
                ax.plot(recon_times, reconstruction, label="Reconstruction",
                        color="red", linewidth=2)
            ax.set_title(asset)
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            ax.legend(fontsize='small')

        # Remove any unused subplots.
        for j in range(num_assets, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

        print(f"Found {len(self.asset_names)} assets.")
        print("First few asset names:", self.asset_names[:5])
        return self.asset_names, self.asset_times, self.asset_prices

    def baseline_forecast(self):
        """
        Simple linear forecast using a least-squares linear fit.
        For each asset:
          - Uses the entire data as training (from first to last time point).
          - Forecasts to time = (last time + 100).
        Returns a dictionary with forecast details.
        """
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
            predicted_return = (price_at_forecast - price_at_training) / price_at_training
            recon_times = np.linspace(training_start, forecast_time, int(forecast_time - training_start))
            linear_reconstruction = np.polyval(coeffs, recon_times)
            baseline_pred[name] = {
                "model_predictions": (t_used, model_predictions),
                "full_reconstruction": (recon_times, linear_reconstruction),
                "price_at_training": price_at_training,
                "price_at_forecast": price_at_forecast,
                "predicted_return": predicted_return,
                "training_end": training_end,
                "forecast_time": forecast_time
            }
        return baseline_pred

    def _build_candidate_library(self, x, candidate_frequencies):
        """
        Build a design matrix with a constant, linear, quadratic term,
        and for each candidate frequency, sine and cosine functions.
        """
        features = []
        feature_names = []
        features.append(np.ones_like(x))
        feature_names.append('1')
        features.append(x)
        feature_names.append('x')
        features.append(x**2)
        feature_names.append('x^2')
        for w in candidate_frequencies:
            features.append(np.sin(2 * np.pi * w * x))
            feature_names.append(f'sin(2π*{w:.2f}x)')
            features.append(np.cos(2 * np.pi * w * x))
            feature_names.append(f'cos(2π*{w:.2f}x)')
        X = np.column_stack(features)
        return X, feature_names

    def sparse_forecast(self, alpha=0.01):
        """
        Sparse regression forecast using FFT-based candidate frequencies and LASSO.
        For each asset the training period is taken as the full dataset,
        and the forecast time is set to training_end + 100.
        """
        sparse_pred = {}
        for i, name in enumerate(self.asset_names):
            times = np.array(self.asset_times[i])
            prices = np.array(self.asset_prices[i])
            training_start = times[0]
            training_end = times[-1]
            forecast_time = training_end + 100

            prices_denoised = savgol_filter(prices, window_length=11, polyorder=2)
            mask = (times >= training_start) & (times <= training_end)
            t_train = times[mask]
            p_train_denoised = prices_denoised[mask]
            N = len(t_train)
            T = t_train[1] - t_train[0] if N > 1 else 1
            fft_vals = np.fft.fft(p_train_denoised)
            freq = np.fft.fftfreq(N, T)
            pos_mask = freq > 0
            freq_pos = freq[pos_mask]
            fft_magnitude = np.abs(fft_vals)[pos_mask]
            threshold = np.mean(fft_magnitude) + np.std(fft_magnitude)
            candidate_frequencies = freq_pos[fft_magnitude > threshold]
            candidate_frequencies = np.unique(np.round(candidate_frequencies, 2))
            p_train_raw = prices[mask]
            coeffs_trend = np.polyfit(t_train, p_train_raw, 1)
            trend_train = np.polyval(coeffs_trend, t_train)
            p_train_detrended = p_train_denoised - trend_train
            X, _ = self._build_candidate_library(times, candidate_frequencies)
            X_train = X[mask, :]
            lasso = Lasso(alpha=alpha, max_iter=10000)
            lasso.fit(X_train, p_train_detrended)
            coefs = lasso.coef_
            intercept_sparse = lasso.intercept_
            p_detrended_reconstructed = intercept_sparse + X.dot(coefs)
            trend_full = np.polyval(coeffs_trend, times)
            p_reconstructed = np.maximum(p_detrended_reconstructed + trend_full, 0)
            def predict_new(x_new):
                X_new, _ = self._build_candidate_library(np.array([x_new]), candidate_frequencies)
                value = intercept_sparse + X_new.dot(coefs) + np.polyval(coeffs_trend, x_new)
                return max(value.item(), 0)
            price_at_training = predict_new(training_end)
            price_at_forecast = predict_new(forecast_time)
            predicted_return = (price_at_forecast - price_at_training) / price_at_training
            recon_times = np.linspace(training_start, forecast_time, int(forecast_time - training_start))
            X_new, _ = self._build_candidate_library(recon_times, candidate_frequencies)
            p_detrended_new = intercept_sparse + X_new.dot(coefs)
            p_reconstructed_new = np.maximum(p_detrended_new + np.polyval(coeffs_trend, recon_times), 0)
            sparse_pred[name] = {
                "model_predictions": (t_train, np.maximum(p_reconstructed[mask], 0)),
                "full_reconstruction": (recon_times, p_reconstructed_new),
                "price_at_training": price_at_training,
                "price_at_forecast": price_at_forecast,
                "predicted_return": predicted_return,
                "training_end": training_end,
                "forecast_time": forecast_time
            }
        return sparse_pred

    def arima_forecast(self, order=(1, 1, 1), use_log=True):
        """
        ARIMA forecast. For each asset the training period is the full dataset,
        and forecast time is training_end + 100.
        """
        arima_pred = {}
        for i, name in enumerate(self.asset_names):
            times = np.array(self.asset_times[i])
            prices = np.array(self.asset_prices[i])
            training_start = times[0]
            training_end = times[-1]
            forecast_time = training_end + 100

            mask = (times >= training_start) & (times <= training_end)
            t_train = times[mask]
            p_train = prices[mask]
            if use_log:
                if np.any(p_train <= 0):
                    raise ValueError(f"Asset {name} has non-positive prices; cannot use log transform.")
                p_train_trans = np.log(p_train)
            else:
                p_train_trans = p_train.copy()
            try:
                model = ARIMA(p_train_trans, order=order)
                model_fit = model.fit()
            except Exception as e:
                print(f"ARIMA fit failed for asset {name}: {e}")
                continue
            pred_train_trans = model_fit.predict(start=0, end=len(p_train_trans) - 1)
            if use_log:
                pred_train = np.exp(pred_train_trans)
            else:
                pred_train = pred_train_trans
            forecast_steps = int(forecast_time - training_end)
            if forecast_steps <= 0:
                raise ValueError("Forecast time must be greater than training end.")
            forecast_trans = model_fit.forecast(steps=forecast_steps)
            if use_log:
                forecast_values = np.exp(forecast_trans)
            else:
                forecast_values = forecast_trans
            forecast_times = np.arange(training_end + 1, forecast_time + 1)
            full_times = np.concatenate([t_train, forecast_times])
            full_predictions = np.maximum(np.concatenate([pred_train, forecast_values]), 0)
            price_at_training = pred_train[-1]
            price_at_forecast = forecast_values[-1]
            predicted_return = (price_at_forecast - price_at_training) / price_at_training
            arima_pred[name] = {
                "model_predictions": (t_train, pred_train),
                "full_reconstruction": (full_times, full_predictions),
                "price_at_training": price_at_training,
                "price_at_forecast": price_at_forecast,
                "predicted_return": predicted_return,
                "training_end": training_end,
                "forecast_time": forecast_time
            }
        return arima_pred

    def exponential_smoothing_forecast(self, trend='add', seasonal=None, seasonal_periods=None,
                                         use_multiplicative=False):
        """
        Forecast using Exponential Smoothing.
        For each asset the training period is the full dataset,
        and forecast time is training_end + 100.
        """
        exp_pred = {}
        for i, name in enumerate(self.asset_names):
            times = np.array(self.asset_times[i])
            prices = np.array(self.asset_prices[i])
            training_start = times[0]
            training_end = times[-1]
            forecast_time = training_end + 100

            mask = (times >= training_start) & (times <= training_end)
            t_train = times[mask]
            p_train = prices[mask]
            if use_multiplicative:
                if np.any(p_train <= 0):
                    print(f"Warning: Asset {name} has non-positive values. Multiplicative model not appropriate. Using additive model.")
                else:
                    trend = 'mul'
                    if seasonal is not None:
                        seasonal = 'mul'
            try:
                model = ExponentialSmoothing(p_train, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
                model_fit = model.fit()
            except Exception as e:
                print(f"Exponential Smoothing model fit failed for asset {name}: {e}")
                continue
            pred_train = np.maximum(model_fit.fittedvalues, 0)
            forecast_steps = int(forecast_time - training_end)
            if forecast_steps <= 0:
                raise ValueError("Forecast time must be greater than training end.")
            forecast_values = np.maximum(model_fit.forecast(steps=forecast_steps), 0)
            forecast_times = np.arange(training_end + 1, forecast_time + 1)
            full_times = np.concatenate([t_train, forecast_times])
            full_predictions = np.maximum(np.concatenate([pred_train, forecast_values]), 0)
            price_at_training = pred_train[-1]
            price_at_forecast = forecast_values[-1]
            predicted_return = (price_at_forecast - price_at_training) / price_at_training
            exp_pred[name] = {
                "model_predictions": (t_train, pred_train),
                "full_reconstruction": (full_times, full_predictions),
                "price_at_training": price_at_training,
                "price_at_forecast": price_at_forecast,
                "predicted_return": predicted_return,
                "training_end": training_end,
                "forecast_time": forecast_time
            }
        return exp_pred

    def naive_forecast(self, strategy="last"):
        """
        Naive forecast using sktime's NaiveForecaster.
        For each asset the training period is the full dataset,
        and forecast time is training_end + 100.
        """
        naive_pred = {}
        for i, name in enumerate(self.asset_names):
            times = np.array(self.asset_times[i])
            prices = np.array(self.asset_prices[i])
            training_start = times[0]
            training_end = times[-1]
            forecast_time = training_end + 100

            mask = (times >= training_start) & (times <= training_end)
            t_train = times[mask]
            p_train = prices[mask]
            if strategy == "last":
                in_sample_predictions = np.empty_like(p_train)
                in_sample_predictions[0] = p_train[0]
                if len(p_train) > 1:
                    in_sample_predictions[1:] = p_train[:-1]
            elif strategy == "drift":
                if len(p_train) > 1:
                    drift = (p_train[-1] - p_train[0]) / (len(p_train) - 1)
                else:
                    drift = 0
                in_sample_predictions = np.empty_like(p_train)
                in_sample_predictions[0] = p_train[0]
                for j in range(1, len(p_train)):
                    in_sample_predictions[j] = p_train[j - 1] + drift
            elif strategy == "mean":
                mean_value = np.mean(p_train)
                in_sample_predictions = np.full_like(p_train, fill_value=mean_value)
            else:
                raise ValueError("Unsupported strategy. Use 'last', 'drift', or 'mean'.")
            in_sample_predictions = np.maximum(in_sample_predictions, 0)
            forecast_steps = int(forecast_time - training_end)
            if forecast_steps <= 0:
                raise ValueError("Forecast time must be greater than training end.")
            y_train = pd.Series(p_train, index=pd.RangeIndex(start=int(t_train[0]),
                                                             stop=int(t_train[0]) + len(t_train)))
            forecaster = NaiveForecaster(strategy=strategy)
            forecaster.fit(y_train)
            fh = np.arange(1, forecast_steps + 1)
            y_forecast = forecaster.predict(fh)
            forecast_values = np.maximum(y_forecast.values, 0)
            forecast_times = np.arange(training_end + 1, forecast_time + 1)
            full_times = np.concatenate([t_train, forecast_times])
            full_predictions = np.concatenate([in_sample_predictions, forecast_values])
            price_at_training = p_train[-1]
            price_at_forecast = forecast_values[-1]
            predicted_return = (price_at_forecast - price_at_training) / price_at_training
            naive_pred[name] = {
                "model_predictions": (t_train, in_sample_predictions),
                "full_reconstruction": (full_times, full_predictions),
                "price_at_training": price_at_training,
                "price_at_forecast": price_at_forecast,
                "predicted_return": predicted_return,
                "training_end": training_end,
                "forecast_time": forecast_time
            }
        return naive_pred

    def decision_tree_forecast(self, seasonal_period=50, max_depth=50):
        """
        Forecast using a Decision Tree Regressor with conditional deseasonalization.
        For each asset the training period is the full dataset,
        and forecast time is training_end + 100.
        """
        dt_pred = {}
        for i, name in enumerate(self.asset_names):
            times = np.array(self.asset_times[i])
            prices = np.array(self.asset_prices[i])
            training_start = times[0]
            training_end = times[-1]
            forecast_time = training_end + 100

            mask = (times >= training_start) & (times <= training_end)
            t_train = times[mask]
            p_train = prices[mask]
            coeffs = np.polyfit(t_train, p_train, deg=1)
            trend_train = np.polyval(coeffs, t_train)
            p_detrended = p_train - trend_train
            if seasonal_period is not None and seasonal_period > 1:
                seasonal_effect = np.zeros(seasonal_period)
                count = np.zeros(seasonal_period)
                for idx, t in enumerate(t_train):
                    pos = int((t - training_start) % seasonal_period)
                    seasonal_effect[pos] += p_detrended[idx]
                    count[pos] += 1
                seasonal_effect = np.where(count > 0, seasonal_effect / count, 0)
                seasonal_train = np.array([seasonal_effect[int((t - training_start) % seasonal_period)]
                                           for t in t_train])
                p_deseasonalized = p_detrended - seasonal_train
            else:
                p_deseasonalized = p_detrended
                seasonal_train = np.zeros_like(p_train)
            dt_reg = DecisionTreeRegressor(max_depth=max_depth)
            dt_reg.fit(t_train.reshape(-1, 1), p_deseasonalized)
            pred_deseasonalized_train = dt_reg.predict(t_train.reshape(-1, 1))
            pred_train_reconstructed = pred_deseasonalized_train + trend_train
            if seasonal_period is not None and seasonal_period > 1:
                pred_train_reconstructed += seasonal_train
            forecast_times = np.arange(training_end + 1, forecast_time + 1)
            pred_deseasonalized_forecast = dt_reg.predict(forecast_times.reshape(-1, 1))
            trend_forecast = np.polyval(coeffs, forecast_times)
            if seasonal_period is not None and seasonal_period > 1:
                seasonal_forecast = np.array([seasonal_effect[int((t - training_start) % seasonal_period)]
                                              for t in forecast_times])
            else:
                seasonal_forecast = np.zeros_like(forecast_times)
            forecast_pred = pred_deseasonalized_forecast + trend_forecast + seasonal_forecast
            full_times = np.concatenate([t_train, forecast_times])
            full_predictions = np.concatenate([pred_train_reconstructed, forecast_pred])
            price_at_training = pred_train_reconstructed[-1]
            price_at_forecast = forecast_pred[-1]
            predicted_return = (price_at_forecast - price_at_training) / price_at_training
            dt_pred[name] = {
                "model_predictions": (t_train, pred_train_reconstructed),
                "full_reconstruction": (full_times, full_predictions),
                "price_at_training": price_at_training,
                "price_at_forecast": price_at_forecast,
                "predicted_return": predicted_return,
                "training_end": training_end,
                "forecast_time": forecast_time
            }
        return dt_pred

    def _create_dataset(self, data, look_back):
        """
        Helper function to create sliding-window sequences.
        Returns X (inputs) and y (targets).
        """
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:i + look_back])
            y.append(data[i + look_back])
        return np.array(X), np.array(y)

    def rnn_forecast(self, look_back=10, epochs=20, batch_size=64):
        """
        Forecast using a simple RNN. Creates sliding-window sequences,
        trains an RNN on normalized training data, and then recursively forecasts.
        For each asset the training period is the full dataset,
        and forecast time is training_end + 100.
        """
        rnn_pred = {}
        for i, name in enumerate(self.asset_names):
            times = np.array(self.asset_times[i])
            prices = np.array(self.asset_prices[i])
            training_start = times[0]
            training_end = times[-1]
            forecast_time = training_end + 100

            mask = (times >= training_start) & (times <= training_end)
            t_train = times[mask]
            p_train = prices[mask]
            if len(p_train) <= look_back:
                raise ValueError(f"Not enough training data for asset {name} with look_back = {look_back}.")
            train_min = p_train.min()
            train_max = p_train.max()
            p_train_norm = (p_train - train_min) / (train_max - train_min)
            X_train, y_train = self._create_dataset(p_train_norm, look_back)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            model = Sequential()
            model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(SimpleRNN(units=50, return_sequences=False))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            pred_train_norm = model.predict(X_train, verbose=0).flatten()
            pred_train = pred_train_norm * (train_max - train_min) + train_min
            pred_train = np.maximum(pred_train, 0)
            t_train_pred = t_train[look_back:]
            forecast_steps = int(forecast_time - training_end)
            if forecast_steps <= 0:
                raise ValueError("Forecast time must be greater than training end.")
            last_sequence = p_train_norm[-look_back:].tolist()
            forecasted_norm = []
            for _ in range(forecast_steps):
                input_seq = np.array(last_sequence[-look_back:]).reshape((1, look_back, 1))
                next_val_norm = model.predict(input_seq, verbose=0)[0, 0]
                forecasted_norm.append(next_val_norm)
                last_sequence.append(next_val_norm)
            forecasted = np.array(forecasted_norm) * (train_max - train_min) + train_min
            forecasted = np.maximum(forecasted, 0)
            forecast_times = np.arange(training_end + 1, forecast_time + 1)
            full_times = np.concatenate([t_train_pred, forecast_times])
            full_predictions = np.concatenate([pred_train, forecasted])
            full_predictions = np.maximum(full_predictions, 0)
            price_at_training = pred_train[-1]
            price_at_forecast = forecasted[-1]
            predicted_return = (price_at_forecast - price_at_training) / price_at_training
            rnn_pred[name] = {
                "model_predictions": (t_train_pred, pred_train),
                "full_reconstruction": (full_times, full_predictions),
                "price_at_training": price_at_training,
                "price_at_forecast": price_at_forecast,
                "predicted_return": predicted_return,
                "training_end": training_end,
                "forecast_time": forecast_time
            }
        return rnn_pred

    def aggregate_and_plot_mean_median(self, methods, training_start=0, training_end=100, forecast_time=200):
        """
        For each asset, this function:
          - Prints a table of predicted training price, forecast price, and expected return (in %)
            for each forecasting method in the provided 'methods' dictionary.
          - Computes the pointwise median and mean forecasts across the selected methods.
          - Displays two separate 5×4 grids: one for the median forecast and one for the mean forecast.
        
        Expected return is computed as (forecast / training_price)*100%.
        
        Returns a dictionary with two keys: "median" and "mean". Each value is a dictionary where each asset
        is represented with keys:
          - "model_predictions": (training_times, aggregated in-sample predictions) – the training period is filled with the aggregated training price.
          - "full_reconstruction": (common_times, aggregated full forecast)
          - "price_at_training": aggregated training price (scalar)
          - "price_at_forecast": aggregated forecast price (scalar)
          - "predicted_return": aggregated expected return (scalar)
        """
        # 'methods' is a dictionary mapping method names to forecast result dictionaries.
        # Create a common time axis.
        common_times = np.linspace(training_start, forecast_time, int(forecast_time - training_start))
        
        # Containers for aggregated results.
        median_results = {}
        mean_results = {}
        
        # Loop over assets.
        for idx, asset in enumerate(self.asset_names):
            print(f"\n{'='*60}\nAsset: {asset}")
            
            # Get asset training data.
            times = np.array(self.asset_times[idx])
            prices = np.array(self.asset_prices[idx])
            mask = (times >= training_start) & (times <= training_end)
            t_train = times[mask]
            real_training_price = prices[mask][-1]
            print(f"Real training price (last value in training data): {real_training_price:.4f}")
            
            # Initialize lists to collect scalar predictions and aligned full forecasts.
            training_preds = []    # from each method: price_at_training
            forecast_preds = []    # from each method: price_at_forecast
            exp_returns = []       # expected return in percentage
            full_predictions_aligned = []  # each element is an array aligned on common_times
            
            # Print table header.
            header = f"{'Method':<15} {'TrainPred':>10} {'Forecast':>10} {'Return (%)':>12}"
            print(header)
            print("-" * len(header))
            
            for method_name, result in methods.items():
                if asset not in result:
                    continue
                res = result[asset]
                train_pred = res["price_at_training"]
                forecast_pred = res["price_at_forecast"]
                exp_return = (forecast_pred / train_pred) * 100.0
                print(f"{method_name:<15} {train_pred:10.4f} {forecast_pred:10.4f} {exp_return:12.2f}")
                training_preds.append(train_pred)
                forecast_preds.append(forecast_pred)
                exp_returns.append(exp_return)
                # Interpolate full reconstruction on common_times.
                method_times, method_preds = res["full_reconstruction"]
                aligned = np.interp(common_times, method_times, method_preds)
                full_predictions_aligned.append(aligned)
            
            # Compute aggregated scalar values.
            median_train = np.median(training_preds)
            median_forecast = np.median(forecast_preds)
            median_return = np.median(exp_returns)
            mean_train = np.mean(training_preds)
            mean_forecast = np.mean(forecast_preds)
            mean_return = np.mean(exp_returns)
            print("-" * len(header))
            print(f"{'Median':<15} {median_train:10.4f} {median_forecast:10.4f} {median_return:12.2f}")
            print(f"{'Mean':<15} {mean_train:10.4f} {mean_forecast:10.4f} {mean_return:12.2f}")
            
            # Compute pointwise aggregated forecasts.
            full_predictions_aligned = np.array(full_predictions_aligned)
            median_full = np.median(full_predictions_aligned, axis=0)
            mean_full = np.mean(full_predictions_aligned, axis=0)
            
            # Save aggregated results for this asset.
            median_results[asset] = {
                "model_predictions": (t_train, np.full_like(t_train, median_train)),
                "full_reconstruction": (common_times, median_full),
                "price_at_training": median_train,
                "price_at_forecast": median_forecast,
                "predicted_return": median_return
            }
            mean_results[asset] = {
                "model_predictions": (t_train, np.full_like(t_train, mean_train)),
                "full_reconstruction": (common_times, mean_full),
                "price_at_training": mean_train,
                "price_at_forecast": mean_forecast,
                "predicted_return": mean_return
            }
        
        # Plotting: Two separate 5x4 charts.
        n_assets = len(self.asset_names)
        n_rows, n_cols = 5, 4
        
        # Figure for Median forecasts.
        fig_med, axes_med = plt.subplots(n_rows, n_cols, figsize=(20, 25), sharex=True, sharey=False)
        axes_med = axes_med.flatten()
        for idx, asset in enumerate(self.asset_names):
            ax = axes_med[idx]
            ax.plot(common_times, median_results[asset]["full_reconstruction"][1],
                    label="Median Forecast", color='black', linewidth=2)
            # Plot each method's forecast.
            for method_name, result in methods.items():
                if asset in result:
                    m_times, m_preds = result[asset]["full_reconstruction"]
                    aligned = np.interp(common_times, m_times, m_preds)
                    ax.plot(common_times, aligned, label=method_name, linestyle="--", alpha=0.6)
            # Plot real training data.
            asset_idx = self.asset_names.index(asset)
            mask = (np.array(self.asset_times[asset_idx]) >= training_start) & (np.array(self.asset_times[asset_idx]) <= training_end)
            t_train_real = np.array(self.asset_times[asset_idx])[mask]
            ax.plot(t_train_real, np.array(self.asset_prices[asset_idx])[mask],
                    label="Real Training", color="blue", linewidth=2)
            ax.set_title(asset)
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            ax.legend(fontsize=8)
        for j in range(idx+1, n_rows*n_cols):
            axes_med[j].axis('off')
        fig_med.tight_layout()
        
        # Figure for Mean forecasts.
        fig_mean, axes_mean = plt.subplots(n_rows, n_cols, figsize=(20, 25), sharex=True, sharey=False)
        axes_mean = axes_mean.flatten()
        for idx, asset in enumerate(self.asset_names):
            ax = axes_mean[idx]
            ax.plot(common_times, mean_results[asset]["full_reconstruction"][1],
                    label="Mean Forecast", color='black', linewidth=2)
            for method_name, result in methods.items():
                if asset in result:
                    m_times, m_preds = result[asset]["full_reconstruction"]
                    aligned = np.interp(common_times, m_times, m_preds)
                    ax.plot(common_times, aligned, label=method_name, linestyle="--", alpha=0.6)
            asset_idx = self.asset_names.index(asset)
            mask = (np.array(self.asset_times[asset_idx]) >= training_start) & (np.array(self.asset_times[asset_idx]) <= training_end)
            t_train_real = np.array(self.asset_times[asset_idx])[mask]
            ax.plot(t_train_real, np.array(self.asset_prices[asset_idx])[mask],
                    label="Real Training", color="blue", linewidth=2)
            ax.set_title(asset)
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            ax.legend(fontsize=8)
        for j in range(idx+1, n_rows*n_cols):
            axes_mean[j].axis('off')
        fig_mean.tight_layout()
        
        plt.show()
        
        return {"median": median_results, "mean": mean_results}


# Example usage when running as a script.
if __name__ == "__main__":
    sp = StockPredictor(data_folder="Bundle2")
    
    # Run each forecasting method.
    linear_results = sp.baseline_forecast()
    sparse_results = sp.sparse_forecast(alpha=0.01)
    arima_results = sp.arima_forecast(order=(2, 1, 2), use_log=True)
    exp_results = sp.exponential_smoothing_forecast(trend='add', seasonal=None, seasonal_periods=None,
                                                    use_multiplicative=True)
    naive_results = sp.naive_forecast(strategy="last")
    dt_results = sp.decision_tree_forecast(seasonal_period=50, max_depth=50)
    rnn_results = sp.rnn_forecast(look_back=30, epochs=20, batch_size=64)
    
    # Optionally, plot individual forecasts.
    sp.plot_forecasts(linear_results)
    sp.plot_forecasts(sparse_results)
    sp.plot_forecasts(arima_results)
    sp.plot_forecasts(exp_results)
    sp.plot_forecasts(naive_results)
    sp.plot_forecasts(dt_results)
    sp.plot_forecasts(rnn_results)
    
    # Create a dictionary of selected methods (dynamically defined).
    methods = {
        'RNN': rnn_results,
        'DecisionTree': dt_results,
        'Naive': naive_results,
        'ExpSmoothing': exp_results,
        'ARIMA': arima_results,
        'Sparse': sparse_results,
        'Linear': linear_results
    }
    
    # Aggregate and plot mean and median forecasts.
    aggregated = sp.aggregate_and_plot_mean_median(methods,
                                                   training_start=0,
                                                   training_end=sp.asset_times[0][-1],
                                                   forecast_time=sp.asset_times[0][-1] + 100)
