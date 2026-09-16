
# -*- coding: utf-8 -*-
"""
Machine-learning workflow for monthly GPP attribution.

A single fitted model is used for all counterfactual scenario groups at each pixel:

Reference and legacy-by-lag predictions
reference prediction
replace lag t−3
replace lags t−2 to t−3
replace lags t−1 to t−3
Process-level predictions
concurrent climate
lagged climate
antecedent GPP
Individual climate-driver predictions
lagged climate-driver replacement
concurrent climate-driver replacement

For each target pixel, one local model is fitted and then applied to the
reference and counterfactual predictor sets. Supported algorithms are
XGBoost, Random Forest and LightGBM.
"""

import argparse
import os
import time
from dataclasses import dataclass, field
from math import sqrt
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor


REGION_CONFIG = {
    "china": {
        "label": "China",
        "lat_start": 220,
        "lat_end": 400,
        "lon_start": 100,
        "lon_end": 600,
        "target_months": ("Aug", "Sep", "Oct"),
    },
    "europe": {
        "label": "Europe",
        "lat_start": 170,
        "lat_end": 370,
        "lon_start": 60,
        "lon_end": 410,
        "target_months": ("Jul", "Aug", "Sep"),
    },
    "us": {
        "label": "US",
        "lat_start": 100,
        "lat_end": 449,
        "lon_start": 120,
        "lon_end": 400,
        "target_months": ("Jul", "Aug", "Sep"),
    },
}


@dataclass
class Config:
    # Input/output
    base_dir: str = ""
    output_root: str = "./project_data/model_predictions"
    region_name: str = "china"
    region_label: str = "China"
    model_name: str = "xgboost"
    reference_out_dir: str = ""
    legacy_out_dir: str = ""
    process_out_dir: str = ""
    lagged_driver_out_dir: str = ""
    concurrent_driver_out_dir: str = ""

    # Variables and months
    target_month: str = "Sep"
    month_order: List[str] = field(default_factory=lambda: [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ])
    var_names: List[str] = field(default_factory=lambda: [
        "sm", "pre", "tas", "VPD", "ssrd", "gpp"
    ])
    climate_factors: List[str] = field(default_factory=lambda: [
        "sm", "pre", "tas", "VPD", "ssrd"
    ])
    factor_labels: Dict[str, str] = field(default_factory=lambda: {
        "sm": "soil_moisture",
        "pre": "precipitation",
        "tas": "air_temperature",
        "VPD": "vapor_pressure_deficit",
        "ssrd": "solar_radiation",
    })

    # Model settings
    n_years: int = 23          # 2000-2022
    clim_index: int = 23       # index 23 stores the climatology layer
    window_radius: int = 5     # 11 x 11 moving window
    test_size: float = 0.1
    split_random_state: int = 3
    min_train_samples: int = 20

    # Processing range
    lat_start: int = 220
    lat_end: int = 400
    lon_start: int = 100
    lon_end: int = 600

    # Progress display
    progress_every: int = 50


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(
        description="Pixel-wise ML attribution of monthly FluxSat GPP anomalies."
    )

    p.add_argument(
        "--region",
        choices=["china", "europe", "us"],
        required=True,
        help="Study region; this automatically sets the spatial processing range.",
    )
    p.add_argument(
        "--model",
        choices=["xgboost", "rf", "lightgbm"],
        required=True,
        help="Machine-learning algorithm.",
    )
    p.add_argument(
        "--target-month",
        required=True,
        help="Target month. China: Aug/Sep/Oct; Europe and US: Jul/Aug/Sep.",
    )
    p.add_argument(
        "--base-dir",
        default=None,
        help=(
            "Directory containing the regional monthly input NetCDF files. "
            "If omitted, ./project_data/inputs/<Region> is used."
        ),
    )
    p.add_argument(
        "--output-root",
        default=cfg.output_root,
        help="Root prediction directory (default: ./project_data/model_predictions).",
    )
    p.add_argument("--progress-every", type=int, default=cfg.progress_every)

    a = p.parse_args()

    region = REGION_CONFIG[a.region]
    cfg.region_name = a.region
    cfg.region_label = region["label"]
    cfg.model_name = a.model
    cfg.target_month = a.target_month
    cfg.output_root = a.output_root
    cfg.progress_every = a.progress_every

    if cfg.target_month not in region["target_months"]:
        allowed = ", ".join(region["target_months"])
        raise ValueError(
            f"Invalid target month '{cfg.target_month}' for {cfg.region_label}. "
            f"Allowed target months: {allowed}"
        )

    cfg.lat_start = region["lat_start"]
    cfg.lat_end = region["lat_end"]
    cfg.lon_start = region["lon_start"]
    cfg.lon_end = region["lon_end"]

    cfg.base_dir = (
        a.base_dir
        if a.base_dir is not None
        else os.path.join(".", "project_data", "inputs", cfg.region_label)
    )

    model_folder = {
        "xgboost": "XGBoost",
        "rf": "RandomForest",
        "lightgbm": "LightGBM",
    }[cfg.model_name]

    model_root = os.path.join(cfg.output_root, cfg.region_label, model_folder)
    cfg.reference_out_dir = os.path.join(model_root, "reference")
    cfg.legacy_out_dir = os.path.join(model_root, "legacy_by_lag")
    cfg.process_out_dir = os.path.join(model_root, "process_partition")
    cfg.lagged_driver_out_dir = os.path.join(model_root, "climate_drivers", "lagged")
    cfg.concurrent_driver_out_dir = os.path.join(model_root, "climate_drivers", "concurrent")
    return cfg


def get_months(cfg: Config) -> List[str]:
    if cfg.target_month not in cfg.month_order:
        raise ValueError(f"target_month={cfg.target_month} is not in month_order")

    target_idx = cfg.month_order.index(cfg.target_month)
    months = cfg.month_order[target_idx - 3: target_idx + 1]

    if len(months) != 4:
        raise ValueError("At least three preceding months are required for the four-month predictor window")

    return months


def get_file_path(cfg: Config, var_name: str, month: str) -> str:
    file_stems = {
        "gpp": "gpp",
        "sm": "soil_moisture",
        "pre": "precipitation",
        "tas": "air_temperature",
        "VPD": "vapor_pressure_deficit",
        "ssrd": "solar_radiation",
    }
    filename = f"{file_stems[var_name]}_{month}.nc"
    return os.path.join(cfg.base_dir, filename)


def read_nc_var(cfg: Config, var_name: str, month: str) -> np.ndarray:
    path = get_file_path(cfg, var_name, month)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    public_var_names = {
        "gpp": "gpp",
        "sm": "soil_moisture",
        "pre": "precipitation",
        "tas": "air_temperature",
        "VPD": "vapor_pressure_deficit",
        "ssrd": "solar_radiation",
    }

    with Dataset(path) as ds:
        nc_var = public_var_names[var_name]
        arr = ds.variables[nc_var][:]
        arr = np.ma.filled(arr, np.nan).astype("float32")

    if arr.shape[0] <= cfg.clim_index:
        raise ValueError(f"Insufficient time dimension in {path}; cannot read clim_index={cfg.clim_index}")

    return arr


def read_all_data(cfg: Config, months: List[str]):
    with Dataset(get_file_path(cfg, "gpp", cfg.target_month)) as ds_ref:
        lon = ds_ref.variables["lon"][:]
        lat = ds_ref.variables["lat"][:]

    data: Dict[str, Dict[str, np.ndarray]] = {}
    clim: Dict[str, Dict[str, np.ndarray]] = {}

    for mon in months:
        data[mon] = {}
        clim[mon] = {}
        for var in cfg.var_names:
            arr = read_nc_var(cfg, var, mon)
            data[mon][var] = arr
            clim[mon][var] = arr[cfg.clim_index, :, :]

    return data, clim, lat, lon


def feature_names(months: List[str], var_names: List[str]) -> List[str]:
    return [f"{m}_{v}" for m in months for v in var_names]


ReplaceRule = Callable[[str, str], bool]


def no_replace(month: str, var_name: str) -> bool:
    return False


def build_pixel_matrix(
    cfg: Config,
    data,
    clim,
    months: List[str],
    lat_i: int,
    lon_i: int,
    replace_rule: ReplaceRule = no_replace,
) -> np.ndarray:
    """Assemble predictors for one pixel."""
    n_features = len(months) * len(cfg.var_names)
    mat = np.full((cfg.n_years, n_features), np.nan, dtype="float32")

    col = 0
    for mon in months:
        for var in cfg.var_names:
            if replace_rule(mon, var):
                mat[:, col] = clim[mon][var][lat_i, lon_i]
            else:
                mat[:, col] = data[mon][var][:cfg.n_years, lat_i, lon_i]
            col += 1

    return mat


def build_training_dataset(
    cfg: Config,
    data,
    clim,
    months: List[str],
    lat_i: int,
    lon_i: int,
    nlat: int,
    nlon: int,
) -> np.ndarray:
    """Assemble the local training sample."""
    blocks = []

    for ii in range(lat_i - cfg.window_radius, lat_i + cfg.window_radius + 1):
        for jj in range(lon_i - cfg.window_radius, lon_i + cfg.window_radius + 1):
            if 0 <= ii < nlat and 0 <= jj < nlon:
                blocks.append(
                    build_pixel_matrix(cfg, data, clim, months, ii, jj, no_replace)
                )

    if not blocks:
        return np.empty((0, len(months) * len(cfg.var_names)), dtype="float32")

    return pd.DataFrame(np.vstack(blocks)).dropna().to_numpy(dtype="float32")


def build_prediction_dataset(
    cfg: Config,
    data,
    clim,
    months: List[str],
    lat_i: int,
    lon_i: int,
    replace_rule: ReplaceRule,
    y_col: int,
):
    mat = build_pixel_matrix(cfg, data, clim, months, lat_i, lon_i, replace_rule)
    clean = pd.DataFrame(mat).dropna()
    X_pred = clean.drop(columns=[y_col])
    y_true = clean.iloc[:, y_col]
    return X_pred, y_true


def make_model(cfg: Config):
    """Create the selected regression model."""
    if cfg.model_name == "xgboost":
        return xgb.XGBRegressor(
            max_depth=3,
            learning_rate=0.05,
            n_estimators=300,
            objective="reg:squarederror",
            random_state=52,
        )

    if cfg.model_name == "rf":
        return RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            min_samples_leaf=5,
            min_samples_split=10,
            max_features="sqrt",
            bootstrap=True,
            random_state=52,
            n_jobs=-1,
        )

    if cfg.model_name == "lightgbm":
        return LGBMRegressor(
            objective="regression",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=16,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=52,
            n_jobs=1,
            verbosity=-1,
        )

    raise ValueError(f"Unsupported model: {cfg.model_name}")


def calc_eval(y_true, y_pred) -> Tuple[float, float, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    rss = float(np.sum((np.asarray(y_pred) - np.asarray(y_true)) ** 2))
    return float(mse), float(rmse), rss


def save_3d_nc(
    filename: str,
    varname: str,
    data_array: np.ndarray,
    time_values: Iterable[int],
    lat_values,
    lon_values,
):
    time_values = np.asarray(list(time_values))

    if data_array.shape[0] != len(time_values):
        raise ValueError(f"Time dimension mismatch: {data_array.shape[0]} vs {len(time_values)}")

    ds = xr.Dataset(
        data_vars={varname: (("time", "lat", "lon"), data_array.astype("float32"))},
        coords={"time": time_values, "lat": lat_values, "lon": lon_values},
    )
    ds[varname].attrs["long_name"] = varname
    ds[varname].attrs["units"] = "gC m-2 d-1" if varname == "prediction" else "1"
    ds.to_netcdf(filename, encoding={varname: {"zlib": True, "complevel": 4}})
    ds.close()


def init_outputs(names: Iterable[str], cfg: Config, nlat: int, nlon: int):
    data_pres = {
        name: np.full((cfg.n_years, nlat, nlon), np.nan, dtype="float32")
        for name in names
    }
    result_mse = {
        name: np.full((nlat, nlon), np.nan, dtype="float32")
        for name in names
    }
    result_rmse = {
        name: np.full((nlat, nlon), np.nan, dtype="float32")
        for name in names
    }
    result_rss = {
        name: np.full((nlat, nlon), np.nan, dtype="float32")
        for name in names
    }
    return data_pres, result_mse, result_rmse, result_rss


def store_prediction(
    cfg: Config,
    data_pres: Dict[str, np.ndarray],
    result_mse: Dict[str, np.ndarray],
    result_rmse: Dict[str, np.ndarray],
    result_rss: Dict[str, np.ndarray],
    scenario_name: str,
    lat_i: int,
    lon_i: int,
    y_true: pd.Series,
    y_pred: np.ndarray,
):
    for out_pos, year_idx in enumerate(y_true.index):
        if 0 <= year_idx < cfg.n_years:
            data_pres[scenario_name][year_idx, lat_i, lon_i] = y_pred[out_pos]

    mse, rmse, rss = calc_eval(y_true.to_numpy(), y_pred)
    result_mse[scenario_name][lat_i, lon_i] = mse
    result_rmse[scenario_name][lat_i, lon_i] = rmse
    result_rss[scenario_name][lat_i, lon_i] = rss


def write_feature_names(out_dirs: List[str], target_month: str, names_without_y: List[str]):
    for out_dir in out_dirs:
        with open(os.path.join(out_dir, f"feature_names_{target_month}.txt"), "w", encoding="utf-8") as f:
            for i, name in enumerate(names_without_y):
                f.write(f"{i}\t{name}\n")


def write_model_evaluate(out_dirs: List[str], cfg: Config, lat, lon, model_mse, model_rmse, model_rss):
    model_eval = np.stack([model_mse, model_rmse, model_rss])
    for out_dir in out_dirs:
        save_3d_nc(
            os.path.join(out_dir, f"model_evaluates_{cfg.target_month}.nc"),
            "model_evaluate",
            model_eval,
            np.arange(1, 4),
            lat,
            lon,
        )


def main():
    start_time = time.perf_counter()
    cfg = parse_args()

    for out_dir in [
        cfg.reference_out_dir,
        cfg.legacy_out_dir,
        cfg.process_out_dir,
        cfg.lagged_driver_out_dir,
        cfg.concurrent_driver_out_dir,
    ]:
        os.makedirs(out_dir, exist_ok=True)

    months = get_months(cfg)
    data, clim, lat, lon = read_all_data(cfg, months)
    nlat, nlon = len(lat), len(lon)

    lat_start = max(cfg.window_radius, cfg.lat_start)
    lat_end = min(nlat - cfg.window_radius, cfg.lat_end)
    lon_start = max(cfg.window_radius, cfg.lon_start)
    lon_end = min(nlon - cfg.window_radius, cfg.lon_end)

    names = feature_names(months, cfg.var_names)
    y_col = names.index(f"{cfg.target_month}_gpp")
    names_without_y = list(np.delete(np.array(names), y_col))

    # Counterfactual scenarios
    legacy_rules: Dict[str, ReplaceRule] = {
        "replace_lag1_to_lag3": lambda mon, var: mon in set(months[:3]),
        "replace_lag2_to_lag3": lambda mon, var: mon in set(months[:2]),
        "replace_lag3": lambda mon, var: mon in set(months[:1]),
        "reference": lambda mon, var: False,
    }

    process_rules: Dict[str, ReplaceRule] = {
        "CCE": lambda mon, var: (mon == cfg.target_month and var != "gpp"),
        "LCE": lambda mon, var: (mon != cfg.target_month and var != "gpp"),
        "VGC": lambda mon, var: (mon != cfg.target_month and var == "gpp"),
    }

    lagged_driver_rules: Dict[str, ReplaceRule] = {}
    concurrent_driver_rules: Dict[str, ReplaceRule] = {}

    for factor in cfg.climate_factors:
        label = cfg.factor_labels[factor]
        lagged_driver_rules[label] = (
            lambda mon, var, factor=factor: (mon != cfg.target_month and var == factor)
        )
        concurrent_driver_rules[label] = (
            lambda mon, var, factor=factor: (mon == cfg.target_month and var == factor)
        )

    legacy_pre, legacy_mse, legacy_rmse, legacy_rss = init_outputs(
        legacy_rules.keys(), cfg, nlat, nlon
    )
    process_pre, process_mse, process_rmse, process_rss = init_outputs(
        process_rules.keys(), cfg, nlat, nlon
    )
    lagged_driver_pre, lagged_driver_mse, lagged_driver_rmse, lagged_driver_rss = init_outputs(
        lagged_driver_rules.keys(), cfg, nlat, nlon
    )
    concurrent_driver_pre, concurrent_driver_mse, concurrent_driver_rmse, concurrent_driver_rss = init_outputs(
        concurrent_driver_rules.keys(), cfg, nlat, nlon
    )

    model_mse = np.full((nlat, nlon), np.nan, dtype="float32")
    model_rmse = np.full((nlat, nlon), np.nan, dtype="float32")
    model_rss = np.full((nlat, nlon), np.nan, dtype="float32")

    total = (lat_end - lat_start) * (lon_end - lon_start)
    done = 0
    processed = 0
    skipped = 0

    print(
        f"start running: region={cfg.region_label}, model={cfg.model_name}, target_month={cfg.target_month}, months={months}, "
        f"lat={lat_start}:{lat_end}, lon={lon_start}:{lon_end}, total={total}",
        flush=True,
    )

    for lat_i in range(lat_start, lat_end):
        print(f"row start: lat_index={lat_i}", flush=True)

        for lon_i in range(lon_start, lon_end):
            done += 1

            train_data = build_training_dataset(cfg, data, clim, months, lat_i, lon_i, nlat, nlon)
            if train_data.shape[0] < cfg.min_train_samples:
                skipped += 1
                continue

            X = np.delete(train_data, y_col, axis=1)
            y = train_data[:, y_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=cfg.test_size,
                random_state=cfg.split_random_state,
            )

            model = make_model(cfg)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            mse, rmse, rss = calc_eval(y_test, y_test_pred)
            model_mse[lat_i, lon_i] = mse
            model_rmse[lat_i, lon_i] = rmse
            model_rss[lat_i, lon_i] = rss

            # Legacy timing
            for scenario_name, rule in legacy_rules.items():
                X_pred, y_true = build_prediction_dataset(
                    cfg, data, clim, months, lat_i, lon_i, rule, y_col
                )
                if len(y_true) >= 2:
                    y_pred = model.predict(X_pred.to_numpy(dtype="float32"))
                    store_prediction(
                        cfg, legacy_pre, legacy_mse, legacy_rmse, legacy_rss,
                        scenario_name, lat_i, lon_i, y_true, y_pred
                    )

            # CCE / LCE / VGC
            for scenario_name, rule in process_rules.items():
                X_pred, y_true = build_prediction_dataset(
                    cfg, data, clim, months, lat_i, lon_i, rule, y_col
                )
                if len(y_true) >= 2:
                    y_pred = model.predict(X_pred.to_numpy(dtype="float32"))
                    store_prediction(
                        cfg, process_pre, process_mse, process_rmse, process_rss,
                        scenario_name, lat_i, lon_i, y_true, y_pred
                    )

            # Lagged climate drivers
            for scenario_name, rule in lagged_driver_rules.items():
                X_pred, y_true = build_prediction_dataset(
                    cfg, data, clim, months, lat_i, lon_i, rule, y_col
                )
                if len(y_true) >= 2:
                    y_pred = model.predict(X_pred.to_numpy(dtype="float32"))
                    store_prediction(
                        cfg, lagged_driver_pre, lagged_driver_mse, lagged_driver_rmse, lagged_driver_rss,
                        scenario_name, lat_i, lon_i, y_true, y_pred
                    )

            # Concurrent climate drivers
            for scenario_name, rule in concurrent_driver_rules.items():
                X_pred, y_true = build_prediction_dataset(
                    cfg, data, clim, months, lat_i, lon_i, rule, y_col
                )
                if len(y_true) >= 2:
                    y_pred = model.predict(X_pred.to_numpy(dtype="float32"))
                    store_prediction(
                        cfg, concurrent_driver_pre, concurrent_driver_mse, concurrent_driver_rmse, concurrent_driver_rss,
                        scenario_name, lat_i, lon_i, y_true, y_pred
                    )

            processed += 1

            if cfg.progress_every and done % cfg.progress_every == 0:
                elapsed_h = (time.perf_counter() - start_time) / 3600
                pct = done / total * 100 if total else 100
                print(
                    f"progress: {done}/{total} ({pct:.2f}%) | "
                    f"processed={processed}, skipped={skipped} | "
                    f"elapsed={elapsed_h:.2f} h",
                    flush=True,
                )

    # Save predictions
    time_model = np.arange(1, cfg.n_years + 1)

    # Reference
    save_3d_nc(
        os.path.join(
            cfg.reference_out_dir,
            f"reference_prediction_{cfg.target_month}.nc",
        ),
        "prediction",
        legacy_pre["reference"],
        time_model,
        lat,
        lon,
    )

    legacy_public_names = {
        "replace_lag3": f"counterfactual_replace_lag3_{cfg.target_month}.nc",
        "replace_lag2_to_lag3": f"counterfactual_replace_lag2_to_lag3_{cfg.target_month}.nc",
        "replace_lag1_to_lag3": f"counterfactual_replace_lag1_to_lag3_{cfg.target_month}.nc",
    }
    for key, filename in legacy_public_names.items():
        save_3d_nc(
            os.path.join(cfg.legacy_out_dir, filename),
            "prediction",
            legacy_pre[key],
            time_model,
            lat,
            lon,
        )

    process_public_names = {
        "VGC": f"counterfactual_replace_antecedent_gpp_{cfg.target_month}.nc",
        "LCE": f"counterfactual_replace_lagged_climate_{cfg.target_month}.nc",
        "CCE": f"counterfactual_replace_concurrent_climate_{cfg.target_month}.nc",
    }
    for key, filename in process_public_names.items():
        save_3d_nc(
            os.path.join(cfg.process_out_dir, filename),
            "prediction",
            process_pre[key],
            time_model,
            lat,
            lon,
        )

    for public_driver_name in lagged_driver_rules:
        save_3d_nc(
            os.path.join(
                cfg.lagged_driver_out_dir,
                f"counterfactual_replace_{public_driver_name}_{cfg.target_month}.nc",
            ),
            "prediction",
            lagged_driver_pre[public_driver_name],
            time_model,
            lat,
            lon,
        )

    for public_driver_name in concurrent_driver_rules:
        save_3d_nc(
            os.path.join(
                cfg.concurrent_driver_out_dir,
                f"counterfactual_replace_{public_driver_name}_{cfg.target_month}.nc",
            ),
            "prediction",
            concurrent_driver_pre[public_driver_name],
            time_model,
            lat,
            lon,
        )

    # Diagnostics
    model_folder = {
        "xgboost": "XGBoost",
        "rf": "RandomForest",
        "lightgbm": "LightGBM",
    }[cfg.model_name]
    diagnostics_dir = os.path.join(
        cfg.output_root, cfg.region_label, model_folder, "diagnostics"
    )
    os.makedirs(diagnostics_dir, exist_ok=True)

    model_eval = np.stack([model_mse, model_rmse, model_rss])
    save_3d_nc(
        os.path.join(
            diagnostics_dir,
            f"model_evaluation_{cfg.target_month}.nc",
        ),
        "model_evaluation",
        model_eval,
        np.arange(1, 4),
        lat,
        lon,
    )

    write_feature_names(
        [diagnostics_dir],
        cfg.target_month,
        names_without_y,
    )

    elapsed = time.perf_counter() - start_time
    print(
        f"all finish! processed={processed}, skipped={skipped}, "
        f"elapsed={elapsed / 3600:.2f} h",
        flush=True,
    )


if __name__ == "__main__":
    main()
