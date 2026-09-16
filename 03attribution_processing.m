%% attribution_postprocessing_public.m
% =========================================================================
% Public version: post-processing of counterfactual attribution experiments
%
% This script converts model predictions from counterfactual experiments into
% regional attribution results for:
%   1) Legacy effects associated with the previous 1, 2, and 3 months
%   2) Concurrent climate effects (CCE), lagged climate effects (LCE),
%      and vegetation growth carryover (VGC)
%   3) Contributions from individual climate drivers:
%      precipitation, soil moisture, solar radiation, air temperature,
%      and vapor pressure deficit
%
% General attribution rule:
%   contribution = reference prediction - counterfactual prediction
%
% IMPORTANT
% ---------
% All paths and file names below are generic examples for public sharing.
% Rename the Figshare files to match these descriptive names, or modify the
% file-name templates in Section 3.
%
% Expected NetCDF variables:
%   observed GPP files       : "gpp"
%   model prediction files   : "prediction"
%   coordinates              : "lon", "lat"
%
% Expected time dimension:
%   1:22 = 2000-2021
%   23   = 2022
%
% Example public directory structure:
%
% project_data/
% ├── observed_gpp/
% │   └── observed_gpp_<month>.nc
% │
% └── model_predictions/
%     └── <region>/
%         └── <model>/
%             ├── reference/
%             │   └── reference_prediction_<month>.nc
%             │
%             ├── legacy_by_lag/
%             │   ├── counterfactual_replace_lag3_<month>.nc
%             │   ├── counterfactual_replace_lag2_to_lag3_<month>.nc
%             │   └── counterfactual_replace_lag1_to_lag3_<month>.nc
%             │
%             ├── process_partition/
%             │   ├── counterfactual_replace_antecedent_gpp_<month>.nc
%             │   ├── counterfactual_replace_lagged_climate_<month>.nc
%             │   └── counterfactual_replace_concurrent_climate_<month>.nc
%             │
%             └── climate_drivers/
%                 ├── lagged/
%                 │   └── counterfactual_replace_<driver>_<month>.nc
%                 └── concurrent/
%                     └── counterfactual_replace_<driver>_<month>.nc
%
% =========================================================================

clear; clc;

%% ========================================================================
% 1. USER SETTINGS
% ========================================================================

cfg.data_root = fullfile('path', 'to', 'project_data');

% Region:
%   'China', 'Europe', or 'US'
cfg.region = 'US';

% Machine-learning model:
%   'RandomForest', 'XGBoost', or 'LightGBM'
cfg.model = 'XGBoost';

% Select attribution modules
cfg.run_legacy_by_lag = true;
cfg.run_process_split = true;
cfg.run_driver_split  = true;

% Save summary tables
cfg.save_results = true;
cfg.result_dir = fullfile(pwd, 'regional_attribution_results');

if cfg.save_results && ~isfolder(cfg.result_dir)
    mkdir(cfg.result_dir);
end

%% ========================================================================
% 2. ANALYSIS MONTHS
% ========================================================================
switch cfg.region
    case 'China'
        analysis_months = {'Aug','Sep','Oct'};
    case 'Europe'
        analysis_months = {'Jul','Aug','Sep'};
    case 'US'
        analysis_months = {'Jul','Aug','Sep'};
    otherwise
        error('Unknown region: %s', cfg.region);
end

n_months = numel(analysis_months);


%% ========================================================================
% 3. PUBLIC DIRECTORY AND FILE-NAME TEMPLATES
% ========================================================================

observed_dir = fullfile(cfg.data_root, 'observed_gpp');

prediction_root = fullfile( ...
    cfg.data_root, 'model_predictions', cfg.region, cfg.model);

reference_dir     = fullfile(prediction_root, 'reference');
legacy_lag_dir    = fullfile(prediction_root, 'legacy_by_lag');
process_split_dir = fullfile(prediction_root, 'process_partition');

driver_lagged_dir = fullfile( ...
    prediction_root, 'climate_drivers', 'lagged');

driver_current_dir = fullfile( ...
    prediction_root, 'climate_drivers', 'concurrent');

fprintf('\n============================================================\n');
fprintf('Region : %s\n', cfg.region);
fprintf('Model  : %s\n', cfg.model);
fprintf('Months : %s\n', strjoin(analysis_months, ', '));
fprintf('============================================================\n\n');

%% ========================================================================
% 4. LEGACY EFFECTS BY LAG MONTH
% ========================================================================

if cfg.run_legacy_by_lag

    legacy_by_lag = nan(6, n_months);

    for m = 1:n_months

        month_name = analysis_months{m};

        observed_file = fullfile( ...
            observed_dir, sprintf('observed_gpp_%s.nc', month_name));

        reference_file = fullfile( ...
            reference_dir, sprintf('reference_prediction_%s.nc', month_name));

        replace_lag3_file = fullfile( ...
            legacy_lag_dir, ...
            sprintf('counterfactual_replace_lag3_%s.nc', month_name));

        replace_lag2_to_lag3_file = fullfile( ...
            legacy_lag_dir, ...
            sprintf('counterfactual_replace_lag2_to_lag3_%s.nc', month_name));

        replace_lag1_to_lag3_file = fullfile( ...
            legacy_lag_dir, ...
            sprintf('counterfactual_replace_lag1_to_lag3_%s.nc', month_name));

        check_file_exist(observed_file);
        check_file_exist(reference_file);
        check_file_exist(replace_lag3_file);
        check_file_exist(replace_lag2_to_lag3_file);
        check_file_exist(replace_lag1_to_lag3_file);

        [region_mask, latitude_weight] = ...
            build_region_mask_and_weight(reference_file, cfg.region);

        observed_mean = regional_anomaly_mean( ...
            ncread(observed_file, 'gpp'), latitude_weight, region_mask);

        reference_mean = regional_anomaly_mean( ...
            ncread(reference_file, 'prediction'), latitude_weight, region_mask);

        replace_lag3_mean = regional_anomaly_mean( ...
            ncread(replace_lag3_file, 'prediction'), ...
            latitude_weight, region_mask);

        replace_lag2_to_lag3_mean = regional_anomaly_mean( ...
            ncread(replace_lag2_to_lag3_file, 'prediction'), ...
            latitude_weight, region_mask);

        replace_lag1_to_lag3_mean = regional_anomaly_mean( ...
            ncread(replace_lag1_to_lag3_file, 'prediction'), ...
            latitude_weight, region_mask);

        legacy_t_minus_3 = reference_mean - replace_lag3_mean;
        legacy_t_minus_2 = replace_lag3_mean - replace_lag2_to_lag3_mean;
        legacy_t_minus_1 = replace_lag2_to_lag3_mean - replace_lag1_to_lag3_mean;

        legacy_by_lag(:,m) = [
            observed_mean
            reference_mean
            legacy_t_minus_3
            legacy_t_minus_2
            legacy_t_minus_1
            replace_lag1_to_lag3_mean
        ];

        fprintf('Legacy-by-lag finished: %s %s %s\n', ...
            cfg.region, cfg.model, month_name);
    end

    row_names = {
        'Observed_GPP_anomaly'
        'Reference_prediction'
        'Legacy_t_minus_3'
        'Legacy_t_minus_2'
        'Legacy_t_minus_1'
        'Prediction_after_replacing_all_lagged_predictors'
    };

    T_legacy = matrix_to_table( ...
        legacy_by_lag, row_names, analysis_months);

    fprintf('\n--- Legacy effects by lag month ---\n');
    disp(T_legacy);

    if cfg.save_results
        output_file = fullfile( ...
            cfg.result_dir, ...
            sprintf('%s_%s_legacy_effects_by_lag.csv', ...
            cfg.region, cfg.model));
        writetable(T_legacy, output_file, 'WriteRowNames', true);
    end
end

%% ========================================================================
% 5. CCE / LCE / VGC PROCESS PARTITION
% ========================================================================

if cfg.run_process_split

    process_contributions = nan(5, n_months);

    for m = 1:n_months

        month_name = analysis_months{m};

        observed_file = fullfile( ...
            observed_dir, sprintf('observed_gpp_%s.nc', month_name));

        reference_file = fullfile( ...
            reference_dir, sprintf('reference_prediction_%s.nc', month_name));

        replace_antecedent_gpp_file = fullfile( ...
            process_split_dir, ...
            sprintf('counterfactual_replace_antecedent_gpp_%s.nc', month_name));

        replace_lagged_climate_file = fullfile( ...
            process_split_dir, ...
            sprintf('counterfactual_replace_lagged_climate_%s.nc', month_name));

        replace_concurrent_climate_file = fullfile( ...
            process_split_dir, ...
            sprintf('counterfactual_replace_concurrent_climate_%s.nc', month_name));

        check_file_exist(observed_file);
        check_file_exist(reference_file);
        check_file_exist(replace_antecedent_gpp_file);
        check_file_exist(replace_lagged_climate_file);
        check_file_exist(replace_concurrent_climate_file);

        [region_mask, latitude_weight] = ...
            build_region_mask_and_weight(reference_file, cfg.region);

        observed_mean = regional_anomaly_mean( ...
            ncread(observed_file, 'gpp'), latitude_weight, region_mask);

        reference_mean = regional_anomaly_mean( ...
            ncread(reference_file, 'prediction'), latitude_weight, region_mask);

        antecedent_gpp_counterfactual = regional_anomaly_mean( ...
            ncread(replace_antecedent_gpp_file, 'prediction'), ...
            latitude_weight, region_mask);

        lagged_climate_counterfactual = regional_anomaly_mean( ...
            ncread(replace_lagged_climate_file, 'prediction'), ...
            latitude_weight, region_mask);

        concurrent_climate_counterfactual = regional_anomaly_mean( ...
            ncread(replace_concurrent_climate_file, 'prediction'), ...
            latitude_weight, region_mask);

        VGC_contribution = reference_mean - antecedent_gpp_counterfactual;
        LCE_contribution = reference_mean - lagged_climate_counterfactual;
        CCE_contribution = reference_mean - concurrent_climate_counterfactual;

        process_contributions(:,m) = [
            observed_mean
            reference_mean
            VGC_contribution
            LCE_contribution
            CCE_contribution
        ];

        fprintf('Process partition finished: %s %s %s\n', ...
            cfg.region, cfg.model, month_name);
    end

    row_names = {
        'Observed_GPP_anomaly'
        'Reference_prediction'
        'Vegetation_growth_carryover_VGC'
        'Lagged_climate_effect_LCE'
        'Concurrent_climate_effect_CCE'
    };

    T_process = matrix_to_table( ...
        process_contributions, row_names, analysis_months);

    fprintf('\n--- CCE / LCE / VGC contributions ---\n');
    disp(T_process);

    if cfg.save_results
        output_file = fullfile( ...
            cfg.result_dir, ...
            sprintf('%s_%s_process_contributions.csv', ...
            cfg.region, cfg.model));
        writetable(T_process, output_file, 'WriteRowNames', true);
    end
end

%% ========================================================================
% 6. INDIVIDUAL CLIMATE-DRIVER CONTRIBUTIONS
% ========================================================================

if cfg.run_driver_split

    driver_file_names = {
        'precipitation'
        'soil_moisture'
        'solar_radiation'
        'air_temperature'
        'vapor_pressure_deficit'
    };

    driver_labels = {
        'Precipitation_PRE'
        'Soil_moisture_SM'
        'Surface_solar_radiation_SSRD'
        'Air_temperature_TAS'
        'Vapor_pressure_deficit_VPD'
    };

    lagged_driver_contributions     = nan(7, n_months);
    concurrent_driver_contributions = nan(7, n_months);

    for m = 1:n_months

        month_name = analysis_months{m};

        observed_file = fullfile( ...
            observed_dir, sprintf('observed_gpp_%s.nc', month_name));

        reference_file = fullfile( ...
            reference_dir, sprintf('reference_prediction_%s.nc', month_name));

        check_file_exist(observed_file);
        check_file_exist(reference_file);

        [region_mask, latitude_weight] = ...
            build_region_mask_and_weight(reference_file, cfg.region);

        observed_mean = regional_anomaly_mean( ...
            ncread(observed_file, 'gpp'), latitude_weight, region_mask);

        reference_mean = regional_anomaly_mean( ...
            ncread(reference_file, 'prediction'), latitude_weight, region_mask);

        lagged_driver_values     = nan(5,1);
        concurrent_driver_values = nan(5,1);

        for d = 1:numel(driver_file_names)

            driver_name = driver_file_names{d};

            lagged_file = fullfile( ...
                driver_lagged_dir, ...
                sprintf('counterfactual_replace_%s_%s.nc', ...
                driver_name, month_name));

            concurrent_file = fullfile( ...
                driver_current_dir, ...
                sprintf('counterfactual_replace_%s_%s.nc', ...
                driver_name, month_name));

            check_file_exist(lagged_file);
            check_file_exist(concurrent_file);

            lagged_counterfactual_mean = regional_anomaly_mean( ...
                ncread(lagged_file, 'prediction'), ...
                latitude_weight, region_mask);

            concurrent_counterfactual_mean = regional_anomaly_mean( ...
                ncread(concurrent_file, 'prediction'), ...
                latitude_weight, region_mask);

            lagged_driver_values(d) = ...
                reference_mean - lagged_counterfactual_mean;

            concurrent_driver_values(d) = ...
                reference_mean - concurrent_counterfactual_mean;
        end

        lagged_driver_contributions(:,m) = [
            observed_mean
            reference_mean
            lagged_driver_values
        ];

        concurrent_driver_contributions(:,m) = [
            observed_mean
            reference_mean
            concurrent_driver_values
        ];

        fprintf('Climate-driver partition finished: %s %s %s\n', ...
            cfg.region, cfg.model, month_name);
    end

    row_names = [
        {'Observed_GPP_anomaly'; 'Reference_prediction'}
        driver_labels
    ];

    T_lagged_drivers = matrix_to_table( ...
        lagged_driver_contributions, row_names, analysis_months);

    T_concurrent_drivers = matrix_to_table( ...
        concurrent_driver_contributions, row_names, analysis_months);

    fprintf('\n--- Lagged climate-driver contributions ---\n');
    disp(T_lagged_drivers);

    fprintf('\n--- Concurrent climate-driver contributions ---\n');
    disp(T_concurrent_drivers);

    if cfg.save_results

        lagged_output_file = fullfile( ...
            cfg.result_dir, ...
            sprintf('%s_%s_lagged_climate_driver_contributions.csv', ...
            cfg.region, cfg.model));

        concurrent_output_file = fullfile( ...
            cfg.result_dir, ...
            sprintf('%s_%s_concurrent_climate_driver_contributions.csv', ...
            cfg.region, cfg.model));

        writetable(T_lagged_drivers, ...
            lagged_output_file, 'WriteRowNames', true);

        writetable(T_concurrent_drivers, ...
            concurrent_output_file, 'WriteRowNames', true);
    end
end

fprintf('\n============================================================\n');
fprintf('All requested attribution post-processing is complete.\n');
fprintf('============================================================\n');

%% ========================================================================
% LOCAL FUNCTIONS
% ========================================================================

function [region_mask, latitude_weight] = ...
    build_region_mask_and_weight(nc_file, region_name)

    latitude  = ncread(nc_file, 'lat');
    longitude = ncread(nc_file, 'lon');

    longitude_spacing = abs(median(diff(longitude)));
    latitude_spacing  = abs(median(diff(latitude)));

    longitude_center = longitude + longitude_spacing / 2;
    latitude_center  = latitude  - latitude_spacing  / 2;

    [longitude_grid, latitude_grid] = ...
        ndgrid(longitude_center, latitude_center);

    [polygon_lon, polygon_lat] = get_region_polygon(region_name);

    region_mask = inpolygon( ...
        longitude_grid, latitude_grid, polygon_lon, polygon_lat);

    [~, latitude_for_weight] = ...
        ndgrid(longitude_center, latitude_center);

    latitude_weight = abs(cosd(latitude_for_weight));
end


function regional_mean = ...
    regional_anomaly_mean(data, latitude_weight, region_mask)

    climatology_2000_2021 = nanmean(data(:,:,1:22), 3);
    anomaly_2022 = data(:,:,23) - climatology_2000_2021;

    weighted_anomaly = anomaly_2022 .* latitude_weight;
    weighted_anomaly(~region_mask) = NaN;

    regional_mean = nanmean(weighted_anomaly(:));
end


function [polygon_lon, polygon_lat] = get_region_polygon(region_name)

    switch region_name

        case 'US'
            polygon_lon = [-104, -114, -103, -93, -104];
            polygon_lat = [  30,   49,   49,  30,   30];

        case 'China'
            polygon_lon = [102, 102, 123, 123, 102];
            polygon_lat = [ 25,  33,  33,  25,  25];

        case 'Europe'
            polygon_lon = [-5, -5, 24, 24, -5];
            polygon_lat = [40, 55, 55, 40, 40];

        otherwise
            error('Unknown region: %s', region_name);
    end
end


function output_table = matrix_to_table(data, row_names, month_names)

    output_table = array2table( ...
        data, ...
        'VariableNames', matlab.lang.makeValidName(month_names), ...
        'RowNames', row_names);
end


function check_file_exist(file_path)

    if ~isfile(file_path)
        error('Required input file was not found: %s', file_path);
    end
end
