%% CHWD identification using TCI, SMDI and a bivariate Gumbel copula
% Figshare-ready version
%
% Purpose
% -------
% Calculate monthly compound heatwave-drought (CHWD) joint probabilities
% for 2000-2025 from monthly air temperature (TAS) and soil moisture (SM).
%
% The calculation follows the analysis used in the manuscript:
%   1) detrend TAS and SM over 2000-2025;
%   2) remove the calendar-month climatology based on 2000-2021;
%   3) calculate TCI and SMDI using 2000-2021 as the reference period;
%   4) define drought severity as -SMDI, so larger values indicate drier
%      conditions;
%   5) fit a bivariate Gumbel copula to empirical marginal probabilities;
%   6) calculate the joint upper-tail probability
%          P(U > u, V > v) = 1 - u - v + C(u,v).
%
% Smaller joint probabilities indicate rarer / more extreme compound
% hot-dry conditions.
%
% INPUT REQUIREMENTS
% ------------------
% Two monthly NetCDF files covering 2000.01-2025.12 (312 months):
%   - TAS: lon x lat x time
%   - SM : lon x lat x time
%
% Edit only the file paths and variable names in the "User settings" section.
%
% OUTPUT
% ------
% joint_prob_hd_Gumbel_2000_2025.nc
% variable: joint_prob_hd_Gumbel (lon x lat x time)
%
% Notes
% -----
% - The climatology/reference period is 2000-2021.
% - The later 2015-2025 extremity comparison is a separate post-processing
%   step and is not part of the copula fitting itself.
% - This script preserves the calculation logic of the final analysis while
%   removing plotting, diagnostic and temporary code.

clear; clc;

%% =========================
% User settings
% ==========================
tas_file = 'TAS_2000_2025.nc';
sm_file  = 'SM_2000_2025.nc';

tas_var = 'tas';
sm_var  = 'sm';

lon_var  = 'lon';
lat_var  = 'lat';
time_var = 'time';

out_nc  = 'joint_prob_hd_Gumbel_2000_2025.nc';
out_var = 'joint_prob_hd_Gumbel';

start_year = 2000;
end_year   = 2025;

clim_start_year = 2000;
clim_end_year   = 2021;

min_samples = 20;
block_lon = 100;
block_lat = 100;

%% =========================
% Read input data
% ==========================
TAS = single(ncread(tas_file, tas_var));
SM  = single(ncread(sm_file,  sm_var));

lon  = single(ncread(tas_file, lon_var));
lat  = single(ncread(tas_file, lat_var));
time = single(ncread(tas_file, time_var));

[nlon, nlat, nt] = size(TAS);

expected_nt = (end_year - start_year + 1) * 12;
if nt ~= expected_nt
    error('Expected %d monthly time steps for 2000-2025, but found %d.', ...
        expected_nt, nt);
end

if ~isequal(size(SM), size(TAS))
    error('TAS and SM must have identical dimensions.');
end

nmonths = 12;
nyears_clim = clim_end_year - clim_start_year + 1;
clim_end_idx = nyears_clim * nmonths;   % 264 for 2000-2021
stat_idx = 1:clim_end_idx;

fprintf('Input grid: %d lon x %d lat x %d months\n', nlon, nlat, nt);

%% =========================
% 1. Linear detrending
% ==========================
tas_d = nan(nlon, nlat, nt, 'single');
sm_d  = nan(nlon, nlat, nt, 'single');

for i = 1:nlon
    for j = 1:nlat

        tas_ts = double(squeeze(TAS(i,j,:)));
        sm_ts  = double(squeeze(SM(i,j,:)));

        if sum(isfinite(tas_ts)) >= min_samples
            tas_d(i,j,:) = single(detrend(tas_ts, 'linear', 'omitnan'));
        end

        if sum(isfinite(sm_ts)) >= min_samples
            sm_d(i,j,:) = single(detrend(sm_ts, 'linear', 'omitnan'));
        end
    end
end

clear TAS SM

%% =========================
% 2. Remove calendar-month climatology (2000-2021)
% ==========================
tas_anom = nan(nlon, nlat, nt, 'single');
sm_anom  = nan(nlon, nlat, nt, 'single');

for m = 1:nmonths

    idx_all  = m:nmonths:nt;
    idx_clim = m:nmonths:clim_end_idx;

    tas_clim_m = mean(tas_d(:,:,idx_clim), 3, 'omitnan');
    sm_clim_m  = mean(sm_d(:,:,idx_clim),  3, 'omitnan');

    tas_anom(:,:,idx_all) = tas_d(:,:,idx_all) - tas_clim_m;
    sm_anom(:,:,idx_all)  = sm_d(:,:,idx_all)  - sm_clim_m;
end

clear tas_d sm_d

%% =========================
% 3. Temperature condition index (TCI)
% ==========================
% TCI = (T - Tmin) / (Tmax - Tmin)
% Tmin/Tmax are calculated from 2000-2021 only.
% Larger TCI = hotter conditions.
%
% Values outside the historical 2000-2021 range are clipped to [0,1],
% following the final analysis.

TCI = nan(nlon, nlat, nt, 'single');

for i_lon = 1:block_lon:nlon

    i_end_lon = min(i_lon + block_lon - 1, nlon);

    for i_lat = 1:block_lat:nlat

        i_end_lat = min(i_lat + block_lat - 1, nlat);

        T_block = single(tas_anom(i_lon:i_end_lon, i_lat:i_end_lat, :));

        [nLonB, nLatB, ~] = size(T_block);
        ngrid = nLonB * nLatB;

        T_flat = reshape(T_block, ngrid, nt);
        T_stat = T_flat(:, stat_idx);

        Tmin = min(T_stat, [], 2, 'omitnan');
        Tmax = max(T_stat, [], 2, 'omitnan');

        range_val = Tmax - Tmin;
        range_val(range_val == 0) = NaN;

        TCI_flat = (T_flat - Tmin) ./ range_val;

        TCI_flat(TCI_flat < 0) = 0;
        TCI_flat(TCI_flat > 1) = 1;

        invalid_grid = isnan(Tmin) | isnan(Tmax) | isnan(range_val);
        TCI_flat(invalid_grid, :) = NaN;

        TCI(i_lon:i_end_lon, i_lat:i_end_lat, :) = ...
            reshape(single(TCI_flat), nLonB, nLatB, nt);

        clear T_block T_flat T_stat TCI_flat Tmin Tmax range_val invalid_grid
    end
end

clear tas_anom

%% =========================
% 4. Soil-moisture deficit index (SMDI)
% ==========================
% Reference statistics (median, minimum, maximum) use 2000-2021 only.
% Negative SMDI values indicate dry conditions.

SMDI = nan(nlon, nlat, nt, 'single');

for i_lon = 1:block_lon:nlon

    i_end_lon = min(i_lon + block_lon - 1, nlon);

    for i_lat = 1:block_lat:nlat

        i_end_lat = min(i_lat + block_lat - 1, nlat);

        SM_block = single(sm_anom(i_lon:i_end_lon, i_lat:i_end_lat, :));

        [nLonB, nLatB, ~] = size(SM_block);
        ngrid = nLonB * nLatB;

        SM_flat = reshape(SM_block, ngrid, nt);
        SM_stat = SM_flat(:, stat_idx);

        MSW   = median(SM_stat, 2, 'omitnan');
        maxSW = max(SM_stat, [], 2, 'omitnan');
        minSW = min(SM_stat, [], 2, 'omitnan');

        MSW_mat = repmat(MSW, 1, nt);

        denom_low  = repmat(MSW - minSW, 1, nt);
        denom_high = repmat(maxSW - MSW, 1, nt);

        denom_low(denom_low == 0) = NaN;
        denom_high(denom_high == 0) = NaN;

        SD = nan(ngrid, nt, 'single');

        mask_low = SM_flat < MSW_mat;
        SD(mask_low) = ...
            (SM_flat(mask_low) - MSW_mat(mask_low)) ./ ...
            denom_low(mask_low) * 100;

        mask_high = SM_flat > MSW_mat;
        SD(mask_high) = ...
            (SM_flat(mask_high) - MSW_mat(mask_high)) ./ ...
            denom_high(mask_high) * 100;

        mask_equal = SM_flat == MSW_mat;
        SD(mask_equal) = 0;

        SMDI_flat = nan(ngrid, nt, 'single');
        SMDI_flat(:,1) = SD(:,1) / 50;

        for t = 2:nt
            SMDI_flat(:,t) = 0.5 * SMDI_flat(:,t-1) + SD(:,t) / 50;
        end

        SMDI(i_lon:i_end_lon, i_lat:i_end_lat, :) = ...
            reshape(SMDI_flat, nLonB, nLatB, nt);

        clear SM_block SM_flat SM_stat MSW maxSW minSW MSW_mat
        clear denom_low denom_high SD mask_low mask_high mask_equal SMDI_flat
    end
end

clear sm_anom

%% =========================
% 5. Bivariate Gumbel copula
% ==========================
% X = TCI       : larger = hotter
% Y = -SMDI     : larger = drier
%
% Reversing the sign of SMDI does NOT change the drought definition.
% It only makes heat and drought severity point in the same direction.
%
% Joint upper-tail probability:
%   P(U > u, V > v) = 1 - u - v + C(u,v)

joint_prob = nan(nlon, nlat, nt, 'single');
fit_success = false(nlon, nlat);

for i_lon = 1:block_lon:nlon

    i_end_lon = min(i_lon + block_lon - 1, nlon);

    for i_lat = 1:block_lat:nlat

        i_end_lat = min(i_lat + block_lat - 1, nlat);

        TCI_block  = TCI(i_lon:i_end_lon, i_lat:i_end_lat, :);
        SMDI_block = SMDI(i_lon:i_end_lon, i_lat:i_end_lat, :);

        [nLonB, nLatB, ~] = size(TCI_block);
        ngrid = nLonB * nLatB;

        TCI_flat  = reshape(TCI_block,  ngrid, nt);
        SMDI_flat = reshape(SMDI_block, ngrid, nt);

        joint_flat = nan(ngrid, nt, 'single');
        success_flat = false(ngrid, 1);

        for k = 1:ngrid

            x0 = double(TCI_flat(k, :))';       % larger = hotter
            y0 = double(-SMDI_flat(k, :))';     % larger = drier

            valid = isfinite(x0) & isfinite(y0);

            if sum(valid) < min_samples
                continue
            end

            x = x0(valid);
            y = y0(valid);

            if range(x) == 0 || range(y) == 0
                continue
            end

            % Empirical marginal probabilities
            u = (tiedrank(x) - 0.5) / length(x);
            v = (tiedrank(y) - 0.5) / length(y);

            % Avoid exact 0/1 boundaries
            u = min(max(u, 1e-6), 1 - 1e-6);
            v = min(max(v, 1e-6), 1 - 1e-6);

            try
                theta = copulafit('Gumbel', [u, v]);
                C_uv  = copulacdf('Gumbel', [u, v], theta);

                % Joint upper-tail probability
                prob_hd = 1 - u - v + C_uv;
                prob_hd = max(0, min(1, prob_hd));

            catch
                continue
            end

            joint_full = nan(nt, 1, 'single');
            joint_full(valid) = single(prob_hd);

            joint_flat(k, :) = joint_full;
            success_flat(k) = true;
        end

        joint_block = reshape(joint_flat, nLonB, nLatB, nt);

        joint_prob(i_lon:i_end_lon, i_lat:i_end_lat, :) = joint_block;
        fit_success(i_lon:i_end_lon, i_lat:i_end_lat) = ...
            reshape(success_flat, nLonB, nLatB);

        fprintf('Finished lon %d-%d, lat %d-%d\n', ...
            i_lon, i_end_lon, i_lat, i_end_lat);

        clear TCI_block SMDI_block TCI_flat SMDI_flat
        clear joint_flat joint_block success_flat
    end
end

%% =========================
% 6. Save output
% ==========================
if exist(out_nc, 'file')
    delete(out_nc);
end

nccreate(out_nc, out_var, ...
    'Dimensions', {'lon', nlon, 'lat', nlat, 'time', nt}, ...
    'Datatype', 'single', ...
    'DeflateLevel', 5, ...
    'Format', 'netcdf4');

nccreate(out_nc, 'lon', ...
    'Dimensions', {'lon', nlon}, 'Datatype', 'single');

nccreate(out_nc, 'lat', ...
    'Dimensions', {'lat', nlat}, 'Datatype', 'single');

nccreate(out_nc, 'time', ...
    'Dimensions', {'time', nt}, 'Datatype', 'single');

ncwrite(out_nc, out_var, joint_prob);
ncwrite(out_nc, 'lon', lon);
ncwrite(out_nc, 'lat', lat);
ncwrite(out_nc, 'time', time);

ncwriteatt(out_nc, out_var, 'long_name', ...
    'Joint upper-tail probability of high temperature and drought from a bivariate Gumbel copula');

ncwriteatt(out_nc, out_var, 'description', ...
    'TCI is positively oriented for heat and SMDI is sign-reversed so that larger values indicate stronger drought; lower joint probability indicates rarer compound hot-dry conditions.');

ncwriteatt(out_nc, out_var, 'analysis_period', '2000-2025');
ncwriteatt(out_nc, out_var, 'reference_period', '2000-2021');

save('fit_success_Gumbel_2000_2025.mat', ...
    'fit_success', 'lon', 'lat', '-v7.3');

fprintf('Gumbel Copula calculation completed.\n');
fprintf('Output: %s\n', out_nc);
