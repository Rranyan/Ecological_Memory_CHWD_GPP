%% CHWD identification using TCI, SMDI and a bivariate Gumbel copula
% =========================================================================
% Public version
%
% Purpose
% -------
% Calculate monthly compound heatwave-drought (CHWD) joint probabilities
% for 2000-2025 from monthly air temperature (TAS) and soil moisture (SM).
%
% Processing workflow
% -------------------
% 1) Linearly detrend TAS and SM over 2000-2025.
% 2) Calculate TCI and SMDI using 2000-2021 as the reference period.
% 3) Reverse the sign of SMDI so that larger values indicate drier
%    conditions.
% 4) Transform TCI and -SMDI to empirical marginal probabilities.
% 5) Fit a bivariate Gumbel copula at each grid cell.
% 6) Calculate the joint upper-tail probability:
%
%       P(U > u, V > v) = 1 - u - v + C(u,v)
%
% Smaller joint probabilities indicate rarer and more extreme compound
% hot-dry conditions.
%
% INPUT REQUIREMENTS
% ------------------
% Two monthly NetCDF files covering January 2000 to December 2025:
%
%   TAS : lon x lat x time
%   SM  : lon x lat x time
%
% Edit file paths and variable names in the USER SETTINGS section.
%
% OUTPUT
% ------
% joint_prob_hd_Gumbel_2000_2025.nc
%
% Main variable:
% joint_prob_hd_Gumbel
%
% Dimensions:
% lon x lat x time
%
% =========================================================================

clear;
clc;


%% =========================================================================
% 1. USER SETTINGS
% =========================================================================

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

reference_start_year = 2000;
reference_end_year   = 2021;

min_samples = 20;

block_lon = 100;
block_lat = 100;


%% =========================================================================
% 2. READ INPUT DATA
% =========================================================================

TAS = single(ncread(tas_file, tas_var));
SM  = single(ncread(sm_file,  sm_var));

lon  = single(ncread(tas_file, lon_var));
lat  = single(ncread(tas_file, lat_var));
time = single(ncread(tas_file, time_var));

[nlon, nlat, nt] = size(TAS);

expected_nt = (end_year - start_year + 1) * 12;

if nt ~= expected_nt
    error( ...
        'Expected %d monthly time steps for %d-%d, but found %d.', ...
        expected_nt, start_year, end_year, nt);
end

if ~isequal(size(SM), size(TAS))
    error('TAS and SM must have identical dimensions.');
end

n_reference_years = reference_end_year - reference_start_year + 1;
reference_end_idx = n_reference_years * 12;
reference_idx = 1:reference_end_idx;

fprintf( ...
    'Input grid: %d lon x %d lat x %d months\n', ...
    nlon, nlat, nt);


%% =========================================================================
% 3. LINEAR DETRENDING OF TAS AND SM
% =========================================================================

tas_d = nan(nlon, nlat, nt, 'single');
sm_d  = nan(nlon, nlat, nt, 'single');

fprintf('Detrending TAS and SM...\n');

for i = 1:nlon
    for j = 1:nlat

        tas_ts = double(squeeze(TAS(i,j,:)));
        sm_ts  = double(squeeze(SM(i,j,:)));

        if sum(isfinite(tas_ts)) >= min_samples
            valid = isfinite(tas_ts);

            if all(valid)
                tas_d(i,j,:) = single(detrend(tas_ts, 'linear'));
            else
                temp = nan(size(tas_ts));
                x = find(valid);
                p = polyfit(x, tas_ts(valid), 1);
                temp(valid) = tas_ts(valid) - polyval(p, x);
                tas_d(i,j,:) = single(temp);
            end
        end

        if sum(isfinite(sm_ts)) >= min_samples
            valid = isfinite(sm_ts);

            if all(valid)
                sm_d(i,j,:) = single(detrend(sm_ts, 'linear'));
            else
                temp = nan(size(sm_ts));
                x = find(valid);
                p = polyfit(x, sm_ts(valid), 1);
                temp(valid) = sm_ts(valid) - polyval(p, x);
                sm_d(i,j,:) = single(temp);
            end
        end

    end
end

clear TAS SM

fprintf('Detrending completed.\n');


%% =========================================================================
% 4. TEMPERATURE CONDITION INDEX (TCI)
% =========================================================================
%
% TCI = (T - Tmin) / (Tmax - Tmin)
%
% Tmin and Tmax are calculated from detrended TAS during 2000-2021.
%
% Larger TCI values indicate hotter conditions.
%
% Values outside the 2000-2021 reference range are clipped to [0,1].
% =========================================================================

TCI = nan(nlon, nlat, nt, 'single');

fprintf('Calculating TCI...\n');

for i_lon = 1:block_lon:nlon

    i_end_lon = min(i_lon + block_lon - 1, nlon);

    for i_lat = 1:block_lat:nlat

        i_end_lat = min(i_lat + block_lat - 1, nlat);

        T_block = single( ...
            tas_d(i_lon:i_end_lon, ...
                  i_lat:i_end_lat, :) );

        [nLonB, nLatB, ~] = size(T_block);

        ngrid = nLonB * nLatB;

        T_flat = reshape(T_block, ngrid, nt);

        T_reference = T_flat(:, reference_idx);

        Tmin = min(T_reference, [], 2, 'omitnan');
        Tmax = max(T_reference, [], 2, 'omitnan');

        range_val = Tmax - Tmin;
        range_val(range_val == 0) = NaN;

        TCI_flat = ...
            (T_flat - Tmin) ./ range_val;

        TCI_flat(TCI_flat < 0) = 0;
        TCI_flat(TCI_flat > 1) = 1;

        invalid_grid = ...
            isnan(Tmin) | ...
            isnan(Tmax) | ...
            isnan(range_val);

        TCI_flat(invalid_grid,:) = NaN;

        TCI( ...
            i_lon:i_end_lon, ...
            i_lat:i_end_lat, :) = ...
            reshape( ...
                single(TCI_flat), ...
                nLonB, nLatB, nt);

        clear T_block T_flat T_reference
        clear TCI_flat Tmin Tmax range_val invalid_grid

    end
end

clear tas_d

fprintf('TCI calculation completed.\n');


%% =========================================================================
% 5. SOIL-MOISTURE DEFICIT INDEX (SMDI)
% =========================================================================
%
% Reference statistics are calculated from detrended SM during 2000-2021.
%
% Negative SMDI values indicate dry conditions.
%
% for SM < median:
%
%     SD = (SM - median) / (median - minimum) * 100
%
% for SM > median:
%
%     SD = (SM - median) / (maximum - median) * 100
%
% Then:
%
%     SMDI(t) = 0.5 * SMDI(t-1) + SD(t)/50
%
% =========================================================================

SMDI = nan(nlon, nlat, nt, 'single');

fprintf('Calculating SMDI...\n');

for i_lon = 1:block_lon:nlon

    i_end_lon = min(i_lon + block_lon - 1, nlon);

    for i_lat = 1:block_lat:nlat

        i_end_lat = min(i_lat + block_lat - 1, nlat);

        SM_block = single( ...
            sm_d(i_lon:i_end_lon, ...
                 i_lat:i_end_lat, :) );

        [nLonB, nLatB, ~] = size(SM_block);

        ngrid = nLonB * nLatB;

        SM_flat = reshape(SM_block, ngrid, nt);

        SM_reference = SM_flat(:, reference_idx);

        median_SM = ...
            median(SM_reference, 2, 'omitnan');

        max_SM = ...
            max(SM_reference, [], 2, 'omitnan');

        min_SM = ...
            min(SM_reference, [], 2, 'omitnan');

        median_mat = ...
            repmat(median_SM, 1, nt);

        denom_low = ...
            repmat(median_SM - min_SM, 1, nt);

        denom_high = ...
            repmat(max_SM - median_SM, 1, nt);

        denom_low(denom_low == 0)   = NaN;
        denom_high(denom_high == 0) = NaN;

        SD = nan(ngrid, nt, 'single');

        mask_low = ...
            SM_flat < median_mat;

        SD(mask_low) = ...
            (SM_flat(mask_low) - ...
             median_mat(mask_low)) ./ ...
            denom_low(mask_low) * 100;

        mask_high = ...
            SM_flat > median_mat;

        SD(mask_high) = ...
            (SM_flat(mask_high) - ...
             median_mat(mask_high)) ./ ...
            denom_high(mask_high) * 100;

        mask_equal = ...
            SM_flat == median_mat;

        SD(mask_equal) = 0;

        SMDI_flat = ...
            nan(ngrid, nt, 'single');

        SMDI_flat(:,1) = ...
            SD(:,1) / 50;

        for t = 2:nt
            SMDI_flat(:,t) = ...
                0.5 * SMDI_flat(:,t-1) + ...
                SD(:,t) / 50;
        end

        SMDI( ...
            i_lon:i_end_lon, ...
            i_lat:i_end_lat, :) = ...
            reshape( ...
                SMDI_flat, ...
                nLonB, nLatB, nt);

        clear SM_block SM_flat SM_reference
        clear median_SM max_SM min_SM median_mat
        clear denom_low denom_high
        clear SD mask_low mask_high mask_equal SMDI_flat

    end
end

clear sm_d

fprintf('SMDI calculation completed.\n');


%% =========================================================================
% 6. BIVARIATE GUMBEL COPULA
% =========================================================================
%
% X = TCI
%     larger values = hotter conditions
%
% Y = -SMDI
%     larger values = drier conditions
%
% Empirical marginal probabilities are estimated using ranks.
%
% Joint upper-tail probability:
%
%     P(U > u, V > v)
%       = 1 - u - v + C(u,v)
%
% where C(u,v) is the fitted Gumbel copula CDF.
%
% Smaller probabilities indicate rarer compound hot-dry conditions.
% =========================================================================

joint_prob = ...
    nan(nlon, nlat, nt, 'single');

fit_success = ...
    false(nlon, nlat);

fprintf('Fitting Gumbel copula...\n');

for i_lon = 1:block_lon:nlon

    i_end_lon = ...
        min(i_lon + block_lon - 1, nlon);

    for i_lat = 1:block_lat:nlat

        i_end_lat = ...
            min(i_lat + block_lat - 1, nlat);

        TCI_block = ...
            TCI(i_lon:i_end_lon, ...
                i_lat:i_end_lat, :);

        SMDI_block = ...
            SMDI(i_lon:i_end_lon, ...
                 i_lat:i_end_lat, :);

        [nLonB, nLatB, ~] = ...
            size(TCI_block);

        ngrid = ...
            nLonB * nLatB;

        TCI_flat = ...
            reshape(TCI_block, ngrid, nt);

        SMDI_flat = ...
            reshape(SMDI_block, ngrid, nt);

        joint_flat = ...
            nan(ngrid, nt, 'single');

        success_flat = ...
            false(ngrid, 1);

        for k = 1:ngrid

            x0 = ...
                double(TCI_flat(k,:))';

            y0 = ...
                double(-SMDI_flat(k,:))';

            valid = ...
                isfinite(x0) & ...
                isfinite(y0);

            if sum(valid) < min_samples
                continue
            end

            x = x0(valid);
            y = y0(valid);

            if range(x) == 0 || range(y) == 0
                continue
            end

            u = ...
                (tiedrank(x) - 0.5) / length(x);

            v = ...
                (tiedrank(y) - 0.5) / length(y);

            u = ...
                min(max(u, 1e-6), 1 - 1e-6);

            v = ...
                min(max(v, 1e-6), 1 - 1e-6);

            try

                theta = ...
                    copulafit( ...
                        'Gumbel', ...
                        [u, v]);

                C_uv = ...
                    copulacdf( ...
                        'Gumbel', ...
                        [u, v], ...
                        theta);

                prob_hd = ...
                    1 - u - v + C_uv;

                prob_hd = ...
                    max(0, min(1, prob_hd));

            catch
                continue
            end

            joint_full = ...
                nan(nt, 1, 'single');

            joint_full(valid) = ...
                single(prob_hd);

            joint_flat(k,:) = ...
                joint_full;

            success_flat(k) = ...
                true;

        end

        joint_block = ...
            reshape( ...
                joint_flat, ...
                nLonB, nLatB, nt);

        joint_prob( ...
            i_lon:i_end_lon, ...
            i_lat:i_end_lat, :) = ...
            joint_block;

        fit_success( ...
            i_lon:i_end_lon, ...
            i_lat:i_end_lat) = ...
            reshape( ...
                success_flat, ...
                nLonB, nLatB);

        fprintf( ...
            'Finished lon %d-%d, lat %d-%d\n', ...
            i_lon, i_end_lon, ...
            i_lat, i_end_lat);

        clear TCI_block SMDI_block
        clear TCI_flat SMDI_flat
        clear joint_flat joint_block success_flat

    end
end

clear TCI SMDI


%% =========================================================================
% 7. SAVE OUTPUT
% =========================================================================

if exist(out_nc, 'file')
    delete(out_nc);
end

nccreate( ...
    out_nc, ...
    out_var, ...
    'Dimensions', ...
    {'lon', nlon, ...
     'lat', nlat, ...
     'time', nt}, ...
    'Datatype', 'single', ...
    'DeflateLevel', 5, ...
    'Format', 'netcdf4');

nccreate( ...
    out_nc, ...
    'lon', ...
    'Dimensions', {'lon', nlon}, ...
    'Datatype', 'single');

nccreate( ...
    out_nc, ...
    'lat', ...
    'Dimensions', {'lat', nlat}, ...
    'Datatype', 'single');

nccreate( ...
    out_nc, ...
    'time', ...
    'Dimensions', {'time', nt}, ...
    'Datatype', 'single');

ncwrite(out_nc, out_var, joint_prob);
ncwrite(out_nc, 'lon', lon);
ncwrite(out_nc, 'lat', lat);
ncwrite(out_nc, 'time', time);

ncwriteatt( ...
    out_nc, ...
    out_var, ...
    'long_name', ...
    ['Joint upper-tail probability of high temperature and drought ', ...
     'from a bivariate Gumbel copula']);

ncwriteatt( ...
    out_nc, ...
    out_var, ...
    'description', ...
    ['TAS and SM were linearly detrended over 2000-2025. ', ...
     'TCI and SMDI reference statistics were calculated using 2000-2021. ', ...
     'SMDI was sign-reversed so that larger values indicate stronger ', ...
     'drought. Lower joint probabilities indicate rarer compound ', ...
     'hot-dry conditions.']);

ncwriteatt( ...
    out_nc, ...
    out_var, ...
    'analysis_period', ...
    '2000-2025');

ncwriteatt( ...
    out_nc, ...
    out_var, ...
    'reference_period', ...
    '2000-2021');

ncwriteatt( ...
    out_nc, ...
    out_var, ...
    'detrending', ...
    'Linear detrending applied to TAS and SM over 2000-2025');

ncwriteatt( ...
    out_nc, ...
    out_var, ...
    'climatology_removal', ...
    'No calendar-month climatology removal applied');

save( ...
    'fit_success_Gumbel_2000_2025.mat', ...
    'fit_success', ...
    'lon', ...
    'lat', ...
    '-v7.3');

fprintf('\n==============================================\n');
fprintf('Gumbel copula calculation completed.\n');
fprintf('Output: %s\n', out_nc);
fprintf('==============================================\n');
