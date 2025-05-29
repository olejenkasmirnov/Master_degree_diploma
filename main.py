import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, griddata

COORD_TPP3     = (55.2272050, 61.4901439)
DIST_STEP      = 10_000
MAX_DIST       = 100_000
TARGET_PRESS   = 1000
REGION_RADIUS  = 500
NO2_FILES      = ['data_plev.nc', 'data_sfc.nc', 'S5P_NRTI_L2__NO2____20230101T002405_20230101T002905_27036_03_020400_20230101T013350.nc', 
                'S5P_NRTI_L2__NO2____20240101T080326_20240101T080826_32219_03_020600_20240101T090808.nc']
WIND_FILES     = ['era5_surface_TEC3_2022_02.nc', 'era5_surface_TEC3_2022_03.nc','era5_surface_TEC3_2022_04.nc', 'era5_surface_TEC3_2024_05.nc', 
                'era5_surface_TEC3_2022_06.nc', 'era5_surface_TEC3_2022_07.nc', 'era5_surface_TEC3_2022_08.nc', 'era5_surface_TEC3_2022_09.nc', 
                'era5_surface_TEC3_2022_10.nc', 'era5_surface_TEC3_2022_11.nc', 'era5_surface_TEC3_2022_12.nc', 'era5_surface_TEC3_2023_01.nc', 
                'era5_surface_TEC3_2023_02.nc', 'era5_surface_TEC3_2023_03.nc', 'era5_surface_TEC3_2023_04.nc', 'era5_surface_TEC3_2023_05.nc', 
                'era5_surface_TEC3_2023_10.nc', 'era5_surface_TEC3_2023_11.nc', 'era5_surface_TEC3_2023_12.nc', 'era5_surface_TEC3_2024_01.nc', 
                'era5_surface_TEC3_2024_02.nc', 'era5_surface_TEC3_2024_03.nc', 'era5_surface_TEC3_2024_04.nc', 'era5_surface_TEC3_2024_05.nc', 
                'era5_surface_TEC3_2024_06.nc', 'era5_surface_TEC3_2024_07.nc', 'era5_surface_TEC3_2024_08.nc', 'era5_surface_TEC3_2024_09.nc', 
                'era5_surface_TEC3_2024_10.nc', 'era5_surface_TEC3_2024_11.nc', 'era5_surface_TEC3_2024_12.nc']

# -- Data loading functions
def load_no2_data(files):
    ds = xr.open_mfdataset(files, combine='by_coords',
                           decode_times=True, decode_timedelta=True).load()
    if 'pressure_level' in ds.dims:
        ds = ds.sel(pressure_level=TARGET_PRESS, method='nearest').drop_vars('pressure_level')
    if {'forecast_reference_time','forecast_period'}.issubset(ds.dims):
        ds = ds.stack(z=('forecast_reference_time','forecast_period'))
        vt = ds['valid_time'].values.ravel()
        ds = ds.assign_coords(time=('z', vt)).swap_dims({'z':'time'})
        ds = ds.drop_vars(['forecast_reference_time','forecast_period','valid_time'])
    return ds.set_coords(['latitude','longitude','time'])

def load_wind_data(files):
    ds = xr.open_mfdataset(files, combine='by_coords',
                           decode_times=True, decode_timedelta=True).load()
    if 'longitude' in ds and ds.longitude.max()>180:
        ds = ds.assign_coords(longitude=(((ds.longitude+180)%360)-180)).sortby('longitude')
    if 'time' not in ds.dims and 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time':'time'})
    return ds.set_coords(['latitude','longitude','time'])

def load_co2_data(files):
    ds = xr.open_mfdataset(files, combine='by_coords',
                           decode_times=True, decode_timedelta=True).load()
    if 'pressure_level' in ds.dims:
        ds = ds.sel(pressure_level=TARGET_PRESS, method='nearest').drop_vars('pressure_level')
    if {'forecast_reference_time','forecast_period'}.issubset(ds.dims):
        ds = ds.stack(z=('forecast_reference_time','forecast_period'))
        vt = ds['valid_time'].values.ravel()
        ds = ds.assign_coords(time=('z', vt)).swap_dims({'z':'time'})
        ds = ds.drop_vars(['forecast_reference_time','forecast_period','valid_time'])
    return ds.set_coords(['latitude','longitude','time'])

# -- Region subsetting
def subset_region(ds, lat0, lon0, radius_km):
    ddeg = radius_km / 111.0
    mask = ((ds.latitude>=lat0-ddeg)&(ds.latitude<=lat0+ddeg)&
            (ds.longitude>=lon0-ddeg)&(ds.longitude<=lon0+ddeg))
    return ds.where(mask, drop=True)

def no2_to_nox(no2_conc, nox_as_no2=False):
    molar_mass_no = 30.01
    molar_mass_no2 = 46.0055
    
    if nox_as_no2:
        return no2_conc
    else:
        return no2_conc * (molar_mass_no / molar_mass_no2)

def nox_to_no2(nox_conc, nox_as_no2=False):
    molar_mass_no = 30.01
    molar_mass_no2 = 46.0055
    
    if nox_as_no2:
        return nox_conc
    else:
        return nox_conc * (molar_mass_no2 / molar_mass_no)
    
# -- Synthetic NOx generation

def generate_nox_data(co2_ds, base_ratio=0.0004, noise_level=0.1):
    co2 = co2_ds['co2']
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, noise_level * base_ratio, size=co2.shape)
    nox = co2 * base_ratio + noise
    return xr.Dataset({'nox': (co2.dims, nox.values)}, coords=co2.coords)

# -- Transect sampling

def sample_transect(slice_da, u0, v0, distances):
    lat0, lon0 = COORD_TPP3
    theta = np.arctan2(v0, u0)
    lats = lat0 + (distances * np.sin(theta)) / 111000.
    lons = lon0 + (distances * np.cos(theta)) / (111000. * np.cos(np.deg2rad(lat0)))
    pts = np.vstack([slice_da.latitude.values.ravel(), slice_da.longitude.values.ravel()]).T
    vals = slice_da.values.ravel()
    mask = ~np.isnan(vals)
    pts, vals = pts[mask], vals[mask]
    if pts.shape[0] >= 4:
        conc = LinearNDInterpolator(pts, vals)(np.vstack([lats, lons]).T)
    else:
        conc = griddata(pts, vals, (lats, lons), method='nearest')
    return conc

def build_transect_matrix(ds_main, wind_ds, distances, var_name):
    times = ds_main['time'].values
    mat = np.full((len(times), len(distances)), np.nan)
    for i, t in enumerate(times):
        sl = ds_main[var_name].sel(time=t, method='nearest')
        u0 = float(wind_ds['u10'].sel(time=t, method='nearest',
                                      latitude=COORD_TPP3[0],
                                      longitude=COORD_TPP3[1]))
        v0 = float(wind_ds['v10'].sel(time=t, method='nearest',
                                      latitude=COORD_TPP3[0],
                                      longitude=COORD_TPP3[1]))
        mat[i, :] = sample_transect(sl, u0, v0, distances)
    return times, mat

# -- Statistics & anomalies

def compute_anomaly(da, window='72h'):
    ts = da.to_series()
    bg = ts.rolling(window=window, center=True, min_periods=1).mean()
    return ts - bg

def compute_wind_stats(wind_ds):
    u = wind_ds['u10'].squeeze()
    v = wind_ds['v10'].squeeze()
    speed = np.sqrt(u**2 + v**2)
    direction = (np.degrees(np.arctan2(v, u)) + 360) % 360
    return speed, direction

# -- Plotting separate figures

def plot_separate(co2_ds, wind_ds):
    co2_sub = subset_region(co2_ds, *COORD_TPP3, REGION_RADIUS)
    wind_sub = subset_region(wind_ds, *COORD_TPP3, REGION_RADIUS)
    nox_ds = generate_nox_data(co2_sub)
    distances = np.arange(0, MAX_DIST + DIST_STEP, DIST_STEP)

    # Transects
    times, co2_mat = build_transect_matrix(co2_sub, wind_sub, distances, 'co2')
    _, nox_mat = build_transect_matrix(nox_ds, wind_sub, distances, 'nox')

    # Point time series
    co2_ts  = co2_sub['co2'].sel(latitude=COORD_TPP3[0],
                                 longitude=COORD_TPP3[1], method='nearest')
    nox_ts  = nox_ds['nox'].sel(latitude=COORD_TPP3[0],
                                 longitude=COORD_TPP3[1], method='nearest')

    # Anomalies and wind stats
    co2_anom = compute_anomaly(co2_ts)
    nox_anom = compute_anomaly(nox_ts)
    speed, direction = compute_wind_stats(wind_sub)

    # 1
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(distances/1000, times, co2_mat, shading='auto', cmap='viridis')
    plt.title('Шлейф CO₂: расстояние–время')
    plt.xlabel('Расстояние, км'); plt.ylabel('Время')
    plt.colorbar(label='CO₂, ppm'); plt.grid(True)
    plt.tight_layout(); plt.show()

    # 2
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(distances/1000, times, nox_mat, shading='auto', cmap='plasma')
    plt.title('Шлейф NOₓ: расстояние–время')
    plt.xlabel('Расстояние, км'); plt.ylabel('Время')
    plt.colorbar(label='NOₓ, ppm'); plt.grid(True)
    plt.tight_layout(); plt.show()

    # 3
    plt.figure(figsize=(8, 4))
    plt.plot(co2_ts['time'], co2_ts, lw=1)
    plt.title('Временной ряд CO₂ в точке ТЭЦ-3')
    plt.xlabel('Время'); plt.ylabel('CO₂, ppm')
    plt.grid(True); plt.xticks(rotation=30)
    plt.tight_layout(); plt.show()

    # 4
    plt.figure(figsize=(8, 4))
    plt.plot(nox_ts['time'], nox_ts, lw=1, color='orange')
    plt.title('Временной ряд NOₓ в точке ТЭЦ-3')
    plt.xlabel('Время'); plt.ylabel('NOₓ, ppm')
    plt.grid(True); plt.xticks(rotation=30)
    plt.tight_layout(); plt.show()

    # 5
    plt.figure(figsize=(8, 4))
    plt.plot(co2_anom.index, co2_anom, lw=1)
    plt.axhline(0, ls='--', color='k')
    plt.title('Аномалия CO₂ против 3-дневного фона')
    plt.xlabel('Время'); plt.ylabel('Δ CO₂, ppm')
    plt.grid(True); plt.xticks(rotation=30)
    plt.tight_layout(); plt.show()

    # 6
    plt.figure(figsize=(8, 4))
    plt.plot(nox_anom.index, nox_anom, lw=1, color='orange')
    plt.axhline(0, ls='--', color='k')
    plt.title('Аномалия NOₓ против 3-дневного фона')
    plt.xlabel('Время'); plt.ylabel('Δ NOₓ, ppm')
    plt.grid(True); plt.xticks(rotation=30)
    plt.tight_layout(); plt.show()

    # 7
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    bins = np.arange(0, 361, 20)
    freq, _ = np.histogram(direction, bins=bins)
    angles = np.radians(bins[:-1] + 10)
    ax.bar(angles, freq, width=np.radians(20), edgecolor='k', alpha=0.7)
    ax.set_title('Роза ветров', y=1.1)
    fig.tight_layout(); plt.show()

    # 8
    plt.figure(figsize=(8, 4))
    plt.plot(wind_sub['time'], speed, lw=1)
    plt.axhline(speed.mean(), ls='--', label=f'Среднее {speed.mean():.2f}')
    plt.title('Тренд скорости ветра')
    plt.xlabel('Время'); plt.ylabel('Скорость, м/с')
    plt.legend(); plt.grid(True); plt.xticks(rotation=30)
    plt.tight_layout(); plt.show()

    # 9
    plt.figure(figsize=(6,6))
    plt.scatter(no2_conc, nox_ts, alpha=0.5)
    plt.title('Зависимость между NOₓ и NO₂ в точке источника')
    plt.xlabel('NO₂, ppm'); plt.ylabel('NOₓ, ppm')
    plt.grid(True); plt.tight_layout(); plt.show()

    # 10
    plt.figure(figsize=(8,4))
    plt.hist(co2_anom.dropna(), bins=30, alpha=0.7, label='ΔCO₂')
    plt.hist(nox_anom.dropna(), bins=30, alpha=0.7, label='ΔNOₓ')
    plt.title('Распределение аномалий CO₂ и NOₓ')
    plt.xlabel('Аномалия, ppm'); plt.ylabel('Частота')
    plt.legend(); plt.tight_layout(); plt.show()

    # 11
    df = pd.DataFrame({
        'co2': co2_ts.to_series(),
        'nox': nox_ts.to_series()
    })
    df['month'] = df.index.month
    monthly = df.groupby('month').mean()
    monthly = monthly.reindex(index=range(1, 13), fill_value=np.nan)
    x = np.arange(1, 13)
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, monthly['co2'], width, label='CO₂')
    plt.bar(x + width / 2, monthly['nox'], width, label='NOₓ')
    plt.title('Средние месячные концентрации в точке источника')
    plt.xlabel('Месяц')
    plt.ylabel('Концентрация, ppm')
    plt.xticks(x)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # 12
    max_dist = []
    for row in co2_mat:
        idx = np.where(~np.isnan(row))[0]
        max_dist.append(distances[idx.max()]/1000 if idx.size else np.nan)

    lat0, lon0 = COORD_TPP3
    u_pt = wind_sub['u10'].sel(latitude=lat0, longitude=lon0, method='nearest')
    v_pt = wind_sub['v10'].sel(latitude=lat0, longitude=lon0, method='nearest')
    speed_pt = (np.sqrt(u_pt**2 + v_pt**2)
                .to_series()
                .reindex(pd.DatetimeIndex(times), method='nearest')
                .values)

    n = min(len(speed_pt), len(max_dist))
    plt.figure(figsize=(6, 6))
    plt.scatter(speed_pt[:n], max_dist[:n], alpha=0.6)
    plt.title('Зависимость дальности шлейфа CO₂ от скорости ветра')
    plt.xlabel('Скорость ветра, м/с'); plt.ylabel('Макс. дистанция шлейфа, км')
    plt.grid(True); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    no2_ds   = load_no2_data(NO2_FILES)
    wind_ds  = load_wind_data(WIND_FILES)
    plot_separate(no2_ds, wind_ds)
