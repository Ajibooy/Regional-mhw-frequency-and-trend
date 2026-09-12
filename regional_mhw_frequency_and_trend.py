# ==============================================================
# TROPICAL NORTH EAST ATLANTIC
#
# REGIONAL ANNUAL MHW FREQUENCY
# AND RELATIONSHIP WITH ANNUAL MEAN SST
#
# Region:
#   0-30°N, 60-10°W
#
# Climatological baseline:
#   1981-2010
#
# Annual analysis:
#   1982-2024
#
# MHW definition:
#   SST > daily P90 for >= 5 consecutive days
#
# Daily P90:
#   +/-5-day climatological window
#   31-day circular smoothing
#
# PANEL (a):
#   Regional average annual MHW frequency
#   Units = Count/year
#
# PANEL (b):
#   Annual Mean SST vs Regional MHW Frequency
#   Black line = best-fit linear regression
#   R² displayed
#
# Regional averages use cosine(latitude) area weighting.
# ==============================================================


import os
import glob
import gc
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from scipy.stats import linregress

warnings.filterwarnings("ignore")


# ==============================================================
# 1. SETTINGS
# ==============================================================

DATA_DIR = r"C:\Users\Aina Ajibola\Desktop\oisst_data"

LAT_MIN = 0.0
LAT_MAX = 30.0

LON_MIN = -60.0
LON_MAX = -10.0


# --------------------------------------------------------------
# Climatological baseline
# --------------------------------------------------------------

BASE_START = "1981-01-01"
BASE_END = "2010-12-31"


# --------------------------------------------------------------
# Annual analysis
# --------------------------------------------------------------

START_YEAR = 1982
END_YEAR = 2024

ANALYSIS_END = "2024-12-31"


# --------------------------------------------------------------
# MHW parameters
# --------------------------------------------------------------

PERCENTILE = 90

HALF_WINDOW = 5

SMOOTH_WINDOW = 31

MIN_DURATION = 5


# --------------------------------------------------------------
# Memory processing
# --------------------------------------------------------------

LAT_BLOCK_SIZE = 5


# ==============================================================
# 2. PREPROCESS OISST
# ==============================================================

def preprocess(ds):

    rename = {}

    for old, new in {
        "latitude": "lat",
        "longitude": "lon",
        "Time": "time",
        "TIME": "time"
    }.items():

        if old in ds.coords or old in ds.dims:
            rename[old] = new


    if rename:
        ds = ds.rename(rename)


    # ----------------------------------------------------------
    # Remove singleton vertical dimensions
    # ----------------------------------------------------------

    for dim in [
        "zlev",
        "depth",
        "lev",
        "level"
    ]:

        if (
            dim in ds.dims
            and
            ds.sizes[dim] == 1
        ):

            ds = ds.squeeze(
                dim,
                drop=True
            )


    if "sst" not in ds.data_vars:

        raise KeyError(
            "Variable 'sst' was not found."
        )


    ds = ds[["sst"]]


    # ----------------------------------------------------------
    # Convert longitude 0-360 -> -180...180
    # ----------------------------------------------------------

    if float(ds.lon.max()) > 180:

        ds = ds.assign_coords(
            lon=((ds.lon + 180.0) % 360.0) - 180.0
        )


    ds = ds.sortby("lat")
    ds = ds.sortby("lon")


    # ----------------------------------------------------------
    # Select region
    # ----------------------------------------------------------

    ds = ds.sel(

        lat=slice(
            LAT_MIN,
            LAT_MAX
        ),

        lon=slice(
            LON_MIN,
            LON_MAX
        )

    )


    return ds


# ==============================================================
# 3. CLIMATOLOGICAL DAY
#
# Map dates to leap year 2000:
#
# Jan 1  = 1
# Feb 29 = 60
# Dec 31 = 366
# ==============================================================

def get_clim_day(dates):

    dates = pd.DatetimeIndex(
        dates
    )


    reference = pd.to_datetime(
        {
            "year":
                np.full(
                    len(dates),
                    2000
                ),

            "month":
                dates.month,

            "day":
                dates.day
        }
    )


    return (
        pd.DatetimeIndex(reference)
        .dayofyear
        .to_numpy(dtype=np.int16)
    )


# ==============================================================
# 4. 31-DAY CIRCULAR SMOOTHING
# ==============================================================

def circular_smooth_3d(
    values,
    window=31
):

    half = window // 2


    extended = np.concatenate(
        [
            values[-half:, :, :],
            values,
            values[:half, :, :]
        ],
        axis=0
    )


    valid = np.isfinite(
        extended
    )


    filled = np.where(
        valid,
        extended,
        0.0
    )


    csum = np.cumsum(
        filled,
        axis=0,
        dtype=np.float64
    )


    ccount = np.cumsum(
        valid.astype(np.int32),
        axis=0
    )


    csum = np.concatenate(
        [
            np.zeros(
                (
                    1,
                    csum.shape[1],
                    csum.shape[2]
                ),
                dtype=np.float64
            ),

            csum
        ],
        axis=0
    )


    ccount = np.concatenate(
        [
            np.zeros(
                (
                    1,
                    ccount.shape[1],
                    ccount.shape[2]
                ),
                dtype=np.int32
            ),

            ccount
        ],
        axis=0
    )


    smoothed = np.full(
        values.shape,
        np.nan,
        dtype=np.float32
    )


    for d in range(366):

        start = d
        end = d + window


        total = (
            csum[end]
            -
            csum[start]
        )


        count = (
            ccount[end]
            -
            ccount[start]
        )


        np.divide(
            total,
            count,
            out=smoothed[d],
            where=(count > 0)
        )


    return smoothed


# ==============================================================
# 5. ANNUAL MHW FREQUENCY
#
# Each confirmed >=5-day event is counted once.
#
# Events crossing Dec-Jan remain ONE event and are assigned
# to the year in which the event STARTS.
# ==============================================================

def calculate_annual_frequency(
    above,
    dates,
    years
):

    n_time, n_lat, n_lon = (
        above.shape
    )


    n_cells = (
        n_lat
        *
        n_lon
    )


    hot = above.reshape(
        n_time,
        n_cells
    )


    annual_frequency = np.zeros(
        (
            len(years),
            n_cells
        ),
        dtype=np.int16
    )


    for cell in range(
        n_cells
    ):

        series = hot[
            :,
            cell
        ]


        start = None


        for t in range(
            n_time
        ):

            is_hot = bool(
                series[t]
            )


            # --------------------------------------------------
            # Start event
            # --------------------------------------------------

            if (
                is_hot
                and
                start is None
            ):

                start = t


            # --------------------------------------------------
            # Finish event
            # --------------------------------------------------

            if start is not None:

                run_finished = (

                    (not is_hot)

                    or

                    (
                        t
                        ==
                        n_time - 1
                    )

                )


                if run_finished:

                    if (
                        is_hot
                        and
                        t == n_time - 1
                    ):

                        end = t

                    else:

                        end = t - 1


                    duration = (
                        end
                        -
                        start
                        +
                        1
                    )


                    # ------------------------------------------
                    # Confirm MHW
                    # ------------------------------------------

                    if duration >= MIN_DURATION:

                        event_year = int(
                            dates[start].year
                        )


                        if (
                            START_YEAR
                            <=
                            event_year
                            <=
                            END_YEAR
                        ):

                            year_index = (
                                event_year
                                -
                                START_YEAR
                            )


                            annual_frequency[
                                year_index,
                                cell
                            ] += 1


                    start = None


    return annual_frequency.reshape(
        len(years),
        n_lat,
        n_lon
    )


# ==============================================================
# 6. FIND FILES
# ==============================================================

files = sorted(

    glob.glob(

        os.path.join(
            DATA_DIR,
            "*_oisst.nc"
        )

    )

)


if not files:

    files = sorted(

        glob.glob(

            os.path.join(
                DATA_DIR,
                "*.nc"
            )

        )

    )


if not files:

    raise FileNotFoundError(
        f"No NetCDF files found in:\n{DATA_DIR}"
    )


print("=" * 90)

print(
    "REGIONAL MHW FREQUENCY AND SST ANALYSIS"
)

print("=" * 90)


print(
    f"\nFiles found: "
    f"{len(files):,}"
)


print(
    f"First file: "
    f"{os.path.basename(files[0])}"
)


print(
    f"Last file: "
    f"{os.path.basename(files[-1])}"
)


# ==============================================================
# 7. OPEN OISST
# ==============================================================

print(
    "\nOpening OISST..."
)


ds = xr.open_mfdataset(

    files,

    combine="by_coords",

    preprocess=preprocess,

    parallel=False,

    data_vars="minimal",

    coords="minimal",

    compat="override",

    join="outer",

    engine="netcdf4"

)


ds = ds.sortby(
    "time"
)


sst = ds[
    "sst"
]


# ==============================================================
# 8. NORMALIZE TIME
# ==============================================================

time_index = pd.DatetimeIndex(
    sst.time.values
).normalize()


sst = sst.assign_coords(
    time=time_index
)


# Remove duplicate dates

keep = np.where(

    ~time_index.duplicated(
        keep="first"
    )

)[0]


sst = sst.isel(
    time=keep
)


sst = sst.sortby(
    "time"
)


# ==============================================================
# 9. SST UNIT CHECK
# ==============================================================

sample = float(

    sst.isel(

        time=slice(
            0,
            min(
                10,
                sst.sizes["time"]
            )
        )

    )

    .mean(
        skipna=True
    )

    .compute()

)


if sample > 100:

    print(
        "\nConverting SST from Kelvin to °C..."
    )

    sst = sst - 273.15


else:

    print(
        "\nSST already appears to be °C."
    )


# ==============================================================
# 10. BASELINE
# ==============================================================

baseline = sst.sel(

    time=slice(
        BASE_START,
        BASE_END
    )

)


baseline_dates = pd.DatetimeIndex(
    baseline.time.values
)


if len(baseline_dates) == 0:

    raise ValueError(
        "No baseline data found."
    )


print(
    f"\nBaseline available: "
    f"{baseline_dates[0].date()} "
    f"to {baseline_dates[-1].date()}"
)


if baseline_dates[0] > pd.Timestamp(
    BASE_START
):

    print(
        "\nWARNING:"
    )

    print(
        "1981 is incomplete in your local OISST archive."
    )


baseline_clim_day = get_clim_day(
    baseline_dates
)


# ==============================================================
# 11. ANALYSIS PERIOD
#
# Include available 1981 data internally so Dec-Jan events
# can be detected continuously.
# ==============================================================

analysis_start = pd.Timestamp(
    sst.time.values[0]
)


analysis = sst.sel(

    time=slice(
        analysis_start,
        ANALYSIS_END
    )

)


analysis_dates = pd.DatetimeIndex(
    analysis.time.values
)


analysis_clim_day = get_clim_day(
    analysis_dates
)


years = np.arange(
    START_YEAR,
    END_YEAR + 1
)


n_years = len(
    years
)


# ==============================================================
# 12. GRID DIMENSIONS
# ==============================================================

n_lat = sst.sizes[
    "lat"
]


n_lon = sst.sizes[
    "lon"
]


annual_frequency = np.zeros(

    (
        n_years,
        n_lat,
        n_lon
    ),

    dtype=np.int16

)


# ==============================================================
# 13. PROCESS LATITUDE BLOCKS
# ==============================================================

total_blocks = int(

    np.ceil(
        n_lat
        /
        LAT_BLOCK_SIZE
    )

)


print(
    f"\nProcessing "
    f"{total_blocks} latitude blocks..."
)


for block_number, block_start in enumerate(

    range(
        0,
        n_lat,
        LAT_BLOCK_SIZE
    ),

    start=1

):


    block_end = min(

        block_start
        +
        LAT_BLOCK_SIZE,

        n_lat

    )


    block_lat = (
        block_end
        -
        block_start
    )


    print(
        f"\nBlock "
        f"{block_number}/{total_blocks}"
    )


    # ==========================================================
    # 13A. LOAD BASELINE
    # ==========================================================

    base_block = np.asarray(

        baseline.isel(

            lat=slice(
                block_start,
                block_end
            )

        ).values,

        dtype=np.float32

    )


    # ==========================================================
    # 13B. DAILY P90
    # ==========================================================

    daily_p90 = np.full(

        (
            366,
            block_lat,
            n_lon
        ),

        np.nan,

        dtype=np.float32

    )


    for day in range(
        1,
        367
    ):

        distance = np.abs(
            baseline_clim_day
            -
            day
        )


        distance = np.minimum(
            distance,
            366
            -
            distance
        )


        selected = (
            distance
            <=
            HALF_WINDOW
        )


        selected_sst = base_block[
            selected,
            :,
            :
        ]


        daily_p90[
            day - 1
        ] = np.nanpercentile(

            selected_sst,

            PERCENTILE,

            axis=0

        )


    # ==========================================================
    # 13C. SMOOTH P90
    # ==========================================================

    daily_p90 = circular_smooth_3d(

        daily_p90,

        SMOOTH_WINDOW

    )


    # ==========================================================
    # 13D. ANALYSIS SST
    # ==========================================================

    analysis_block = np.asarray(

        analysis.isel(

            lat=slice(
                block_start,
                block_end
            )

        ).values,

        dtype=np.float32

    )


    # ==========================================================
    # 13E. MATCH P90 TO DATES
    # ==========================================================

    threshold_block = daily_p90[
        analysis_clim_day - 1
    ]


    # ==========================================================
    # 13F. ABOVE P90
    # ==========================================================

    valid = (

        np.isfinite(
            analysis_block
        )

        &

        np.isfinite(
            threshold_block
        )

    )


    above = (

        valid

        &

        (
            analysis_block
            >
            threshold_block
        )

    )


    # ==========================================================
    # 13G. DETECT EVENTS
    # ==========================================================

    block_frequency = calculate_annual_frequency(

        above,

        analysis_dates,

        years

    )


    annual_frequency[
        :,
        block_start:block_end,
        :
    ] = block_frequency


    del base_block
    del daily_p90
    del analysis_block
    del threshold_block
    del valid
    del above
    del block_frequency

    gc.collect()


# ==============================================================
# 14. OCEAN MASK
# ==============================================================

print(
    "\nCreating ocean mask..."
)


ocean_mask = np.isfinite(

    sst.sel(

        time="2000-01-01",

        method="nearest"

    ).values

)


annual_frequency_float = (
    annual_frequency.astype(
        np.float32
    )
)


annual_frequency_float[
    :,
    ~ocean_mask
] = np.nan


# ==============================================================
# 15. AREA WEIGHTS
#
# OISST is regular lat/lon.
#
# Grid-cell area is proportional to cos(latitude).
# ==============================================================

latitudes = sst.lat.values


weights_1d = np.cos(
    np.deg2rad(
        latitudes
    )
)


weights_2d = np.broadcast_to(

    weights_1d[:, None],

    (
        n_lat,
        n_lon
    )

).astype(
    np.float64
)


# Remove land from weights

weights_2d = np.where(

    ocean_mask,

    weights_2d,

    np.nan

)


# ==============================================================
# 16. REGIONAL ANNUAL MHW FREQUENCY
#
# Weighted spatial average of grid-cell annual event counts.
#
# Units = Count/year
# ==============================================================

print(
    "\nCalculating regional annual MHW frequency..."
)


regional_annual_frequency = np.full(

    n_years,

    np.nan,

    dtype=np.float64

)


for yi in range(
    n_years
):

    field = annual_frequency_float[
        yi
    ]


    valid = (

        np.isfinite(field)

        &

        np.isfinite(weights_2d)

    )


    regional_annual_frequency[
        yi
    ] = np.sum(

        field[valid]
        *
        weights_2d[valid]

    ) / np.sum(

        weights_2d[valid]

    )


# ==============================================================
# 17. REGIONAL ANNUAL MEAN SST
#
# Step 1:
# Annual mean SST at each grid cell.
#
# Step 2:
# cosine(latitude)-weighted regional average.
#
# Units = °C
# ==============================================================

print(
    "Calculating regional annual mean SST..."
)


regional_annual_sst = np.full(

    n_years,

    np.nan,

    dtype=np.float64

)


for yi, year in enumerate(
    years
):

    print(
        f"Annual SST: {year}"
    )


    year_sst = sst.sel(

        time=slice(
            f"{year}-01-01",
            f"{year}-12-31"
        )

    )


    # ----------------------------------------------------------
    # Annual mean at every grid cell
    # ----------------------------------------------------------

    annual_sst_field = np.asarray(

        year_sst.mean(
            dim="time",
            skipna=True
        ).compute().values,

        dtype=np.float64

    )


    # ----------------------------------------------------------
    # Area-weighted regional mean
    # ----------------------------------------------------------

    valid = (

        np.isfinite(
            annual_sst_field
        )

        &

        np.isfinite(
            weights_2d
        )

    )


    regional_annual_sst[
        yi
    ] = np.sum(

        annual_sst_field[valid]
        *
        weights_2d[valid]

    ) / np.sum(

        weights_2d[valid]

    )


# ==============================================================
# 18. LINEAR REGRESSION
#
# X = annual regional mean SST
# Y = annual regional MHW frequency
# ==============================================================

valid_regression = (

    np.isfinite(
        regional_annual_sst
    )

    &

    np.isfinite(
        regional_annual_frequency
    )

)


x = regional_annual_sst[
    valid_regression
]


y = regional_annual_frequency[
    valid_regression
]


regression = linregress(
    x,
    y
)


slope = regression.slope

intercept = regression.intercept

r_value = regression.rvalue

p_value = regression.pvalue

r_squared = (
    r_value ** 2
)


# ==============================================================
# 19. REGRESSION LINE
# ==============================================================

x_line = np.linspace(

    np.nanmin(x),

    np.nanmax(x),

    200

)


y_line = (

    slope
    *
    x_line

    +

    intercept

)


# ==============================================================
# 20. PRINT RESULTS
# ==============================================================

print(
    "\n"
    + "=" * 90
)


print(
    "REGIONAL RESULTS"
)


print(
    "=" * 90
)


for year, freq, temp in zip(

    years,

    regional_annual_frequency,

    regional_annual_sst

):

    print(
        f"{year}:  "
        f"MHW Frequency = {freq:.2f} count/year   "
        f"Annual SST = {temp:.2f} °C"
    )


print(
    "\nRegression:"
)


print(
    f"Slope = "
    f"{slope:.3f} count °C⁻¹"
)


print(
    f"Intercept = "
    f"{intercept:.3f}"
)


print(
    f"R = "
    f"{r_value:.3f}"
)


print(
    f"R² = "
    f"{r_squared:.3f}"
)


print(
    f"p-value = "
    f"{p_value:.6f}"
)


# ==============================================================
# 21. CREATE FIGURE
# ==============================================================

fig, axes = plt.subplots(

    1,
    2,

    figsize=(
        14,
        5.8
    )

)


# ==============================================================
# 22. PANEL A
#
# REGIONAL ANNUAL MHW FREQUENCY
# ==============================================================

ax = axes[0]


ax.bar(

    years,

    regional_annual_frequency,

    width=0.72,

    edgecolor="black",

    linewidth=0.35

)


# --------------------------------------------------------------
# Labels
# --------------------------------------------------------------

ax.set_xlabel(

    "Year",

    fontsize=11,

    fontweight="bold"

)


ax.set_ylabel(

    "MHW Frequency [Count]",

    fontsize=11,

    fontweight="bold"

)


ax.set_title(

    "(a) MHW Frequency",

    fontsize=12,

    fontweight="bold",

    loc="left"

)


# --------------------------------------------------------------
# X-axis
# ==============================================================

ax.set_xlim(
    START_YEAR - 1,
    END_YEAR + 1
)


# Every second year, similar to reference
xticks = np.arange(

    START_YEAR,

    END_YEAR + 1,

    2

)


ax.set_xticks(
    xticks
)


ax.set_xticklabels(

    xticks,

    rotation=90,

    fontsize=8

)


# --------------------------------------------------------------
# Y axis starts at zero
# --------------------------------------------------------------

ax.set_ylim(
    bottom=0
)


ax.yaxis.set_major_locator(

    mticker.MaxNLocator(
        integer=True
    )

)


# --------------------------------------------------------------
# Grid
# --------------------------------------------------------------

ax.grid(

    True,

    linestyle="-",

    linewidth=0.4,

    alpha=0.35

)


ax.set_axisbelow(
    True
)


# ==============================================================
# 23. PANEL B
#
# ANNUAL SST vs MHW FREQUENCY
# ==============================================================

ax = axes[1]


# --------------------------------------------------------------
# Scatter
# --------------------------------------------------------------

ax.scatter(

    regional_annual_sst,

    regional_annual_frequency,

    s=32,

    zorder=3

)


# --------------------------------------------------------------
# Regression line
# --------------------------------------------------------------

ax.plot(

    x_line,

    y_line,

    color="black",

    linewidth=1.5,

    zorder=2

)


# --------------------------------------------------------------
# Labels
# --------------------------------------------------------------

ax.set_xlabel(

    "Annual Mean SST [°C]",

    fontsize=11,

    fontweight="bold"

)


ax.set_ylabel(

    "MHW Frequency [Count]",

    fontsize=11,

    fontweight="bold"

)


ax.set_title(

    "(b) MHW Frequency",

    fontsize=12,

    fontweight="bold",

    loc="left"

)


# --------------------------------------------------------------
# Grid
# --------------------------------------------------------------

ax.grid(

    True,

    linestyle="-",

    linewidth=0.4,

    alpha=0.35

)


ax.set_axisbelow(
    True
)


# ==============================================================
# 24. R² ANNOTATION
#
# Place at bottom-right, similar to sample.
# ==============================================================

ax.text(

    0.97,

    0.06,

    f"R²={r_squared:.2f}",

    transform=ax.transAxes,

    horizontalalignment="right",

    verticalalignment="bottom",

    fontsize=10,

    fontweight="bold"

)


# ==============================================================
# 25. CONSISTENT PANEL APPEARANCE
# ==============================================================

for ax in axes:

    ax.tick_params(
        direction="out",
        width=0.8
    )


    for spine in ax.spines.values():

        spine.set_linewidth(
            0.8
        )


# ==============================================================
# 26. MAIN TITLE
# ==============================================================

fig.suptitle(

    "Regional Annual Marine Heatwave Frequency "
    "in the Tropical North East Atlantic (1982–2024)",

    fontsize=14,

    fontweight="bold",

    y=0.99

)


# ==============================================================
# 27. LAYOUT
# ==============================================================

plt.tight_layout(

    rect=[
        0,
        0,
        1,
        0.95
    ]

)


# ==============================================================
# 28. SHOW
# ==============================================================

plt.show()


# ==============================================================
# 29. CLOSE DATASET
# ==============================================================

ds.close()


print(
    "\nRegional MHW frequency–SST analysis completed successfully."
)