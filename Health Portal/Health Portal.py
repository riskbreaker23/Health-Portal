"""
Samsung Health Dashboard Generator
====================================
Reads Samsung Health CSV exports and produces a self-contained HTML dashboard.

Usage:
    python samsung_health_dashboard.py
        (uses the sample_data folder next to this script)

    python samsung_health_dashboard.py --data-dir /path/to/samsung_health_export
        (uses your own Samsung Health export)

Samsung Health exports data into a folder like:
    com.samsung.shealth.YYYY-MM-DD/
        jsons/
        csvs/
            com.samsung.health.step_daily_trend.YYYYMMDD.csv
            com.samsung.health.sleep.YYYYMMDD.csv
            com.samsung.health.heart_rate.YYYYMMDD.csv
            com.samsung.health.exercise.YYYYMMDD.csv
            com.samsung.health.food_intake.YYYYMMDD.csv
            com.samsung.health.weight.YYYYMMDD.csv
            ...

Run with --data-dir pointing to your export folder (or the csvs/ subfolder).
The script will auto-detect all matching files.

Requirements:
    pip install pandas plotly
"""

import os
import sys
import glob
import argparse
import json
import re
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

OUTPUT_FILE = "samsung_health_dashboard.html"
DAYS_DEFAULT = 9999  # show all available data by default
SCRIPT_DIR = Path(__file__).resolve().parent  # folder this script lives in

# Samsung Health CSV file patterns (partial match)
FILE_PATTERNS = {
    "steps":    ["step_daily_trend"],
    "sleep":    ["sleep_combined"],
    "heart":    ["tracker.heart_rate"],
    "exercise": ["shealth.exercise.20"],  # main exercise log, not weather
    "food":     ["health.nutrition"],
    "weight":   ["health.weight"],
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def find_csv(data_dir: Path, patterns: list[str]) -> Path | None:
    """Find the first CSV in data_dir matching any of the given name patterns."""
    for f in sorted(data_dir.rglob("*.csv"), reverse=True):
        for p in patterns:
            if p in f.name.lower():
                return f
    return None


def read_csv_skip_metadata(path: Path) -> pd.DataFrame:
    """
    Samsung Health CSVs always have exactly 1 metadata row at the top:
        row 0: com.samsung.shealth.whatever,<user_id>,<count>   <- skip this
        row 1: start_time,end_time,heart_rate,...               <- REAL header
        row 2+: data

    We skip row 0 and use row 1 as the header.
    """
    # Samsung Health CSVs: row 0 = metadata, row 1 = header, row 2+ = data.
    # Data rows have one extra leading blank column the header lacks.
    # index_col=0 absorbs it and keeps columns aligned.
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(
                path,
                skiprows=1,
                index_col=0,
                encoding=enc,
                on_bad_lines="skip",
                low_memory=False,
            )
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"  Warning: could not parse {path.name}: {e}")
            return pd.DataFrame()
    print(f"  Warning: encoding failed for {path.name}")
    return pd.DataFrame()


def coerce_datetime(series: pd.Series) -> pd.Series:
    """Try multiple datetime formats Samsung Health uses."""
    for fmt in [None, "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
        try:
            if fmt:
                return pd.to_datetime(series, format=fmt, errors="coerce")
            else:
                return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
        except Exception:
            continue
    return pd.to_datetime(series, errors="coerce")


def col_pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Return first column name from candidates that exists in df (case-insensitive).
    Also matches Samsung Health fully-qualified names like
    'com.samsung.health.heart_rate.heart_rate' against candidate 'heart_rate'.
    """
    # Build two maps: exact lowercase, and suffix-after-last-dot lowercase
    lower_map = {c.lower(): c for c in df.columns}
    suffix_map = {c.split(".")[-1].lower(): c for c in df.columns}
    for c in candidates:
        key = c.lower()
        if key in lower_map:
            return lower_map[key]
        if key in suffix_map:
            return suffix_map[key]
    return None


# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────

def load_steps(path: Path, days: int) -> pd.DataFrame:
    """
    Uses step_daily_trend (7584 rows of daily history).
    After index_col=0 absorbs the leading binning_data filename:
      binning_data slot [00] = update_time (date)
      source_type slot  [03] = step count (integer)
      distance slot     [07] = distance (meters)
      calorie slot      [08] = deviceuuid (skip)
      datauuid slot     [11] = day_time as Unix ms timestamp (use as canonical date)
    """
    df = read_csv_skip_metadata(path)
    if df.empty:
        return df
    # datauuid holds the canonical day_time as Unix ms
    date_col = col_pick(df, ["datauuid"])
    step_col = col_pick(df, ["source_type"])
    dist_col = col_pick(df, ["distance"])
    if not date_col or not step_col:
        print(f"  Steps: could not identify required columns. Found: {list(df.columns)}")
        return pd.DataFrame()
    # datauuid is a Unix ms timestamp
    df["date"] = pd.to_datetime(
        pd.to_numeric(df[date_col], errors="coerce"), unit="ms", errors="coerce"
    ).dt.normalize()
    df["steps"] = pd.to_numeric(df[step_col], errors="coerce")
    # Filter to primary source only (source_pkg_name=0) to avoid double-counting
    src_col = col_pick(df, ["source_pkg_name"])
    if src_col:
        primary = df[pd.to_numeric(df[src_col], errors="coerce") == 0]
        if len(primary) > 0:
            df = primary
    df = df[df["steps"] > 0]
    # Take max per day (not sum) — multiple rows can exist per day per source
    agg = {"steps": "max"}
    if dist_col:
        df["distance_km"] = pd.to_numeric(df[dist_col], errors="coerce") / 1000
        agg["distance_km"] = "max"
    daily = df.groupby("date", as_index=False).agg(agg)
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    full_range = f"{daily['date'].min().date()} to {daily['date'].max().date()}" if not daily.empty else "n/a"
    daily = daily[daily["date"] >= cutoff].sort_values("date")
    if daily.empty:
        print(f"    ℹ Steps date range in file: {full_range} (nothing in last {days} days)")
    return daily


def load_sleep(path: Path, days: int) -> pd.DataFrame:
    """
    Uses sleep_combined. Columns are misaligned by 1 after index_col=0,
    so we use the actual column names that land on real values:
      create_sh_ver      -> start datetime
      stage_analyzed_type -> end datetime
    Duration computed from end - start.
    """
    df = read_csv_skip_metadata(path)
    if df.empty:
        return df
    # After index_col=0, these columns hold the actual start/end values
    start_col = col_pick(df, ["create_sh_ver"])
    end_col   = col_pick(df, ["stage_analyzed_type"])
    if not start_col:
        print(f"  Sleep: could not identify columns. Found: {list(df.columns)[:10]}")
        return pd.DataFrame()
    df["start"] = coerce_datetime(df[start_col])
    df["end"]   = coerce_datetime(df[end_col]) if end_col else pd.NaT
    df["duration_h"] = (df["end"] - df["start"]).dt.total_seconds() / 3600
    # Credit each sleep to the day you wake up, so a night that starts after
    # midnight isn't merged with the previous night
    df["date"] = df["end"].dt.normalize() if end_col else df["start"].dt.normalize()
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    full_range = f"{df['date'].min().date()} to {df['date'].max().date()}" if not df.empty else "n/a"
    df = df[(df["duration_h"] > 0.5) & (df["duration_h"] < 16)]
    df = df[df["date"] >= cutoff]
    if df.empty:
        print(f"    ℹ Sleep date range in file: {full_range} (nothing usable in last {days} days)")
    return df[["date", "start", "duration_h"]].sort_values("date")


def load_heart_rate(path: Path, days: int) -> pd.DataFrame:
    df = read_csv_skip_metadata(path)
    if df.empty:
        return df
    # Columns are offset: heart_beat_count[03] holds start_time, datauuid[19] holds heart_rate
    time_col = col_pick(df, ["com.samsung.health.heart_rate.heart_beat_count",
                              "com.samsung.health.heart_rate.update_time"])
    hr_col   = col_pick(df, ["com.samsung.health.heart_rate.datauuid",
                              "com.samsung.health.heart_rate.heart_rate",
                              "com.samsung.health.heart_rate.max"])
    if not time_col or not hr_col:
        print(f"  Heart Rate: could not identify columns. Found: {list(df.columns)}")
        return pd.DataFrame()
    df["datetime"] = coerce_datetime(df[time_col])
    df["heart_rate"] = pd.to_numeric(df[hr_col], errors="coerce")
    df = df[(df["heart_rate"] > 30) & (df["heart_rate"] < 220)]
    df["date"] = df["datetime"].dt.normalize()
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    full_range = f"{df['date'].min().date()} to {df['date'].max().date()}" if not df.empty else "n/a"
    df = df[df["date"] >= cutoff].sort_values("datetime")
    if df.empty:
        print(f"    ℹ Heart Rate date range in file: {full_range} (nothing in last {days} days)")
    return df[["datetime", "date", "heart_rate"]]


def load_exercise(path: Path, days: int) -> pd.DataFrame:
    """
    Uses shealth.exercise — main exercise log.
    Key columns: start_time, exercise_type, total_calorie, duration, distance
    Duration is stored in milliseconds.
    """
    df = read_csv_skip_metadata(path)
    if df.empty:
        return df
    # Columns are offset: location_data[31] holds start_time, additional_internal[26] holds duration(ms)
    # exercise_type is in start_time[32] slot which holds the type code
    time_col = col_pick(df, ["com.samsung.health.exercise.location_data",
                              "com.samsung.health.exercise.update_time"])
    dur_col  = col_pick(df, ["additional_internal",
                              "com.samsung.health.exercise.duration"])
    cal_col  = col_pick(df, ["total_calorie",
                             "com.samsung.health.exercise.calorie"])
    dist_col = col_pick(df, ["com.samsung.health.exercise.distance"])
    type_col = col_pick(df, ["com.samsung.health.exercise.start_time",
                             "com.samsung.health.exercise.exercise_type"])
    if not time_col:
        print(f"  Exercise: could not identify date column. Found: {list(df.columns)}")
        return pd.DataFrame()
    df["date"] = coerce_datetime(df[time_col]).dt.normalize()
    if dur_col:
        # Samsung stores duration in milliseconds
        df["duration_min"] = pd.to_numeric(df[dur_col], errors="coerce") / 60000
        # If values look already in seconds (median < 300), convert from seconds
        if df["duration_min"].median() < 0.5:
            df["duration_min"] = pd.to_numeric(df[dur_col], errors="coerce") / 60
        df = df[df["duration_min"] > 0]
    if cal_col:
        df["calories"] = pd.to_numeric(df[cal_col], errors="coerce")
    if dist_col:
        df["distance_km"] = pd.to_numeric(df[dist_col], errors="coerce")
        if df["distance_km"].median() > 500:
            df["distance_km"] = df["distance_km"] / 1000
    EXERCISE_TYPES = {
        1001: "Walking", 11007: "Running", 10003: "Cycling", 10002: "Biking",
        11001: "Swimming", 11006: "Hiking", 10001: "Elliptical", 13001: "Weight Training",
        11002: "Yoga", 11005: "Pilates", 11003: "Dance", 11004: "Aerobics",
        90001: "Other", 5001: "Basketball", 5002: "Soccer", 5003: "Baseball",
        15006: "Running", 15007: "Walking", 10007: "Cycling",
    }
    if type_col:
        df["exercise_name"] = pd.to_numeric(df[type_col], errors="coerce").map(EXERCISE_TYPES).fillna("Other")
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    full_range = f"{df['date'].min().date()} to {df['date'].max().date()}" if not df.empty else "n/a"
    df = df[df["date"] >= cutoff].sort_values("date")
    if df.empty:
        print(f"    ℹ Exercise date range in file: {full_range} (nothing in last {days} days)")
    cols = ["date"] + (["exercise_name"] if type_col else []) + \
           (["duration_min"] if dur_col else []) + \
           (["calories"] if cal_col else []) + \
           (["distance_km"] if dist_col else [])
    return df[cols]


def load_food(path: Path, days: int) -> pd.DataFrame:
    df = read_csv_skip_metadata(path)
    if df.empty:
        return df
    time_col = col_pick(df, ["update_time", "start_time", "create_time"])
    cal_col  = col_pick(df, ["calorie", "calories", "kcal"])
    carb_col = col_pick(df, ["carbohydrate", "carbs", "total_carbohydrate"])
    prot_col = col_pick(df, ["protein", "total_protein"])
    fat_col  = col_pick(df, ["fat", "total_fat"])
    if not time_col or not cal_col:
        print(f"  Food: could not identify columns. Found: {list(df.columns)}")
        return pd.DataFrame()
    df["date"] = coerce_datetime(df[time_col]).dt.normalize()
    df["calories"] = pd.to_numeric(df[cal_col], errors="coerce")
    if carb_col: df["carbs_g"]   = pd.to_numeric(df[carb_col], errors="coerce")
    if prot_col: df["protein_g"] = pd.to_numeric(df[prot_col], errors="coerce")
    if fat_col:  df["fat_g"]     = pd.to_numeric(df[fat_col], errors="coerce")
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    full_range = f"{df['date'].min().date()} to {df['date'].max().date()}" if not df.empty else "n/a"
    df_filt = df[df["date"] >= cutoff]
    if df_filt.empty:
        print(f"    ℹ Food date range in file: {full_range} (nothing in last {days} days)")
        return pd.DataFrame()
    df = df_filt
    agg = {"calories": "sum"}
    if carb_col: agg["carbs_g"]   = "sum"
    if prot_col: agg["protein_g"] = "sum"
    if fat_col:  agg["fat_g"]     = "sum"
    daily = df.groupby("date", as_index=False).agg(agg).sort_values("date")
    return daily


def load_weight(path: Path, days: int) -> pd.DataFrame:
    df = read_csv_skip_metadata(path)
    if df.empty:
        return df
    time_col   = col_pick(df, ["update_time", "start_time", "create_time"])
    weight_col = col_pick(df, ["weight", "body_weight", "kg"])
    bmi_col    = col_pick(df, ["bmi", "body_mass_index"])
    fat_col    = col_pick(df, ["body_fat", "fat_mass", "fat_rate"])
    if not time_col or not weight_col:
        print(f"  Weight: could not identify columns. Found: {list(df.columns)}")
        return pd.DataFrame()
    df["date"] = coerce_datetime(df[time_col]).dt.normalize()
    df["weight_kg"] = pd.to_numeric(df[weight_col], errors="coerce")
    # Sanity: Samsung stores kg; if values look like lbs, convert
    if df["weight_kg"].median() > 200:
        df["weight_kg"] = df["weight_kg"] * 0.453592
    if bmi_col: df["bmi"] = pd.to_numeric(df[bmi_col], errors="coerce")
    if fat_col: df["body_fat_pct"] = pd.to_numeric(df[fat_col], errors="coerce")
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    full_range = f"{df['date'].min().date()} to {df['date'].max().date()}" if not df.empty else "n/a"
    df = df[df["date"] >= cutoff].sort_values("date")
    if df.empty:
        print(f"    ℹ Weight date range in file: {full_range} (nothing in last {days} days)")
    cols = ["date", "weight_kg"] + (["bmi"] if bmi_col else []) + (["body_fat_pct"] if fat_col else [])
    return df[cols]


# ─────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────

COLORS = {
    "steps":    "#00C9A7",
    "sleep":    "#A78BFA",
    "heart":    "#FF6B6B",
    "exercise": "#F9A825",
    "food":     "#4FC3F7",
    "weight":   "#A5D6A7",
    "bg":       "#0D1117",
    "card":     "#161B22",
    "text":     "#E6EDF3",
    "muted":    "#8B949E",
    "grid":     "#21262D",
}

DEFAULT_VIEW_DAYS = 90  # charts open zoomed to the last 3 months; buttons expand it

RANGESELECTOR = dict(
    buttons=[
        dict(count=1,  label="1M",  step="month", stepmode="backward"),
        dict(count=3,  label="3M",  step="month", stepmode="backward"),
        dict(count=6,  label="6M",  step="month", stepmode="backward"),
        dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
        dict(step="all", label="All"),
    ],
    bgcolor=COLORS["card"],
    activecolor="#30363D",
    font=dict(color=COLORS["text"], size=12),
    bordercolor=COLORS["grid"],
    borderwidth=1,
    x=0, y=1.0, yanchor="bottom",
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'DM Sans', sans-serif", color=COLORS["text"], size=13),
    margin=dict(l=10, r=10, t=40, b=10),
    height=320,
    xaxis=dict(
        showgrid=False, tickfont=dict(size=12, color=COLORS["muted"]),
        rangeselector=RANGESELECTOR,
        type="date",
    ),
    yaxis=dict(showgrid=True, gridcolor=COLORS["grid"], zeroline=False,
               tickfont=dict(size=12, color=COLORS["muted"])),
    legend=dict(orientation="h", x=1, xanchor="right", y=1.0, yanchor="bottom",
                bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    hovermode="x unified",
    hoverlabel=dict(bgcolor=COLORS["card"], bordercolor=COLORS["grid"], font_size=13),
)


def apply_layout(fig, dates: pd.Series, **extra):
    """Shared styling, plus open the chart zoomed to the most recent DEFAULT_VIEW_DAYS."""
    fig.update_layout(**PLOTLY_LAYOUT)
    if extra:
        fig.update_layout(**extra)
    if len(dates):
        end = pd.Timestamp(dates.max()) + pd.Timedelta(days=1)
        start = max(pd.Timestamp(dates.min()), end - pd.Timedelta(days=DEFAULT_VIEW_DAYS))
        fig.update_xaxes(range=[start - pd.Timedelta(days=1), end])


def rolling_7(series: pd.Series) -> pd.Series:
    return series.rolling(7, min_periods=1).mean()


def fig_to_html(fig) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False, "responsive": True})


def chart_steps(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p class='no-data'>No step data found.</p>"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["date"], y=df["steps"], name="Daily",
        marker_color=COLORS["steps"], opacity=0.35,
        hovertemplate="%{y:,} steps<extra>Daily</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=rolling_7(df["steps"]).round(), name="7-day avg",
        mode="lines", line=dict(color=COLORS["steps"], width=3),
        hovertemplate="%{y:,} steps<extra>7-day avg</extra>",
    ))
    fig.add_hline(y=10000, line_dash="dot", line_color=COLORS["muted"],
                  annotation_text="10K goal", annotation_position="top left",
                  annotation_font=dict(size=12, color=COLORS["muted"]))
    apply_layout(fig, df["date"], bargap=0.15)
    fig.update_yaxes(tickformat=",")
    return fig_to_html(fig)


def chart_sleep(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p class='no-data'>No sleep data found.</p>"
    nightly = df.groupby("date", as_index=False)["duration_h"].sum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nightly["date"], y=nightly["duration_h"], name="Nightly",
        mode="markers", marker=dict(color=COLORS["sleep"], size=6, opacity=0.45),
        hovertemplate="%{y:.1f} hrs<extra>Nightly</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=nightly["date"], y=rolling_7(nightly["duration_h"]), name="7-day avg",
        mode="lines", line=dict(color=COLORS["sleep"], width=3),
        hovertemplate="%{y:.1f} hrs<extra>7-day avg</extra>",
    ))
    fig.add_hline(y=8, line_dash="dot", line_color=COLORS["muted"],
                  annotation_text="8 hr goal", annotation_position="top left",
                  annotation_font=dict(size=12, color=COLORS["muted"]))
    apply_layout(fig, nightly["date"])
    fig.update_yaxes(title_text="Hours", range=[0, max(nightly["duration_h"].max() + 1, 9)])
    return fig_to_html(fig)


def chart_heart(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p class='no-data'>No heart rate data found.</p>"
    daily = df.groupby("date")["heart_rate"].agg(["min", "mean"]).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=rolling_7(daily["mean"]), name="Average",
        mode="lines", line=dict(color=COLORS["heart"], width=3),
        hovertemplate="%{y:.0f} bpm<extra>Average (7-day)</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=rolling_7(daily["min"]), name="Resting",
        mode="lines", line=dict(color="#FFB3B3", width=2, dash="dash"),
        hovertemplate="%{y:.0f} bpm<extra>Resting (7-day)</extra>",
    ))
    apply_layout(fig, daily["date"])
    fig.update_yaxes(title_text="BPM")
    return fig_to_html(fig)


def chart_exercise(df: pd.DataFrame) -> str:
    if df.empty or "duration_min" not in df.columns:
        return "<p class='no-data'>No exercise data found.</p>"
    weekly = df.copy()
    weekly["week"] = weekly["date"].dt.to_period("W").dt.start_time
    palette = [COLORS["exercise"], "#00C9A7", "#FF6B6B", "#A78BFA", "#4FC3F7"]
    fig = go.Figure()
    if "exercise_name" in df.columns:
        agg = weekly.groupby(["week", "exercise_name"])["duration_min"].sum().reset_index()
        top_types = agg.groupby("exercise_name")["duration_min"].sum().nlargest(5).index
        for color, name in zip(palette, top_types):
            sub = agg[agg["exercise_name"] == name]
            fig.add_trace(go.Bar(
                x=sub["week"], y=sub["duration_min"].round(), name=name,
                marker_color=color,
                hovertemplate="%{y:.0f} min<extra>" + name + "</extra>",
            ))
    else:
        wk = weekly.groupby("week")["duration_min"].sum().reset_index()
        fig.add_trace(go.Bar(x=wk["week"], y=wk["duration_min"].round(), name="Exercise",
                             marker_color=COLORS["exercise"],
                             hovertemplate="%{y:.0f} min<extra></extra>"))
    apply_layout(fig, weekly["week"], barmode="stack", bargap=0.25)
    fig.update_xaxes(rangeselector=None)
    fig.update_yaxes(title_text="Minutes per week")
    fig.update_layout(legend=dict(x=0, xanchor="left"))
    # weekly bars: show a longer default window than the daily charts
    end = weekly["week"].max() + pd.Timedelta(days=7)
    fig.update_xaxes(range=[max(weekly["week"].min(), end - pd.Timedelta(days=182)) - pd.Timedelta(days=4), end])
    return fig_to_html(fig)


def chart_food(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p class='no-data'>No nutrition data found.</p>"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["date"], y=df["calories"].round(), name="Daily",
        marker_color=COLORS["food"], opacity=0.35,
        hovertemplate="%{y:,} kcal<extra>Daily</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=rolling_7(df["calories"]).round(), name="7-day avg",
        mode="lines", line=dict(color=COLORS["food"], width=3),
        hovertemplate="%{y:,} kcal<extra>7-day avg</extra>",
    ))
    apply_layout(fig, df["date"], bargap=0.15)
    fig.update_yaxes(title_text="kcal", tickformat=",")
    return fig_to_html(fig)


def chart_weight(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p class='no-data'>No weight data found.</p>"
    df = df.copy()
    df["rolling"] = rolling_7(df["weight_kg"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["weight_kg"], name="Weigh-in",
        mode="markers", marker=dict(color=COLORS["weight"], size=7, opacity=0.45),
        hovertemplate="%{y:.1f} kg<extra>Weigh-in</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["rolling"], name="Trend",
        mode="lines", line=dict(color=COLORS["weight"], width=3),
        hovertemplate="%{y:.1f} kg<extra>Trend</extra>",
    ))
    # weight changes slowly, so open on the full history
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_yaxes(title_text="kg")
    return fig_to_html(fig)


# ─────────────────────────────────────────────
# SUMMARY STATS
# ─────────────────────────────────────────────

def stat_card(label: str, value: str, sub: str = "", color: str = "#00C9A7") -> str:
    return f"""
    <div class="stat-card" style="--accent:{color}">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        {"<div class='stat-sub'>" + sub + "</div>" if sub else ""}
    </div>"""


def build_stats(steps_df, sleep_df, heart_df, exercise_df, food_df, weight_df) -> str:
    cards = []

    if not steps_df.empty:
        avg = steps_df["steps"].mean()
        days_goal = (steps_df["steps"] >= 10000).sum()
        cards.append(stat_card("Avg Daily Steps", f"{avg:,.0f}",
                               f"{days_goal} days hit 10K goal", COLORS["steps"]))

    if not sleep_df.empty:
        avg_sleep = sleep_df["duration_h"].mean()
        cards.append(stat_card("Avg Sleep", f"{avg_sleep:.1f} hrs",
                               "per night", COLORS["sleep"]))

    if not heart_df.empty:
        resting = heart_df.groupby("date")["heart_rate"].min().mean()
        cards.append(stat_card("Avg Resting HR", f"{resting:.0f} bpm", "", COLORS["heart"]))

    if not exercise_df.empty and "duration_min" in exercise_df.columns:
        total_min = exercise_df["duration_min"].sum()
        sessions = len(exercise_df)
        cards.append(stat_card("Active Minutes", f"{total_min:,.0f}",
                               f"across {sessions} sessions", COLORS["exercise"]))

    if not food_df.empty:
        avg_cal = food_df["calories"].mean()
        cards.append(stat_card("Avg Daily Calories", f"{avg_cal:,.0f} kcal", "", COLORS["food"]))

    if not weight_df.empty:
        latest = weight_df["weight_kg"].iloc[-1]
        change = weight_df["weight_kg"].iloc[-1] - weight_df["weight_kg"].iloc[0]
        arrow = "↓" if change < 0 else "↑"
        cards.append(stat_card("Current Weight", f"{latest:.1f} kg",
                               f"{arrow} {abs(change):.1f} kg in period", COLORS["weight"]))

    return "".join(cards)


# ─────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Samsung Health Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:      #0D1117;
    --card:    #161B22;
    --border:  #21262D;
    --text:    #E6EDF3;
    --muted:   #9DA7B3;
    --steps:   #00C9A7;
    --sleep:   #845EC2;
    --heart:   #FF6B6B;
    --exercise:#F9A825;
    --food:    #4FC3F7;
    --weight:  #A5D6A7;
    --radius:  12px;
  }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    line-height: 1.6;
    min-height: 100vh;
  }}

  /* Subtle grid background */
  body::before {{
    content: '';
    position: fixed; inset: 0;
    background-image:
      linear-gradient(var(--border) 1px, transparent 1px),
      linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 40px 40px;
    opacity: 0.35;
    pointer-events: none;
    z-index: 0;
  }}

  .wrapper {{ position: relative; z-index: 1; max-width: 1200px; margin: 0 auto; padding: 32px 20px 60px; }}

  /* ── HEADER ── */
  header {{
    display: flex; align-items: flex-end; justify-content: space-between;
    border-bottom: 1px solid var(--border);
    padding-bottom: 20px; margin-bottom: 32px;
    animation: fadeUp .5s ease both;
  }}
  .header-left h1 {{
    font-size: 26px; font-weight: 600; letter-spacing: -.5px;
    background: linear-gradient(120deg, var(--text) 40%, var(--muted));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  .header-left p {{ color: var(--muted); font-size: 13px; margin-top: 2px; }}
  .badge {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 14px;
    font-size: 12px; color: var(--muted); font-family: 'DM Mono', monospace;
  }}

  /* ── STAT CARDS ── */
  .stats-row {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 12px; margin-bottom: 32px;
  }}
  .stat-card {{
    background: var(--card); border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: var(--radius); padding: 16px 18px;
    animation: fadeUp .5s ease both;
    transition: transform .2s, box-shadow .2s;
  }}
  .stat-card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,.4); }}
  .stat-label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .6px; }}
  .stat-value {{ font-size: 22px; font-weight: 600; margin: 4px 0 2px; color: var(--accent); }}
  .stat-sub {{ font-size: 12px; color: var(--muted); }}

  /* ── SECTION HEADERS ── */
  .section-label {{
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1px; color: var(--muted);
    border-left: 3px solid var(--accent-color, var(--steps));
    padding-left: 10px; margin-bottom: 10px;
  }}

  /* ── CHART CARDS ── */
  .charts-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(560px, 1fr));
    gap: 20px;
  }}
  @media (max-width: 600px) {{
    .charts-grid {{ grid-template-columns: 1fr; }}
    .stats-row {{ grid-template-columns: repeat(2, 1fr); }}
  }}
  .chart-card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 20px 16px 12px;
    animation: fadeUp .6s ease both;
    transition: box-shadow .2s;
  }}
  .chart-card:hover {{ box-shadow: 0 6px 20px rgba(0,0,0,.35); }}
  .chart-card h2 {{
    font-size: 15px; font-weight: 500; color: var(--text);
    display: flex; align-items: center; gap: 8px; margin-bottom: 12px;
  }}
  .chart-card h2 .dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--dot-color, var(--steps));
    flex-shrink: 0;
  }}

  .no-data {{ color: var(--muted); font-size: 13px; padding: 40px 0; text-align: center; }}

  /* ── FOOTER ── */
  footer {{
    margin-top: 48px; text-align: center;
    font-size: 12px; color: var(--muted);
    border-top: 1px solid var(--border); padding-top: 20px;
  }}

  @keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(16px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}
  .chart-card:nth-child(1) {{ animation-delay: .05s }}
  .chart-card:nth-child(2) {{ animation-delay: .10s }}
  .chart-card:nth-child(3) {{ animation-delay: .15s }}
  .chart-card:nth-child(4) {{ animation-delay: .20s }}
  .chart-card:nth-child(5) {{ animation-delay: .25s }}
  .chart-card:nth-child(6) {{ animation-delay: .30s }}
</style>
</head>
<body>
<div class="wrapper">

  <header>
    <div class="header-left">
      <h1>Health Dashboard</h1>
      <p>Samsung Health · {date_range} · Generated {generated}</p>
    </div>
    <div class="badge">⬡ {data_sources} data sources</div>
  </header>

  <div class="stats-row">
    {stats_html}
  </div>

  <div class="charts-grid">

    {steps_section}
    {sleep_section}
    {heart_section}
    {exercise_section}
    {food_section}
    {weight_section}

  </div>

  <footer>
    Built with Pandas + Plotly · Samsung Health export · {generated}
  </footer>

</div>
</body>
</html>
"""


def section(title: str, dot_color: str, chart_html: str) -> str:
    return f"""
    <div class="chart-card">
      <h2><span class="dot" style="background:{dot_color}"></span>{title}</h2>
      {chart_html}
    </div>"""


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Samsung Health Dashboard Generator")
    parser.add_argument("--data-dir", default=str(SCRIPT_DIR / "sample_data"), help="Path to Samsung Health export folder")
    parser.add_argument("--days", type=int, default=DAYS_DEFAULT, help="Number of recent days to include")
    parser.add_argument("--diagnose", action="store_true", help="Print raw column samples to debug date parsing")
    parser.add_argument("--output", default=str(SCRIPT_DIR / OUTPUT_FILE), help="Output HTML file path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(1)

    print(f"\n📂 Scanning: {data_dir}")
    print(f"📅 Period: last {args.days} days\n")

    def diagnose_csv(label, path):
        """Print raw column names and a few sample values to debug parsing."""
        df = read_csv_skip_metadata(path)
        if df.empty:
            print(f"    [diag] {label}: DataFrame is empty after read")
            return
        print(f"    [diag] {label} columns: {list(df.columns)}")
        # Find any column that looks date-like and show raw samples
        for col in df.columns:
            suffix = col.split(".")[-1].lower()
            if any(k in suffix for k in ["time", "date"]):
                samples = df[col].dropna().head(3).tolist()
                print(f"    [diag]   {col!r} samples: {samples}")
                break

    def try_load(label, patterns, loader):
        path = find_csv(data_dir, patterns)
        if path:
            print(f"  ✓ {label}: {path.name}")
            if args.diagnose:
                diagnose_csv(label, path)
            df = loader(path, args.days)
            if df.empty:
                print(f"    ⚠ {label}: parsed but empty after filtering")
            return df
        else:
            print(f"  ✗ {label}: no matching file found")
            return pd.DataFrame()

    steps_df    = try_load("Steps",      FILE_PATTERNS["steps"],    load_steps)
    sleep_df    = try_load("Sleep",      FILE_PATTERNS["sleep"],    load_sleep)
    heart_df    = try_load("Heart Rate", FILE_PATTERNS["heart"],    load_heart_rate)
    exercise_df = try_load("Exercise",   FILE_PATTERNS["exercise"], load_exercise)
    food_df     = try_load("Food",       FILE_PATTERNS["food"],     load_food)
    weight_df   = try_load("Weight",     FILE_PATTERNS["weight"],   load_weight)

    sources = sum(1 for d in [steps_df, sleep_df, heart_df, exercise_df, food_df, weight_df] if not d.empty)
    print(f"\n📊 Building dashboard from {sources} data sources...")

    stats_html = build_stats(steps_df, sleep_df, heart_df, exercise_df, food_df, weight_df)

    all_dates = pd.concat([d["date"] for d in
                           [steps_df, sleep_df, heart_df, exercise_df, food_df, weight_df]
                           if not d.empty and "date" in d.columns])
    date_range = (f"{all_dates.min():%b %d, %Y} – {all_dates.max():%b %d, %Y}"
                  if len(all_dates) else "No data")

    html = HTML_TEMPLATE.format(
        date_range=date_range,
        generated=datetime.now().strftime("%b %d, %Y"),
        data_sources=sources,
        stats_html=stats_html,
        steps_section=section("Steps & Activity",    COLORS["steps"],    chart_steps(steps_df)),
        sleep_section=section("Sleep Duration",       COLORS["sleep"],    chart_sleep(sleep_df)),
        heart_section=section("Heart Rate",           COLORS["heart"],    chart_heart(heart_df)),
        exercise_section=section("Weekly Exercise",   COLORS["exercise"], chart_exercise(exercise_df)),
        food_section=section("Daily Calories",        COLORS["food"],     chart_food(food_df)),
        weight_section=section("Weight Tracking",     COLORS["weight"],   chart_weight(weight_df)),
    )

    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    print(f"\n✅ Dashboard saved → {out.resolve()}")
    print("   Open it in any browser — no server needed.\n")


if __name__ == "__main__":
    main()