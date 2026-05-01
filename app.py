"""Streamlit dashboard for the MU5735 (B-1791) FDR data.

Run:
    uv run streamlit run app.py

If the Parquet cache does not yet exist this will run preprocess.py first.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
TABLE_PARQUET = CACHE_DIR / "table_resolution.parquet"
EXACT_PARQUET = CACHE_DIR / "exact_sample.parquet"
META_JSON = CACHE_DIR / "parameter_meta.json"

# ---------------------------------------------------------------------------
# Curated "story" parameters. These are the ones a non-expert reader benefits
# from seeing first, grouped by what they tell you about the flight.
# ---------------------------------------------------------------------------
PRIMARY_FLIGHT = {
    "Altitude Press": "Pressure altitude (ft)",
    "Airspeed Comp": "Computed airspeed (kt)",
    "Ground Spd": "Ground speed (kt)",
    "Pitch Angle": "Pitch (deg, + = nose up)",
    "Roll Angle": "Roll (deg, + = right wing down)",
    "Heading": "Magnetic heading (deg)",
}

ENGINE_PANEL = {
    "Eng1 N1": "Eng-1 fan speed (% RPM)",
    "Eng2 N1": "Eng-2 fan speed (% RPM)",
    "Eng1 EGT": "Eng-1 exhaust gas temp (degC)",
    "Eng2 EGT": "Eng-2 exhaust gas temp (degC)",
    "Eng1 Fuel Flow": "Eng-1 fuel flow (pph)",
    "Eng2 Fuel Flow": "Eng-2 fuel flow (pph)",
}

CONTROLS_PANEL = {
    "Elevator-L": "Elevator L (deg)",
    "Elevator-R": "Elevator R (deg)",
    "Aileron-L": "Aileron L (deg)",
    "Aileron-R": "Aileron R (deg)",
    "Rudder": "Rudder (deg)",
    "Ctrl Col Pos-L": "Control column L (deg)",
    "Ctrl Whl Pos-L": "Control wheel L (deg)",
    "Rudder Ped Pos": "Rudder pedal (deg)",
}

DISCRETE_STATES = [
    "AP-1 Warn",
    "AP-2 Warn",
    "CMD A - FCC",
    "FAC Engage - FCC",
    "VNAV PATH Engaged - FCC",
    "VNAV SPD Engaged - FCC",
    "LNAV Engaged - FCC",
    "ALT HOLD Engaged - FCC",
    "GS Engaged - FCC",
    "LOC Engaged - FCC",
    "TOGA Engaged - FCC",
    "Eng1 Fire",
    "Eng2 Fire",
    "Eng1 Cutoff SW",
    "Eng2 Cutoff SW",
    "APU On",
    "AT FMC SPD Engaged",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _ensure_cache() -> None:
    if TABLE_PARQUET.exists() and META_JSON.exists():
        return
    st.warning("Parquet cache missing — running preprocess.py (one-time, ~10s)...")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "preprocess.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if proc.returncode != 0:
        st.error("Preprocessing failed.")
        st.code(proc.stdout + "\n" + proc.stderr)
        st.stop()


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(TABLE_PARQUET)
    meta = json.loads(META_JSON.read_text())
    return df, meta


@st.cache_data(show_spinner=False)
def load_exact() -> pd.DataFrame | None:
    if EXACT_PARQUET.exists():
        return pd.read_parquet(EXACT_PARQUET)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _series(df: pd.DataFrame, col: str, ffill: bool = True) -> pd.Series:
    """Return the column with NaNs filtered out (or forward-filled)."""
    if col not in df.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df[col], errors="coerce")
    if ffill:
        return s.ffill()
    return s


def detect_upset_time(df: pd.DataFrame) -> float | None:
    """Heuristic: the upset is the first time |Roll| exceeds 15° in a sustained way."""
    roll = _series(df, "Roll Angle", ffill=True)
    over = roll.abs() > 15
    # Use the FIRST sustained crossing (>=4 consecutive samples = 0.25 s at 16 Hz).
    run = over.rolling(4).sum() == 4
    idx = run.idxmax() if run.any() else None
    if idx is None:
        return None
    return float(df.loc[idx, "Time"])


def fmt_time(seconds_from_midnight: float) -> str:
    """Convert seconds-since-midnight to HH:MM:SS UTC."""
    s = seconds_from_midnight % 86400
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:05.2f}"


def overview_chart(df: pd.DataFrame, upset_t: float | None) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            "Altitude (ft)",
            "Computed airspeed (kt)",
            "Pitch (deg)",
            "Roll (deg)",
        ),
    )
    t = df["Time"]
    fig.add_scatter(x=t, y=_series(df, "Altitude Press"), name="Altitude", row=1, col=1, line=dict(color="#1f77b4"))
    fig.add_scatter(x=t, y=_series(df, "Airspeed Comp"), name="Airspeed", row=2, col=1, line=dict(color="#d62728"))
    fig.add_scatter(x=t, y=_series(df, "Pitch Angle"), name="Pitch", row=3, col=1, line=dict(color="#2ca02c"))
    fig.add_scatter(x=t, y=_series(df, "Roll Angle"), name="Roll", row=4, col=1, line=dict(color="#ff7f0e"))

    if upset_t is not None:
        for r in (1, 2, 3, 4):
            fig.add_vline(x=upset_t, line=dict(color="red", width=1, dash="dot"), row=r, col=1)
        fig.add_annotation(
            x=upset_t,
            y=1,
            yref="paper",
            text=f"Upset @ T={upset_t:.1f}s",
            showarrow=False,
            font=dict(color="red"),
            bgcolor="rgba(255,255,255,0.6)",
        )

    fig.update_layout(
        height=720,
        showlegend=False,
        margin=dict(l=60, r=30, t=40, b=30),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="FDR time (s, since-midnight)", row=4, col=1)
    return fig


def multi_panel_chart(
    df: pd.DataFrame,
    panels: dict[str, list[str]],
    upset_t: float | None,
    height_per: int = 180,
) -> go.Figure:
    rows = len(panels)
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=list(panels.keys()),
    )
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf", "#8c564b"]
    t = df["Time"]
    for i, (_title, cols) in enumerate(panels.items(), start=1):
        for j, c in enumerate(cols):
            if c in df.columns:
                fig.add_scatter(
                    x=t,
                    y=_series(df, c),
                    name=c,
                    row=i,
                    col=1,
                    line=dict(color=palette[j % len(palette)]),
                    legendgroup=f"row{i}",
                )
        if upset_t is not None:
            fig.add_vline(x=upset_t, line=dict(color="red", width=1, dash="dot"), row=i, col=1)

    fig.update_layout(
        height=max(height_per * rows, 320),
        margin=dict(l=60, r=30, t=40, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.05),
    )
    fig.update_xaxes(title_text="FDR time (s)", row=rows, col=1)
    return fig


def discrete_chart(df: pd.DataFrame, cols: list[str], meta: dict) -> go.Figure:
    """Render boolean / discrete signals as a stacked timeline."""
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = _series(df, c, ffill=True).fillna(0)
        # Map to 0/1 step. Many discrete signals are already 0/1.
        if s.max() > 1 or s.min() < 0:
            # Normalize to a 0..1 range for display
            s = (s - s.min()) / (max(s.max() - s.min(), 1e-9))
        rows.append((c, s))

    fig = go.Figure()
    if not rows:
        fig.add_annotation(text="No discrete columns selected.", showarrow=False)
        return fig

    t = df["Time"]
    for i, (name, s) in enumerate(rows):
        # Draw as a step-line offset by row index
        offset = i
        y = s + offset
        info = meta["parameters"].get(name, {})
        decoder = info.get("decoder")
        # Build a "ON"/"OFF" hover label using the decoder if available
        hover = []
        for v in s.values:
            if decoder:
                # decoder keys are floats; round s to 0/1 to read the label
                key = 1.0 if v >= 0.5 else 0.0
                hover.append(decoder.get(key, decoder.get(int(key), "")))
            else:
                hover.append("ON" if v >= 0.5 else "OFF")
        fig.add_scatter(
            x=t,
            y=y,
            mode="lines",
            line=dict(shape="hv", width=1.5),
            name=name,
            hovertext=hover,
            hovertemplate=f"{name}<br>%{{hovertext}}<br>T=%{{x:.2f}}s<extra></extra>",
        )

    fig.update_layout(
        height=max(120 + 60 * len(rows), 320),
        margin=dict(l=60, r=30, t=30, b=30),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(rows))),
            ticktext=[r[0] for r in rows],
            range=[-0.5, len(rows) + 0.5],
            zeroline=False,
        ),
        xaxis_title="FDR time (s)",
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def trajectory_xy(df: pd.DataFrame) -> go.Figure:
    """Pseudo-trajectory: integrate ground-speed × heading to get a relative XY path.

    The FDR has no GPS lat/lon in this readout, so we approximate the ground track
    by integrating (ground speed, heading) starting from (0,0). This is good for
    showing the shape of the path, not absolute position.
    """
    gs = _series(df, "Ground Spd", ffill=True).fillna(0)
    hdg = _series(df, "Heading", ffill=True).fillna(0)
    t = df["Time"].values
    dt = np.diff(t, prepend=t[0])
    # heading: 0 = north, increasing clockwise. Convert to math angle.
    rad = np.deg2rad(90 - hdg.values)
    # ground speed in kt -> nm/s = kt/3600. We just want shape, so units don't matter.
    vx = gs.values / 3600.0 * np.cos(rad)
    vy = gs.values / 3600.0 * np.sin(rad)
    x = np.cumsum(vx * dt)
    y = np.cumsum(vy * dt)

    alt = _series(df, "Altitude Press", ffill=True)
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="rgba(31,119,180,0.7)", width=2),
            customdata=np.column_stack([t, alt.values, hdg.values]),
            hovertemplate=(
                "T=%{customdata[0]:.1f}s<br>"
                "Alt=%{customdata[1]:.0f} ft<br>"
                "Hdg=%{customdata[2]:.1f}°<extra></extra>"
            ),
            name="Track",
        )
    )
    # Mark start and end
    fig.add_scatter(
        x=[x[0]], y=[y[0]], mode="markers+text", text=["start"], textposition="top right",
        marker=dict(color="green", size=10), name="start",
    )
    fig.add_scatter(
        x=[x[-1]], y=[y[-1]], mode="markers+text", text=["impact"], textposition="top right",
        marker=dict(color="red", size=12, symbol="x"), name="end",
    )
    fig.update_layout(
        height=520,
        margin=dict(l=60, r=30, t=30, b=30),
        xaxis=dict(title="east-ish (nm, relative)", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="north-ish (nm, relative)"),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# 3D replay
# ---------------------------------------------------------------------------
def _body_axes(heading_deg: float, pitch_deg: float, roll_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (nose, right_wing, body_up) unit vectors in world frame.

    World frame: x = east, y = north, z = up.
    Conventions:
      - heading: 0 = north, clockwise positive (standard compass)
      - pitch: positive = nose up
      - roll: positive = right wing down
    """
    H = np.deg2rad(heading_deg)
    P = np.deg2rad(pitch_deg)
    R = np.deg2rad(roll_deg)
    sH, cH = np.sin(H), np.cos(H)
    sP, cP = np.sin(P), np.cos(P)
    sR, cR = np.sin(R), np.cos(R)

    # Nose direction: compass-style azimuth + elevation by pitch
    nose = np.array([sH * cP, cH * cP, sP])
    # "pre-roll" body up: body up before any rolling (after yaw + pitch)
    pre_up = np.array([-sH * sP, -cH * sP, cP])
    # Right wing before roll: nose x pre_up (right-handed)
    pre_right = np.cross(nose, pre_up)

    # Apply roll about the nose axis
    right_wing = cR * pre_right - sR * pre_up
    body_up = cR * pre_up + sR * pre_right
    return nose, right_wing, body_up


def _aircraft_lines(
    pos: np.ndarray,
    heading_deg: float,
    pitch_deg: float,
    roll_deg: float,
    scale: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xs, ys, zs) for a Scatter3d line trace shaping a stylised aircraft.

    The aircraft is drawn as four line segments (fuselage / wings / vertical tail
    / horizontal stab). Segments are separated by NaN so a single Scatter3d trace
    renders all four as disjoint lines.
    """
    nose, right_wing, body_up = _body_axes(heading_deg, pitch_deg, roll_deg)

    def w(a: float, b: float, c: float) -> np.ndarray:
        return pos + scale * (a * nose + b * right_wing + c * body_up)

    # Body-frame coordinates (a along nose, b along right wing, c along up)
    fuse_back, fuse_front = -3.0, 5.0
    wing = 4.0
    vtail_h = 1.5
    hstab = 1.5

    pts: list[np.ndarray] = []
    nan = np.array([np.nan, np.nan, np.nan])
    # Fuselage
    pts.extend([w(fuse_back, 0, 0), w(fuse_front, 0, 0), nan])
    # Wings
    pts.extend([w(0, -wing, 0), w(0, wing, 0), nan])
    # Vertical tail (rises from rear of fuselage)
    pts.extend([w(fuse_back + 0.2, 0, 0), w(fuse_back, 0, vtail_h), nan])
    # Horizontal stabiliser
    pts.extend([w(fuse_back, -hstab, 0), w(fuse_back, hstab, 0), nan])

    arr = np.array(pts, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _resample_for_replay(df: pd.DataFrame, t_window_s: float, hz: float) -> dict[str, np.ndarray]:
    """Resample the relevant flight columns to a uniform `hz` grid over the
    last `t_window_s` seconds of the recording.
    """
    cols = ["Altitude Press", "Airspeed Comp", "Ground Spd", "Heading", "Pitch Angle", "Roll Angle"]
    end = float(df["Time"].max())
    start = end - t_window_s
    sub = df[df["Time"].between(start, end)].copy()
    sub = sub.sort_values("Time").reset_index(drop=True)
    # ffill / bfill so every needed column has a value at every original sample
    for c in cols:
        sub[c] = sub[c].ffill().bfill()
    targets = np.arange(start, end + 1e-9, 1.0 / hz)
    out: dict[str, np.ndarray] = {"t": targets}
    src_t = sub["Time"].values
    for c in cols:
        out[c] = np.interp(targets, src_t, sub[c].values)
    return out


def replay_3d_chart(
    df: pd.DataFrame,
    t_window_s: float = 60.0,
    hz: float = 5.0,
    glyph_scale: float = 0.10,
) -> go.Figure:
    """Build a 3D replay figure with a trajectory ribbon, an animated aircraft
    glyph, and a scrubbable time slider.

    Spatial units: nautical miles (path integrated from heading × ground-speed,
    altitude converted ft -> nm so axis aspect is meaningful).
    """
    r = _resample_for_replay(df, t_window_s, hz)
    t = r["t"]
    n = len(t)

    # Integrate ground track in nautical miles
    dt = np.diff(t, prepend=t[0])
    rad = np.deg2rad(90.0 - r["Heading"])  # compass -> math angle
    speed_nmps = r["Ground Spd"] / 3600.0
    vx = speed_nmps * np.cos(rad)
    vy = speed_nmps * np.sin(rad)
    x = np.cumsum(vx * dt)
    y = np.cumsum(vy * dt)
    z = r["Altitude Press"] / 6076.12  # ft -> nm

    # Trajectory ribbon (constant trace, index 0)
    traj = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line=dict(color=r["Altitude Press"], colorscale="Viridis", width=4, colorbar=dict(title="Alt (ft)")),
        name="Path",
        hoverinfo="skip",
    )

    # Aircraft glyph (animated, trace index 1)
    init_xs, init_ys, init_zs = _aircraft_lines(
        np.array([x[0], y[0], z[0]]),
        r["Heading"][0],
        r["Pitch Angle"][0],
        r["Roll Angle"][0],
        scale=glyph_scale,
    )
    aircraft = go.Scatter3d(
        x=init_xs,
        y=init_ys,
        z=init_zs,
        mode="lines",
        line=dict(color="#d62728", width=8),
        name="Aircraft",
        hoverinfo="skip",
    )

    # Current-position dot (animated, trace index 2)
    cur_marker = go.Scatter3d(
        x=[x[0]],
        y=[y[0]],
        z=[z[0]],
        mode="markers",
        marker=dict(size=4, color="#d62728"),
        name="Now",
        hoverinfo="skip",
    )

    # Frames
    frames: list[go.Frame] = []
    end_time = float(df["Time"].max())
    for i in range(n):
        ax_x, ax_y, ax_z = _aircraft_lines(
            np.array([x[i], y[i], z[i]]),
            r["Heading"][i],
            r["Pitch Angle"][i],
            r["Roll Angle"][i],
            scale=glyph_scale,
        )
        rel = t[i] - end_time  # seconds before impact (negative)
        title = (
            f"T-{abs(rel):05.2f}s &nbsp;|&nbsp; "
            f"Alt {r['Altitude Press'][i]:>6,.0f} ft &nbsp;|&nbsp; "
            f"AS {r['Airspeed Comp'][i]:>5.0f} kt &nbsp;|&nbsp; "
            f"Pitch {r['Pitch Angle'][i]:>+6.1f}° &nbsp;|&nbsp; "
            f"Roll {r['Roll Angle'][i]:>+7.1f}°"
        )
        frames.append(
            go.Frame(
                name=str(i),
                data=[
                    go.Scatter3d(x=ax_x, y=ax_y, z=ax_z),
                    go.Scatter3d(x=[x[i]], y=[y[i]], z=[z[i]]),
                ],
                traces=[1, 2],
                layout=go.Layout(title=dict(text=title)),
            )
        )

    # Slider
    slider_steps = []
    for i in range(n):
        rel = t[i] - end_time
        slider_steps.append(
            dict(
                method="animate",
                label=f"{rel:+.1f}s",
                args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
            )
        )

    initial_title = (
        f"T-{abs(t[0]-end_time):05.2f}s &nbsp;|&nbsp; "
        f"Alt {r['Altitude Press'][0]:>6,.0f} ft &nbsp;|&nbsp; "
        f"AS {r['Airspeed Comp'][0]:>5.0f} kt &nbsp;|&nbsp; "
        f"Pitch {r['Pitch Angle'][0]:>+6.1f}° &nbsp;|&nbsp; "
        f"Roll {r['Roll Angle'][0]:>+7.1f}°"
    )

    fig = go.Figure(
        data=[traj, aircraft, cur_marker],
        frames=frames,
    )
    fig.update_layout(
        title=dict(text=initial_title, x=0.02, xanchor="left", font=dict(family="monospace", size=14)),
        height=720,
        margin=dict(l=0, r=0, t=50, b=10),
        scene=dict(
            xaxis=dict(title="east (nm, rel)"),
            yaxis=dict(title="north (nm, rel)"),
            zaxis=dict(title="alt (nm)"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.4, z=0.9)),
        ),
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02,
                y=-0.02,
                xanchor="left",
                yanchor="top",
                pad=dict(t=10, r=10),
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True, transition=dict(duration=0))],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=slider_steps,
                x=0.10,
                y=-0.02,
                len=0.85,
                xanchor="left",
                yanchor="top",
                pad=dict(t=10, b=10),
                currentvalue=dict(prefix="t = ", font=dict(family="monospace")),
            )
        ],
    )
    return fig


# ---------------------------------------------------------------------------
# Forensic timeline
# ---------------------------------------------------------------------------
# Events are derived from manual inspection of TableResolution.csv and
# ExactSample.csv. Timestamps are in seconds-relative-to-end-of-recording
# (T-0 is the last FDR sample, treated as impact). FDR clock equivalents:
#   -19.4s = 288958.59  -17.6s = 288960.31  -17.1s = 288960.43
#   -17.1s = 288960.69  -17.0s = 288960.81  -16.7s = 288961.19
#   -16.5s = 288961.44  -13.5s = 288964.50  -9.0s  = 288969.0
#   -3.0s  = 288974.94  -0.5s  = 288977.4   -0.1s  = 288977.43
FORENSIC_EVENTS: list[dict] = [
    {
        "rel": -19.35,
        "label": "Both fuel cutoff switches → CUTOFF",
        "category": "Cockpit input",
        "detail": (
            "Eng1 Cutoff SW and Eng2 Cutoff SW both transition from RUN to CUTOFF "
            "in the same FDR frame. These are physical switches under each "
            "throttle lever; moving them in normal cruise has no procedural reason."
        ),
    },
    {
        "rel": -17.6,
        "label": "Control wheel slammed left (≈ -28°)",
        "category": "Cockpit input",
        "detail": (
            "Control wheel jumps from neutral cruise (+0.35°) to -28° in a single "
            "sample, then continues to -50°…-100° (full deflection) and is held "
            "there for several seconds."
        ),
    },
    {
        "rel": -17.5,
        "label": "AP-1 and AP-2 disconnect warnings",
        "category": "Auto reaction",
        "detail": (
            "Both autopilot warning channels assert (Off → On). This is the "
            "expected response when manual control input forces exceed the "
            "autopilot override threshold."
        ),
    },
    {
        "rel": -17.3,
        "label": "Eng-2 fuel flow collapses",
        "category": "Result",
        "detail": (
            "FF2 drops from 2576 pph to 144 pph (~95% reduction) within ~0.3 s of "
            "the cutoff switch operation. Throttle Resolver Angle (Eng2 TRA) "
            "stays at cruise position 59.2° — i.e. the throttle lever did not "
            "move; only the fuel valve closed."
        ),
    },
    {
        "rel": -17.2,
        "label": "Autopilot disengages (CMD A → Not SELECTED)",
        "category": "Auto reaction",
        "detail": (
            "AP Off transitions On → F/D Only. CMD A (master autopilot mode) "
            "deselects. Aircraft is now under manual control."
        ),
    },
    {
        "rel": -16.8,
        "label": "Eng-1 fuel flow collapses",
        "category": "Result",
        "detail": (
            "FF1 drops from ~2600 pph to 304 pph. Eng1 TRA also stays at 59.6° "
            "— same pattern: fuel valve closed, throttle position unchanged."
        ),
    },
    {
        "rel": -16.5,
        "label": "Aircraft begins rolling left",
        "category": "Result",
        "detail": (
            "Roll passes -6°, then -29°, -44°, -60° in successive samples. "
            "Heading drifts from 102° toward 96° as the rolling aircraft yaws."
        ),
    },
    {
        "rel": -13.5,
        "label": "Aircraft rolls past inverted (≤ -180°)",
        "category": "Result",
        "detail": (
            "Roll reaches ±177° (essentially upside-down) ≈ 13 s before impact. "
            "Pitch begins to drop — gravity is now pulling the nose 'down' "
            "relative to the aircraft (i.e. toward earth)."
        ),
    },
    {
        "rel": -9.0,
        "label": "Recovery attempt — column pulled aft",
        "category": "Cockpit input",
        "detail": (
            "Control column moves to -6° to -8° (pulled aft for the first time), "
            "elevator deflects to -10° to -11°. Roll briefly returns through "
            "level (+115° → +60° → 0°). Too low / too steep / no thrust to "
            "recover."
        ),
    },
    {
        "rel": -3.0,
        "label": "Both engines effectively shut down",
        "category": "Result",
        "detail": (
            "Eng1 and Eng2 N1 ≈ 26% (windmilling, sub-idle). FF1 → 0 pph. "
            "Engines are providing no thrust."
        ),
    },
    {
        "rel": -0.6,
        "label": "Air-Ground discrete reads 'Gnd'",
        "category": "Impact",
        "detail": (
            "The weight-on-wheels / air-ground discrete (driven by gear strut "
            "compression on the 737-NG) flips Air → Gnd. Effective ground impact."
        ),
    },
    {
        "rel": -0.1,
        "label": "Engine fire indicators assert",
        "category": "Impact",
        "detail": (
            "Eng1 Fire and Eng2 Fire both go NO FIRE → FIRE in the very last "
            "samples. Vertical G spikes to +4.1 / -3.4. Aircraft is breaking up."
        ),
    },
]

_CAT_COLOR = {
    "Cockpit input": "#d62728",
    "Auto reaction": "#ff7f0e",
    "Result": "#1f77b4",
    "Impact": "#000000",
}


def forensic_timeline_chart() -> go.Figure:
    """A swimlane timeline: each category occupies its own horizontal row, so
    events that happen at the same instant stack visibly instead of overlapping.

    There are no on-chart labels — labels go in the chronological table below
    the chart, where they have room to be readable.
    """
    # Lane order from top to bottom of the chart
    lanes = ["Cockpit input", "Auto reaction", "Result", "Impact"]
    y_for: dict[str, int] = {cat: (len(lanes) - 1 - i) for i, cat in enumerate(lanes)}

    fig = go.Figure()

    # Subtle horizontal guide for each lane
    for cat in lanes:
        fig.add_shape(
            type="line",
            x0=-20.5, x1=0.5,
            y0=y_for[cat], y1=y_for[cat],
            line=dict(color="#ddd", width=1),
            layer="below",
        )

    # One scatter trace per category so the legend entries line up
    for cat in lanes:
        events = [e for e in FORENSIC_EVENTS if e["category"] == cat]
        if not events:
            continue
        fig.add_scatter(
            x=[e["rel"] for e in events],
            y=[y_for[cat]] * len(events),
            mode="markers",
            marker=dict(
                size=18,
                color=_CAT_COLOR[cat],
                line=dict(color="white", width=2),
                symbol="circle",
            ),
            name=cat,
            customdata=[[e["label"], e["detail"]] for e in events],
            hovertemplate="<b>T%{x:+.2f}s — " + cat + "</b><br>%{customdata[0]}<br><br>%{customdata[1]}<extra></extra>",
        )

    # Mark T-0 (impact) with a vertical line
    fig.add_vline(
        x=0,
        line=dict(color="#000", width=1, dash="dot"),
        annotation_text="impact",
        annotation_position="top",
    )

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=50),
        xaxis=dict(
            title="Seconds relative to impact",
            range=[-20.5, 1.0],
            tickmode="array",
            tickvals=list(range(-20, 1, 2)),
            zeroline=False,
            showgrid=True,
            gridcolor="#f0f0f0",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=[y_for[c] for c in lanes],
            ticktext=lanes,
            range=[-0.6, len(lanes) - 0.4],
            showgrid=False,
            zeroline=False,
        ),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    return fig


def forensic_event_table() -> pd.DataFrame:
    """Build a chronological table of the forensic events for display."""
    rows = []
    for e in FORENSIC_EVENTS:
        rows.append(
            {
                "T (rel impact)": f"T{e['rel']:+.2f} s",
                "Category": e["category"],
                "Event": e["label"],
                "Underlying signal evidence": e["detail"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="MU5735 FDR Explorer",
        layout="wide",
        page_icon="🛬",
    )
    _ensure_cache()
    df, meta = load_data()
    upset_t = detect_upset_time(df)

    file_meta = meta["file_metadata"]
    duration = meta["duration_s"]
    sample_hz = meta["sample_hz"]

    # ------------------------------ Sidebar -------------------------------
    with st.sidebar:
        st.title("MU5735 FDR Explorer")
        st.markdown(
            f"**Operator:** {file_meta.get('Operator', '—')}  \n"
            f"**Aircraft:** {file_meta.get('Vehicle ID/Registration', '—')}  \n"
            f"**Type:** {file_meta.get('Source of Data', '—')}  \n"
            f"**Date:** {file_meta.get('Date of Event', '—')}  \n"
            f"**Location:** {file_meta.get('Location', '—')}  \n"
            f"**NTSB ID:** DCA22WA102"
        )
        st.divider()
        st.markdown(
            f"- **Window:** {duration:.0f} s ({duration/60:.1f} min)\n"
            f"- **Rate:** {sample_hz:g} Hz\n"
            f"- **Rows:** {len(df):,}\n"
            f"- **Parameters:** {len(meta['parameters'])}\n"
            f"- **Sentinel rows dropped:** {meta['sentinels_dropped']:,}"
        )
        st.divider()
        st.caption(
            "Time is the FDR's seconds-since-midnight clock; this readout was "
            "trimmed by the NTSB to about a 13-minute window centered on the "
            "event. The dashboard's *t_rel* axis on some panels shows time "
            "from the start of this window."
        )

        # Time-range slider — applies to all tabs except the overview/event-zoom
        st.divider()
        st.subheader("Time filter")
        t_min, t_max = float(df["Time"].min()), float(df["Time"].max())
        t_range = st.slider(
            "FDR time range (s)",
            min_value=t_min,
            max_value=t_max,
            value=(t_min, t_max),
            step=1.0,
            help="Used by the Engines / Controls / Discrete / Browser / Compare tabs.",
        )

    df_filt = df[(df["Time"] >= t_range[0]) & (df["Time"] <= t_range[1])].reset_index(drop=True)

    # --------------------------- Top KPI strip ----------------------------
    col1, col2, col3, col4 = st.columns(4)
    alt = _series(df, "Altitude Press")
    col1.metric("Cruise alt (max)", f"{alt.max():,.0f} ft")
    col2.metric("Min alt seen", f"{alt.min():,.0f} ft")
    col3.metric(
        "Pitch range",
        f"{_series(df, 'Pitch Angle').min():.1f}° / {_series(df, 'Pitch Angle').max():.1f}°",
    )
    col4.metric(
        "Roll range",
        f"{_series(df, 'Roll Angle').min():.1f}° / {_series(df, 'Roll Angle').max():.1f}°",
    )

    if upset_t is not None:
        secs_to_end = df["Time"].max() - upset_t
        st.info(
            f"**Heuristic upset onset:** T = {upset_t:.1f} s  "
            f"({fmt_time(upset_t)} on the FDR clock) — "
            f"first sustained roll > 15°, **{secs_to_end:.1f} s before the end of the recording**."
        )

    # ------------------------------- Tabs ---------------------------------
    (
        tab_story,
        tab_what,
        tab_replay,
        tab_event,
        tab_engines,
        tab_controls,
        tab_discrete,
        tab_track,
        tab_browser,
        tab_compare,
    ) = st.tabs(
        [
            "Story",
            "What happened?",
            "3D Replay",
            "Event zoom (last 30 s)",
            "Engines",
            "Flight controls",
            "Discrete states",
            "Ground track",
            "Parameter browser",
            "Compare",
        ]
    )

    # ------------------------------- Story tab ----------------------------
    with tab_story:
        st.markdown(
            """
            ### Reading the data

            This is the FDR readout for **China Eastern flight MU5735** — a
            Boeing 737-800 (registration **B-1791**) that crashed near Wuzhou,
            China on **2022-03-21**. The recording covers about **13 minutes**
            ending at impact.

            The four panels below show the canonical "what happened" view:
            altitude, airspeed, pitch and roll. For nearly the entire window
            the aircraft is in a steady cruise around **FL290** at about
            **265 KIAS**, heading roughly east. Then, in the last ~16 seconds,
            it banks hard, pitches over and dives. The red dotted line marks
            the heuristic upset onset (first sustained roll > 15°).
            """
        )
        st.plotly_chart(overview_chart(df, upset_t), width="stretch")

    # ------------------------ "What happened?" tab ------------------------
    with tab_what:
        st.markdown(
            """
            ### What does the data tell us?

            > **Caveat.** The cockpit voice recorder is **not** in this FOIA release.
            > This analysis is built only from the FDR signals. The data shows what
            > happened *mechanically* — the order of switch flips, control
            > deflections and surface responses. It cannot say *who* was at the
            > controls, *why*, or whether the crew was conscious. The official
            > Chinese investigation (CAAC) has not published a final cause as of
            > the FOIA's release; this is one independent reading.
            """
        )

        st.subheader("The forensic timeline")
        st.markdown(
            """
            Reconstructed by stepping through the high-rate (16 Hz) FDR samples
            in the final 20 seconds. Each event is a real, timestamped state
            change in the recorded data. Lanes group events by category — three
            cockpit inputs (red, top), three autopilot/automation reactions
            (orange), several aerodynamic results (blue), and impact (black).
            **Hover any dot for the underlying signal evidence.**
            """
        )
        st.plotly_chart(forensic_timeline_chart(), width="stretch")
        st.markdown("##### Chronological detail")
        st.dataframe(
            forensic_event_table(),
            width="stretch",
            hide_index=True,
            column_config={
                "T (rel impact)": st.column_config.Column(width="small"),
                "Category": st.column_config.Column(width="small"),
                "Event": st.column_config.Column(width="medium"),
                "Underlying signal evidence": st.column_config.Column(width="large"),
            },
        )

        st.subheader("The five anomalies")
        st.markdown(
            """
            #### 1. Both fuel cutoff switches moved to CUTOFF — simultaneously
            At **T-19.4 s** (FDR clock 288958.59), `Eng1 Cutoff SW` and
            `Eng2 Cutoff SW` both transition `RUN → CUTOFF` in the same FDR
            frame. These are two **physical switches**, one under each throttle
            lever, with separate mechanical paths. There is no normal-procedure
            reason to operate either of them in cruise; operating *both* at the
            same instant has no benign explanation.

            #### 2. Hard-over control wheel input — within ~2 seconds
            At **T-17.6 s**, the control wheel deflects from neutral cruise
            (≈ +0.35°) to **-28°** in a single sample, then continues to
            **-50° to -100°** (full deflection) and is held there for several
            seconds. The aileron surface responds proportionally.

            #### 3. The autopilot disconnected as a *consequence*, not a cause
            The autopilot was in `VNAV PATH` / `LNAV` mode through cruise.
            `AP-1 Warn` and `AP-2 Warn` only assert **after** the wheel input
            (T-17.5 s), and `CMD A` deselects at T-17.2 s. This is the
            standard 737 behaviour when manual control forces exceed the
            CWS override threshold — i.e. the autopilot was overridden, not
            commanding the upset.

            #### 4. The dive was caused by the *roll*, not by the elevator
            Through the entire 16-second descent, `Elevator-L` stays in the
            cruise band of -4° to -6°. Pitch goes from +2.5° to -36° entirely
            because the aircraft rolled past inverted and gravity pulled the
            nose toward earth. There is **no nose-down elevator command and no
            stab-trim runaway signature**.

            #### 5. No system warning preceded the upset
            Hydraulic A and B pressures stayed at ≈ 3000 psi throughout. No
            engine fire warning, no overspeed warning, no master-caution
            transition, no flap/slat asymmetry. Vertical G stayed within
            ~0 to 1.3 g during the dive (no sustained negative G that could
            unport fuel pickups). Engine fire indicators only assert in the
            **final 0.1 s**, consistent with impact, not an inflight fire.
            """
        )

        st.subheader("Five hypotheses, scored against the evidence")
        rows = [
            {
                "Scenario": "Deliberate cockpit input",
                "Consistency": "Strong",
                "What supports it": (
                    "Dual cutoff switches require deliberate two-handed action on "
                    "separate switches. Hard-over wheel input within 2 s. No "
                    "preceding system fault. AP override pattern matches manual "
                    "force. Roll-then-dive is exactly what someone deflecting a "
                    "wheel hard-left in cruise would produce."
                ),
                "What contradicts it": (
                    "The data alone cannot identify the actor or motive. CVR "
                    "would be needed to distinguish a single deliberate actor "
                    "from a struggle in the cockpit."
                ),
            },
            {
                "Scenario": "Pilot incapacitation / accidental input",
                "Consistency": "Very weak",
                "What supports it": (
                    "A slumped body could push a wheel to one side."
                ),
                "What contradicts it": (
                    "Cannot explain *both* fuel cutoff switches being moved to "
                    "CUTOFF in the same frame — they are physically separated "
                    "and require deliberate finger lifts to actuate. A "
                    "collapsing body would also typically mash the column "
                    "fore/aft, not the wheel."
                ),
            },
            {
                "Scenario": "Mechanical / electrical failure cascade",
                "Consistency": "Very weak",
                "What supports it": (
                    "Some single faults have produced unexpected upsets in "
                    "other accidents."
                ),
                "What contradicts it": (
                    "The fuel cutoff switches and the wheel input share no "
                    "electrical or hydraulic path. No master caution, no "
                    "hydraulic anomaly, no engine fault prior to cutoff. No "
                    "single failure mode on the 737-800 NG can produce this "
                    "exact signature, and certainly not without lighting any "
                    "warning."
                ),
            },
            {
                "Scenario": "Stab-trim / control-system runaway",
                "Consistency": "Inconsistent",
                "What supports it": (
                    "On other types (e.g. MAX/MCAS), runaway trim has caused "
                    "uncommanded pitch changes."
                ),
                "What contradicts it": (
                    "The 737-800 has no MCAS. The elevator does not move "
                    "during the dive — pitch follows from the roll. No "
                    "trim-related signal anomalies."
                ),
            },
            {
                "Scenario": "Hijacking / unauthorised cockpit access",
                "Consistency": "Cannot be excluded by data",
                "What supports it": (
                    "The data is consistent with a deliberate human actor; "
                    "FDR cannot identify *who*."
                ),
                "What contradicts it": (
                    "No public reporting of a security event on this flight; "
                    "very rare in modern commercial aviation. Indistinguishable "
                    "from #1 on FDR data alone — would need CVR / cabin / "
                    "investigative evidence."
                ),
            },
        ]
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
            column_config={
                "Scenario": st.column_config.Column(width="medium"),
                "Consistency": st.column_config.Column(width="small"),
                "What supports it": st.column_config.Column(width="large"),
                "What contradicts it": st.column_config.Column(width="large"),
            },
        )

        st.subheader("What this data cannot tell us")
        st.markdown(
            """
            - **Who** was at the controls. The FDR records inputs, not the
              identity of the person making them.
            - **Voice / sounds in the cockpit.** The CVR is not in this FOIA
              release. Any struggle, conversation, or alarm callout is invisible
              to this dataset.
            - **Motive.** Even if we are confident the inputs were deliberate,
              the data is silent on *why*.
            - **Whether the pilot or copilot was incapacitated** when the inputs
              were made.
            - **Absolute geographic position.** The FOIA data has no GPS lat/lon
              channels — only barometric altitude, computed airspeed, ground
              speed and heading.

            ##### Public context

            *In May 2022 the Wall Street Journal reported that US officials who
            had reviewed the preliminary FDR data assessed that "inputs to the
            controls pushed the plane into the fatal dive." The Civil Aviation
            Administration of China has issued only brief preliminary updates
            and has not publicly attributed cause as of this archive's release.*
            """
        )

    # --------------------------- 3D Replay tab ----------------------------
    with tab_replay:
        st.markdown(
            """
            Drag the slider (or hit **▶ Play**) to scrub through the final
            seconds. The red aircraft glyph rotates in real time using the FDR's
            pitch / roll / heading channels — you can see it bank past inverted
            and pitch nose-down through the dive. The trajectory ribbon is
            colour-coded by altitude (yellow = cruise, dark = ground).

            Path is integrated from heading × ground-speed (no GPS in the
            readout), so absolute position is approximate but the *shape*
            of the path is faithful.
            """
        )
        c1, c2 = st.columns(2)
        with c1:
            window_s = st.slider("Replay window (last N seconds)", 30, 300, 60, step=15)
        with c2:
            hz_choice = st.select_slider(
                "Frame rate",
                options=[2, 3, 5, 8, 10],
                value=5,
                help="Higher = smoother, but slower to load. 5 Hz is usually plenty.",
            )
        with st.spinner("Building 3D replay..."):
            fig = replay_3d_chart(df, t_window_s=float(window_s), hz=float(hz_choice))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Tip: drag the 3D scene to rotate. Scroll to zoom. Double-click to reset. "
            "Pause and step the slider one frame at a time for the upset onset."
        )

    # ------------------------- Event zoom tab -----------------------------
    with tab_event:
        st.markdown(
            """
            Last **30 seconds** of the recording, plotted at full 16 Hz.
            Roll passes through inverted (±180°), pitch reaches roughly -36°
            and altitude is in free-fall while computed airspeed climbs.
            """
        )
        end = df["Time"].max()
        win = df[df["Time"] > end - 30].reset_index(drop=True)
        st.plotly_chart(overview_chart(win, upset_t), width="stretch")

        # Bonus: pitch + roll + vertical-speed-derived in one panel
        alt_s = _series(win, "Altitude Press")
        vs = alt_s.diff() / win["Time"].diff() * 60.0  # ft/min
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Pitch & Roll", "Derived vertical speed (ft/min)"))
        fig.add_scatter(x=win["Time"], y=_series(win, "Pitch Angle"), name="Pitch (deg)", row=1, col=1)
        fig.add_scatter(x=win["Time"], y=_series(win, "Roll Angle"), name="Roll (deg)", row=1, col=1)
        fig.add_scatter(x=win["Time"], y=vs, name="VS (ft/min)", row=2, col=1, line=dict(color="purple"))
        fig.update_layout(height=520, hovermode="x unified", margin=dict(l=60, r=30, t=40, b=30))
        st.plotly_chart(fig, width="stretch")

    # -------------------------- Engines tab -------------------------------
    with tab_engines:
        st.markdown(
            "Both engines were producing roughly equal power throughout cruise. "
            "During the upset, fan-speed (N1) drops sharply on both engines as "
            "the autothrottle / FADEC reacts to airframe attitude. Look for any "
            "asymmetry."
        )
        panels = {
            "N1 (% RPM)": ["Eng1 N1", "Eng2 N1"],
            "Fuel flow (pph)": ["Eng1 Fuel Flow", "Eng2 Fuel Flow"],
            "EGT (degC)": ["Eng1 EGT", "Eng2 EGT"],
            "Thrust resolver angle (deg)": ["Eng1 TRA", "Eng2 TRA"],
        }
        st.plotly_chart(multi_panel_chart(df_filt, panels, upset_t), width="stretch")

    # --------------------- Flight controls tab ----------------------------
    with tab_controls:
        st.markdown(
            "Control-surface positions versus pilot inputs. Mismatches between "
            "control-column / wheel position and the elevator / aileron response "
            "can be informative."
        )
        panels = {
            "Elevators (deg)": ["Elevator-L", "Elevator-R"],
            "Ailerons (deg)": ["Aileron-L", "Aileron-R"],
            "Rudder (deg)": ["Rudder", "Rudder Ped Pos"],
            "Pilot inputs (deg)": ["Ctrl Col Pos-L", "Ctrl Whl Pos-L"],
        }
        st.plotly_chart(multi_panel_chart(df_filt, panels, upset_t), width="stretch")

    # ------------------------ Discrete states tab -------------------------
    with tab_discrete:
        st.markdown(
            "On/off and mode signals. Each row is one signal; the line steps up "
            "when the bit is asserted. Hover for the decoded label (e.g. "
            "`ON / OFF`, `Engaged / Not Engaged`)."
        )
        avail = [c for c in DISCRETE_STATES if c in df.columns]
        chosen = st.multiselect(
            "Discrete signals", options=avail, default=avail[: min(8, len(avail))]
        )
        st.plotly_chart(discrete_chart(df_filt, chosen, meta), width="stretch")

    # -------------------------- Ground track tab --------------------------
    with tab_track:
        st.markdown(
            """
            The FDR readout has no latitude/longitude, so this is a *relative*
            ground track integrated from heading × ground-speed starting at
            (0, 0). It captures the **shape** of the path, not absolute position.
            The aircraft tracks roughly east at cruise, then briefly drifts
            during the upset.
            """
        )
        st.plotly_chart(trajectory_xy(df_filt), width="stretch")

    # ----------------------- Parameter browser tab ------------------------
    with tab_browser:
        st.markdown(
            "Search across **all 165 parameters**. Click one to see its time "
            "series and metadata. Discrete parameters (booleans, mode flags) "
            "are flagged."
        )
        params = meta["parameters"]
        rows = []
        for name, info in params.items():
            if name in ("Time", "t_rel"):
                continue
            rows.append(
                {
                    "Parameter": name,
                    "Unit": info.get("unit", ""),
                    "Type": "discrete" if info.get("is_discrete") else "numeric",
                    "Samples": info.get("n_valid", 0),
                    "Min": info.get("min"),
                    "Max": info.get("max"),
                }
            )
        cat_df = pd.DataFrame(rows).sort_values("Parameter").reset_index(drop=True)
        q = st.text_input("Filter", placeholder="e.g. 'altitude', 'eng', 'roll'...")
        if q:
            ql = q.lower()
            cat_df = cat_df[cat_df["Parameter"].str.lower().str.contains(ql)]
        st.dataframe(cat_df, width="stretch", height=320)

        choice = st.selectbox(
            "Plot a parameter",
            options=cat_df["Parameter"].tolist() if len(cat_df) else [],
        )
        if choice:
            info = params[choice]
            st.markdown(
                f"**{choice}** — unit: `{info.get('unit') or 'n/a'}`, "
                f"valid samples: `{info.get('n_valid', 0):,}`, "
                f"type: `{'discrete' if info.get('is_discrete') else 'numeric'}`"
            )
            fig = go.Figure()
            fig.add_scatter(x=df_filt["Time"], y=_series(df_filt, choice), name=choice)
            if upset_t is not None and t_range[0] <= upset_t <= t_range[1]:
                fig.add_vline(x=upset_t, line=dict(color="red", width=1, dash="dot"))
            fig.update_layout(
                height=380,
                margin=dict(l=60, r=30, t=20, b=30),
                xaxis_title="FDR time (s)",
                yaxis_title=info.get("unit") or "",
                hovermode="x unified",
            )
            st.plotly_chart(fig, width="stretch")

    # ----------------------------- Compare tab ----------------------------
    with tab_compare:
        st.markdown(
            "Pick up to four parameters to overlay on a shared time axis. Each "
            "gets its own y-axis so units don't have to match."
        )
        all_numeric = sorted(
            n for n, info in meta["parameters"].items() if info.get("n_valid", 0) > 0 and not info.get("is_discrete")
        )
        defaults = [c for c in ("Altitude Press", "Airspeed Comp", "Pitch Angle", "Roll Angle") if c in all_numeric]
        chosen = st.multiselect("Parameters", options=all_numeric, default=defaults, max_selections=4)
        if chosen:
            colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
            fig = go.Figure()
            for i, c in enumerate(chosen):
                axis_id = "y" if i == 0 else f"y{i+1}"
                fig.add_scatter(
                    x=df_filt["Time"], y=_series(df_filt, c), name=c, yaxis=axis_id, line=dict(color=colors[i])
                )
            layout: dict = dict(
                height=520,
                margin=dict(l=60, r=160, t=30, b=30),
                xaxis=dict(title="FDR time (s)", domain=[0.05, 0.78]),
                yaxis=dict(title=dict(text=chosen[0], font=dict(color=colors[0])), tickfont=dict(color=colors[0])),
                hovermode="x unified",
                legend=dict(orientation="h", y=-0.15),
            )
            for i, c in enumerate(chosen[1:], start=2):
                key = f"yaxis{i}"
                layout[key] = dict(
                    title=dict(text=c, font=dict(color=colors[i - 1])),
                    tickfont=dict(color=colors[i - 1]),
                    overlaying="y",
                    side="right" if i % 2 == 0 else "left",
                    position=1.0 - 0.07 * (i - 2) if i % 2 == 0 else 0.05 * (i - 2),
                    anchor="free",
                )
            if upset_t is not None and t_range[0] <= upset_t <= t_range[1]:
                fig.add_vline(x=upset_t, line=dict(color="red", width=1, dash="dot"))
            fig.update_layout(**layout)
            st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    main()
