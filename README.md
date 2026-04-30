# NTSB FOIA MU5735

This repository contains files originally shared by another GitHub user related to FOIA requests on the MU5735 investigation.

The original repository has since been deleted or made private by its owner. To protect their privacy, the files have been re-uploaded here without preserving the original commit history.

This repository serves as an archive to maintain access to those materials.

## Interactive dashboard

A Streamlit-based explorer for the FDR data ([TableResolution.csv](TableResolution.csv) and [ExactSample.csv](ExactSample.csv)) is included.

**Run it:**

```bash
uv run python preprocess.py     # one-time: builds cache/ Parquet + parameter metadata
uv run streamlit run app.py     # opens at http://localhost:8501
```

`preprocess.py` parses the multi-row FDR header (column names, units, discrete-state decoders), drops periodic sentinel/saturation frames where many sensors simultaneously read max-value, and writes a Parquet cache for fast reload.

**Tabs:**

| Tab | What it shows |
| --- | --- |
| Story | The "what happened" view — altitude, airspeed, pitch, roll over the full ~13-minute window with the upset onset marked |
| 3D Replay | Interactive 3D animation of the final seconds. The aircraft glyph rotates in real time using pitch / roll / heading; trajectory ribbon is colour-coded by altitude. Drag the slider or hit ▶ Play to scrub. |
| Event zoom | Last 30 s at full 16 Hz, plus derived vertical speed |
| Engines | N1, fuel flow, EGT, TRA for both engines |
| Flight controls | Elevator / aileron / rudder positions vs. column / wheel / pedal inputs |
| Discrete states | Autopilot mode, fire warnings, etc. as step-line bands |
| Ground track | Relative XY path integrated from heading × ground-speed (no GPS in this readout) |
| Parameter browser | Searchable list of all 165 parameters with units and quick-plot |
| Compare | Overlay up to four numeric parameters with independent y-axes |

The sidebar exposes a global time-range filter and aircraft / event metadata pulled from the CSV header.
