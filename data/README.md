# ScoutEdge Data Directory

## Overview
This directory stores video inputs and manifests for the ScoutEdge Bowling Delivery Evidence Pipeline.

## Files
- `input_manifest.csv`: Input specification file containing clip IDs, relative file paths, SHA-256 checksums, frame rates, resolutions, durations, and anonymized athlete IDs.
- `raw/`: Local video files (never pushed to public repositories as governed by the ScoutEdge Technical Assessment Agreement).

## Schema for `input_manifest.csv`
| Column | Description |
| :--- | :--- |
| `clip_id` | Unique identifier (e.g. `dev_01`) |
| `file_name` | Relative path to MP4 video |
| `sha256` | SHA-256 integrity hash of original file |
| `frame_rate` | Nominal FPS |
| `resolution` | Dimensions (Width x Height) |
| `duration` | Total duration in seconds |
| `camera_view` | Orientation / View type (`side-on`, `front-on`, `behind`, `broadcast_wide`, `mobile_vertical`) |
| `athlete_id` | Anonymized placeholder ID (`ATH-PLACEHOLDER-XX`) |
