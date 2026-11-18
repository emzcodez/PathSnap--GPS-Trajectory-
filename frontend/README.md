# PathSnap UI Module

## Application Experience & Delivery Support (Priyansh)

## Overview
This module provides the evaluator-facing UI shell and trip-selection flow for the PathSnap hackathon project. It handles:
- Trip selection and loading
- Route prediction display
- Interactive map visualizations using Folium
- Evidence organization and screenshot capture
- README and run instructions generation

## Directory Structure
```
frontend/
├── app.py                      # Main CLI entry point
├── map_visualization.py        # Folium-based map visualizations
├── components/                 # UI components (to be developed)
│   ├── trip_selector.py
│   ├── route_display.py
│   └── metrics_panel.py
├── integration/                # Pipeline integration (to be developed)
│   └── pipeline.py
├── artifacts/                  # Data artifacts (populated during runtime)
│   └── trip_XXXX/              # Per-trip data directories
├── results/                    # Prediction results and metrics
└── visualizations/             # Generated map HTML files
```

## Key Components

### app.py - Main Entry Point
The main CLI application that:
- Loads trip data for a given trip ID
- Runs the prediction pipeline
- Generates map visualizations
- Displays prediction results

### map_visualization.py - Map Visualizations
Creates interactive Folium maps showing:
- GPS trajectories (blue line)
- Predicted routes (green line)
- Reference routes (orange dashed line)
- Marker clusters for observations

### API Response Schema
All UI components follow the agreed schema:
```json
{
  "trip_id": "00000001",
  "gps_points": [...],
  "predicted_arc_ids": [...],
  "predicted_route_geometry": [...],
  "metrics": {...},
  "processing_time_ms": 123.4
}
```

## First-Phase Sprint Deliverables (10:00 AM - 11:40 AM)

### 1. Runnable UI Shell
- `python frontend/app.py --trip-id 00000001` should work
- Accepts `--trip-id`, `--visualize`, and `--output` arguments
- Displays trip data and prediction results

### 2. Trip Selector
- List available trips from the `artifacts/` directory
- Allow selection of trip ID for processing

### 3. Map/Results Display Scaffold
- Generate Folium map visualization for a trip
- Save as HTML file in `visualizations/` directory
- Include GPS track, predicted route, and reference route layers

### 4. Screenshot/Evidence Folder
- Organize screenshots with trip ID and partition labels
- Save successful experiment outputs
- Record failed experiments and limitations

### 5. README Draft and Integration Checklist
- Document setup instructions
- List dependencies (requirements.txt)
- Provide integration checklist for full pipeline

## Development Workflow

### Git Workflow (Team Leader Instructions)
1. **Create your own branch** for UI work
2. **Make commits** with clear messages
3. **Raise pull requests** when work is ready
4. **Team reviews and merges** PRs

### Running the UI Shell
```bash
# From the pathnap_ui_work directory
cd frontend
python app.py --trip-id 00000001

# With map visualization
python app.py --trip-id 00000001 --visualize --output visualizations/trip_00000001_map.html
```

### Adding New Trips
1. Place trip data in `artifacts/trip_XXXX/` directory
2. Expected structure:
   - `artifacts/trip_XXXX/gps_data.pkl` or `.json`
   - `artifacts/trip_XXXX/candidate_output.json`
   - `artifacts/trip_XXXX/decoder_output.json`

## Integration with Other Modules

### Data Flow
```
Prarthana's Data Loader --> Prema's Model/Decoder --> Priyansh's UI
```

### Expected Artifact Formats
- **GPS data**: DataFrame with longitude, latitude, timestamp columns
- **Candidate outputs**: `{"observation_index": N, "candidate_arc_ids": [...], "probabilities": [...]}`
- **Decoder output**: Sequence of predicted arc IDs
- **Route geometry**: List of [longitude, latitude] pairs

## Notes for Hackathon
- Keep all generated outputs in `artifacts/`, `results/`, and `visualizations/`
- Never commit large raw data files unless submission requires it
- Follow the shared interface contracts defined in the team strategy
- Document any preprocessing assumptions and limitations
- Disclose AI assistance as required by event policy
