# Prema — PathSnap Phase 1

This module implements candidate road-arc generation, candidate features, route-reference-based candidate labels, candidate miss-rate calculation, and a first Random Forest baseline.

## Dataset contract
Trip files are tab-separated:
- `.track`: longitude, latitude, timestamp
- `.nodes`: longitude, latitude (row index is node ID)
- `.arcs`: from-node ID, to-node ID (row index is arc ID)
- `.route`: reference arc IDs

The current reference-label assignment is an **approximation**: for each GPS observation, it finds the geometrically nearest arc among the reference route arcs, then labels a generated candidate positive when its arc ID matches that route arc. This is explicitly documented because the files do not provide a direct GPS-observation-to-arc label mapping.

## Run
From this folder:
```bash
pip install numpy pandas scipy scikit-learn
python models/run_phase1.py --dataset "../pathsnap_student_dataset/Student folder" --train-trips 5 --test-trips 2
```

The script writes actual candidate tables and metrics to `results/`. Increase trip counts after the initial smoke test.
