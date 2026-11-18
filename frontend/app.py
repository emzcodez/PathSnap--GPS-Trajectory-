"""PathSnap UI Shell - Application Experience & Delivery Support
Priyansh's module for the hackathon project.

This module provides the evaluator-facing UI shell and trip-selection flow,
route/map display component, and evidence organization.
"""

import os
import json
import argparse
from pathlib import Path

# PathSnap UI Configuration
PROJECT_NAME = "PathSnap: Distributed Map-Matching & Route Reconstruction"
VERSION = "1.0.0"

# API Response Schema (consistent with team contracts)
API_RESPONSE_SCHEMA = {
    "trip_id": None,
    "gps_points": [],
    "predicted_arc_ids": [],
    "predicted_route_geometry": [],
    "metrics": {},
    "processing_time_ms": 0
}

# Default data paths (relative to repository root)
DEFAULT_DATA_DIR = Path("artifacts")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_VISUALIZATIONS_DIR = Path("visualizations")


def load_trip_data(trip_id: str) -> dict:
    """Load trip data for the given trip ID.
    
    Args:
        trip_id: The trip identifier (e.g., "00000001")
        
    Returns:
        Dictionary containing trip data
    """
    # TODO: Load from actual artifacts directory
    # Expected structure: artifacts/trip_00000001/{gps, nodes, arcs, route}
    trip_path = DEFAULT_DATA_DIR / f"trip_{trip_id}"
    
    if not trip_path.exists():
        return {"error": f"Trip data not found for {trip_id}"}
    
    # Placeholder - will be integrated with actual data pipeline
    return {
        "trip_id": trip_id,
        "status": "loaded",
        "data_path": str(trip_path)
    }


def predict_route(trip_id: str) -> dict:
    """Run the complete prediction pipeline for a trip.
    
    Args:
        trip_id: The trip identifier
        
    Returns:
        Dictionary with prediction results following API response schema
    """
    # TODO: Integrate with Prateek's decoder and Prema's model
    # Pipeline: data_loader -> candidate generation -> decoder -> UI display
    
    trip_data = load_trip_data(trip_id)
    
    if "error" in trip_data:
        return {"error": trip_data["error"]}
    
    # Placeholder prediction results
    return {
        "trip_id": trip_id,
        "gps_points": [],  # Will be populated from actual GPS data
        "predicted_arc_ids": [],  # From decoder output
        "predicted_route_geometry": [],  # Route geometry for display
        "metrics": {
            "accuracy": 0.0,
            "processing_time_ms": 0.0
        },
        "processing_time_ms": 0.0
    }


def create_map_visualization(trip_id: str, output_path: str = None) -> str:
    """Create a Folium map visualization for a trip.
    
    Args:
        trip_id: The trip identifier
        output_path: Path to save the HTML file
        
    Returns:
        Path to the saved visualization file
    """
    # TODO: Integrate with Prarthana's Folium maps
    # Should include: GPS observations, road geometry, reference route, predicted route
    
    if output_path is None:
        output_path = DEFAULT_VISUALIZATIONS_DIR / f"trip_{trip_id}_map.html"
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Placeholder - will generate actual Folium map with real data
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>PathSnap - Trip {trip_id} Visualization</title>
    <link rel="stylesheet" href="https://unpkg.com/folium@0.15.1/dist/folium.css">
</head>
<body>
    <div id="map">{ PROJECT_NAME }</div>
    <script src="https://unpkg.com/folium@0.15.1/dist/folium.js"></script>
    <script>
        // Initialize map - will be populated with actual data
        var map = L.map('map').setView([0, 0], 12);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org{{z}}{{x}}{{y}}.png', {
            maxZoom: 19,
            attribution: '&copy; OpenStreetMap contributors'
        }).addTo(map);
    </script>
</body>
</html>"""
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    return str(output_path)


def display_results(results: dict) -> str:
    """Format prediction results for display in the UI.
    
    Args:
        results: Dictionary from predict_route() function
        
    Returns:
        Formatted string for UI display
    """
    if "error" in results:
        return f"Error: {results['error']}"
    
    trip_id = results.get("trip_id", "N/A")
    predicted_arcs = results.get("predicted_arc_ids", [])
    metrics = results.get("metrics", {})
    processing_time = results.get("processing_time_ms", 0)
    
    output = f"""## PathSnap Results - Trip {trip_id}
**Predicted Arc IDs:** {predicted_arcs}
**Processing Time:** {processing_time:.2f} ms
**Metrics:** {json.dumps(metrics, indent=2)}
"""
    return output


def main():
    """Main entry point for the PathSnap UI shell."""
    parser = argparse.ArgumentParser(description=f"{PROJECT_NAME} UI Shell")
    parser.add_argument(
        "--trip-id", 
        type=str, 
        default=None,
        help="Trip ID to process (e.g., 00000001)"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Generate Folium map visualization"
    )
    parser.add_argument(
        "--output", 
        type=str,
        default=None,
        help="Output path for map visualization"
    )
    
    args = parser.parse_args()
    
    print(f"=== PathSnap UI Shell ===")
    print(f"Project: {PROJECT_NAME} v{VERSION}")
    print()
    
    if args.trip_id:
        print(f"Processing trip: {args.trip_id}")
        
        # Load trip data
        trip_data = load_trip_data(args.trip_id)
        print(f"Trip data: {trip_data}")
        
        # Run prediction
        results = predict_route(args.trip_id)
        print(f"Prediction results:")
        print(display_results(results))
        
        # Generate visualization if requested
        if args.visualize or not args.output:
            viz_path = create_map_visualization(args.trip_id, args.output)
            print(f"Map visualization saved to: {viz_path}")
    else:
        print("Usage: python frontend/app.py --trip-id <trip_id> [--visualize] [--output <path>]")
        print()
        print("Available trip IDs can be found in the artifacts directory.")


if __name__ == "__main__":
    main()
