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


def load_trip_from_api(trip_id: str, api_data: dict = None) -> dict:
    """Load trip data from API response (for 2-laptop final flow).
    
    Args:
        trip_id: The trip identifier
        api_data: Dictionary from backend API response
        
    Returns:
        Dictionary containing trip data
    """
    # If API data provided, use it; otherwise load from artifacts
    if api_data and "trip_id" in api_data:
        # Transform API response to internal format
        return {
            "trip_id": api_data.get("trip_id", trip_id),
            "status": "api_loaded",
            "api_data": api_data,
            "data_path": f"api_response_{trip_id}"
        }
    
    # Fallback to artifacts loading
    return load_trip_data(trip_id)


def predict_route(trip_id: str, api_data: dict = None) -> dict:
    """Run the complete prediction pipeline for a trip.
    
    Args:
        trip_id: The trip identifier
        api_data: Optional API data for 2-laptop flow
        
    Returns:
        Dictionary with prediction results following API response schema
    """
    # TODO: Integrate with Prateek's decoder and Prema's model
    # Pipeline: data_loader -> candidate generation -> decoder -> UI display
    
    trip_data = load_trip_from_api(trip_id, api_data)
    
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
        output_path: Path to save the HTML file (string or Path)
        
    Returns:
        Path to the saved visualization file as string
    """
    # Convert string to Path if needed
    if output_path is not None and not isinstance(output_path, Path):
        output_path = Path(output_path)
    
    if output_path is None:
        output_path = DEFAULT_VISUALIZATIONS_DIR / f"trip_{trip_id}_map.html"
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate HTML using string concatenation to avoid f-string JavaScript variable conflicts
    # Title part with trip_id
    title_line = "PathSnap - Trip " + trip_id + " Visualization"
    
    # Map div part
    map_div_line = "PathSnap: Distributed Map-Matching & Route Reconstruction"
    
    # Full HTML using concatenation
    html_content = ""
    html_content += "<!DOCTYPE html>\n"
    html_content += "<html>\n"
    html_content += "<head>\n"
    html_content += "    <title>" + title_line + "</title>\n"
    html_content += "    <link rel=\"stylesheet\" href=\"https://unpkg.com/folium@0.15.1/dist/folium.css\">\n"
    html_content += "</head>\n"
    html_content += "<body>\n"
    html_content += "    <div id=\"map\">" + map_div_line + "</div>\n"
    html_content += "    <script src=\"https://unpkg.com/folium@0.15.1/dist/folium.js\"></script>\n"
    html_content += "<script>\n"
    html_content += "        // Initialize map - will be populated with actual data\n"
    html_content += "        var map = L.map('map').setView([0, 0], 12);\n"
    html_content += "        L.tileLayer('https://{s}.tile.openstreetmap.org/z/x/y.png', {\n"
    html_content += "            maxZoom: 19,\n"
    html_content += "            attribution: '&copy; OpenStreetMap contributors'\n"
    html_content += "        }).addTo(map);\n"
    html_content += "    </script>\n"
    html_content += "</body>\n"
    html_content += "</html>"
    
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
    
    output = "## PathSnap Results - Trip " + trip_id + "\n"
    output += "**Predicted Arc IDs:** " + str(predicted_arcs) + "\n"
    output += "**Processing Time:** " + "{:.2f}".format(processing_time) + " ms\n"
    output += "**Metrics:** " + json.dumps(metrics, indent=2) + "\n"
    return output


def main():
    """Main entry point for the PathSnap UI shell."""
    parser = argparse.ArgumentParser(description="PathSnap UI Shell")
    parser.add_argument(
        "--trip-id", 
        type=str, 
        default=None,
        help="Trip ID to process (e.g., 00000001)"
    )
    parser.add_argument(
        "--api-data", 
        type=str,
        default=None,
        help="JSON string of API response data (for 2-laptop flow)"
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
    
    print("=== PathSnap UI Shell ===")
    print("Project: " + PROJECT_NAME + " v" + VERSION)
    print()
    
    # Determine trip ID
    trip_id = args.trip_id
    api_data = None
    
    if args.api_data:
        # Parse API data JSON
        try:
            api_data = json.loads(args.api_data)
            trip_id = api_data.get("trip_id", trip_id or "00000001")
        except json.JSONDecodeError:
            print("Warning: Invalid API data JSON, using trip-id argument")
    
    if not trip_id:
        # List available trips
        print("Available trips:")
        artifacts_path = Path("artifacts")
        if artifacts_path.exists():
            for item in artifacts_path.iterdir():
                if item.is_dir() and item.name.startswith("trip_"):
                    print("  - " + item.name)
        else:
            print("  (no artifacts directory found)")
        print()
        print("Usage: python app.py --trip-id <trip_id> [--api-data '{\"trip_id\": \"00000001\"}'] [--visualize] [--output <path>]")
        print()
        print("Or without arguments to list trips and exit.")
        return
    
    if trip_id:
        print("Processing trip: " + trip_id)
        
        # Load trip data (from API or artifacts)
        trip_data = load_trip_from_api(trip_id, api_data)
        print("Trip data: " + str(trip_data))
        
        # Run prediction
        results = predict_route(trip_id, api_data)
        print("Prediction results:")
        print(display_results(results))
        
        # Generate visualization if requested
        if args.visualize or not args.output:
            viz_path = create_map_visualization(trip_id, args.output)
            print("Map visualization saved to: " + viz_path)
    
    if not args.trip_id and not args.api_data:
        print("\nNo trip specified. Use --trip-id or --api-data to process a specific trip.")


if __name__ == "__main__":
    main()
