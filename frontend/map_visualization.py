"""PathSnap Map Visualization Module
Priyansh's module for creating Folium-based map visualizations.

This module creates interactive map visualizations showing GPS trajectories,
road networks, reference routes, and predicted routes.
"""

import folium
from pathlib import Path
from typing import List, Dict, Any, Optional


def create_folium_map(
    center: List[float] = [0.0, 0.0],
    zoom_start: int = 12,
    tiles: str = "OpenStreetMap",
) -> folium.Map:
    """Create a base Folium map.
    
    Args:
        center: [latitude, longitude] for map center
        zoom_start: Initial zoom level
        tiles: Tile layer provider
        
    Returns:
        A Folium Map object
    """
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles=tiles,
        control_scale=True,
    )
    return m


def add_gps_track(
    m: folium.Map,
    gps_data: List[List[float]],
    color: str = "blue",
    weight: int = 2,
    opacity: float = 1.0,
    name: str = "GPS Track",
) -> folium.Map:
    """Add GPS track line to the map.
    
    Args:
        m: Folium Map object
        gps_data: List of [longitude, latitude] pairs
        color: Line color
        weight: Line thickness
        opacity: Line opacity
        name: Layer name
        
    Returns:
        Modified Folium Map object
    """
    if not gps_data or len(gps_data) < 2:
        return m
    
    # Convert [lon, lat] pairs for Folium
    points = [[lat, lon] for lon, lat in gps_data]
    
    folium.PolyLine(
        locations=points,
        color=color,
        weight=weight,
        opacity=opacity,
        tooltip="GPS Trajectory",
    ).add_to(m)
    
    return m


def add_marker_cluster(
    m: folium.Map,
    points: List[List[float]],
    color: str = "red",
    popup: str = "Observation",
    name: str = "Points",
) -> folium.Map:
    """Add a marker cluster to the map.
    
    Args:
        m: Folium Map object
        points: List of [longitude, latitude] pairs
        color: Marker color
        popup: Popup text for markers
        name: Layer name
        
    Returns:
        Modified Folium Map object
    """
    from folium.plugins import MarkerCluster
    
    marker_cluster = MarkerCluster(name=name).add_to(m)
    
    for lon, lat in points:
        folium.Marker(
            location=[lat, lon],
            popup=popup,
            icon=folium.Icon(color=color, icon="info-sign"),
        ).add_to(marker_cluster)
    
    return m


def add_route_layer(
    m: folium.Map,
    route_geometry: List[List[float]],
    color: str = "green",
    weight: int = 3,
    opacity: float = 0.8,
    name: str = "Predicted Route",
) -> folium.Map:
    """Add predicted route geometry to the map.
    
    Args:
        m: Folium Map object
        route_geometry: List of [longitude, latitude] pairs
        color: Line color
        weight: Line thickness
        opacity: Line opacity
        name: Layer name
        
    Returns:
        Modified Folium Map object
    """
    if not route_geometry or len(route_geometry) < 2:
        return m
    
    # Convert [lon, lat] pairs
    points = [[lat, lon] for lon, lat in route_geometry]
    
    folium.PolyLine(
        locations=points,
        color=color,
        weight=weight,
        opacity=opacity,
        tooltip="Predicted Route",
    ).add_to(m)
    
    return m


def add_reference_route(
    m: folium.Map,
    reference_route: List[List[float]],
    color: str = "orange",
    weight: int = 3,
    opacity: float = 0.8,
    name: str = "Reference Route",
) -> folium.Map:
    """Add reference route geometry to the map.
    
    Args:
        m: Folium Map object
        reference_route: List of [longitude, latitude] pairs
        color: Line color
        weight: Line thickness
        opacity: Line opacity
        name: Layer name
        
    Returns:
        Modified Folium Map object
    """
    if not reference_route or len(reference_route) < 2:
        return m
    
    # Convert [lon, lat] pairs
    points = [[lat, lon] for lon, lat in reference_route]
    
    folium.PolyLine(
        locations=points,
        color=color,
        weight=weight,
        opacity=opacity,
        dash_array="5, 5",
        tooltip="Reference Route",
    ).add_to(m)
    
    return m


def save_map(
    m: folium.Map,
    output_path: str,
    title: str = "PathSnap Map",
) -> str:
    """Save the Folium map to an HTML file.
    
    Args:
        m: Folium Map object
        output_path: Path where to save the HTML file
        title: Title for the map
        
    Returns:
        Path to the saved file
    """
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    m.save(str(output_file))
    return str(output_file)


def generate_trip_visualization(
    trip_id: str,
    gps_coordinates: List[List[float]],
    predicted_route: List[List[float]] = None,
    reference_route: List[List[float]] = None,
    output_path: str = None,
) -> str:
    """Generate a complete trip visualization map.
    
    This is the main entry point for creating trip visualizations
    that include GPS tracks, predicted routes, and optional reference routes.
    
    Args:
        trip_id: Identifier for the trip
        gps_coordinates: List of [longitude, latitude] GPS points
        predicted_route: Optional [longitude, latitude] route geometry from decoder
        reference_route: Optional [longitude, latitude] ground-truth route
        output_path: Path to save the HTML file. If None, auto-generates.
        
    Returns:
        Path to the saved HTML visualization file
    """
    # Determine map center from GPS data
    if gps_coordinates:
        lons = [pt[0] for pt in gps_coordinates]
        lats = [pt[1] for pt in gps_coordinates]
        center = [sum(lats) / len(lats), sum(lons) / len(lons)]
    else:
        center = [0.0, 0.0]
    
    # Create base map
    m = create_folium_map(center=center, zoom_start=13)
    
    # Add GPS track
    m = add_gps_track(m, gps_coordinates, color="blue", name="GPS Track")
    
    # Add predicted route if provided
    if predicted_route:
        m = add_route_layer(m, predicted_route, color="green", name="Predicted Route")
    
    # Add reference route if provided
    if reference_route:
        m = add_reference_route(m, reference_route, color="orange", name="Reference Route")
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    if output_path is None:
        output_path = f"visualizations/trip_{trip_id}_map.html"
    
    return save_map(m, output_path)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="PathSnap Map Visualization")
    parser.add_argument("--trip-id", type=str, default="00000001")
    parser.add_argument("--gps", type=str, help="JSON list of [lon, lat] pairs")
    parser.add_argument("--predicted", type=str, help="JSON list of [lon, lat] predicted route")
    parser.add_argument("--reference", type=str, help="JSON list of [lon, lat] reference route")
    parser.add_argument("--output", type=str, default=None)
    
    args = parser.parse_args()
    
    gps_data = []
    if args.gps:
        gps_data = eval(args.gps)  # Simplified for demo
    
    predicted = None
    if args.predicted:
        predicted = eval(args.predicted)
    
    reference = None
    if args.reference:
        reference = eval(args.reference)
    
    output = generate_trip_visualization(
        trip_id=args.trip_id,
        gps_coordinates=gps_data,
        predicted_route=predicted,
        reference_route=reference,
        output_path=args.output,
    )
    
    print(f"Map saved to: {output}")
