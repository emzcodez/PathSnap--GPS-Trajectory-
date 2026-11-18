"""PathSnap Trip Selector Component
Priyansh's module for trip selection and listing.

Provides functionality to list available trips from the artifacts directory
and select a trip ID for processing.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict


def list_available_trips(artifacts_dir: str = "artifacts") -> List[Dict[str, str]]:
    """List all available trips in the artifacts directory.
    
    Args:
        artifacts_dir: Path to the artifacts directory
        
    Returns:
        List of dictionaries with trip information
    """
    trips = []
    artifacts_path = Path(artifacts_dir)
    
    if not artifacts_path.exists():
        print(f"Artifacts directory not found: {artifacts_dir}")
        return trips
    
    # Look for trip directories (trip_XXXX format)
    for item in artifacts_path.iterdir():
        if item.is_dir() and item.name.startswith("trip_"):
            trip_id = item.name.replace("trip_", "")
            # Check for essential data files
            gps_file = item / "gps_data.json"
            has_data = gps_file.exists()
            
            trips.append({
                "trip_id": trip_id,
                "path": str(item),
                "has_gps_data": has_data,
                "description": f"Trip {trip_id}" + (" (with GPS data)" if has_data else " (no GPS data)")
            })
    
    # Sort by trip ID
    trips.sort(key=lambda x: int(x["trip_id"]) if x["trip_id"].isdigit() else x["trip_id"])
    return trips


def get_trip_info(trip_id: str, artifacts_dir: str = "artifacts") -> Optional[Dict[str, any]]:
    """Get detailed information about a specific trip.
    
    Args:
        trip_id: The trip identifier (e.g., "00000001")
        artifacts_dir: Path to the artifacts directory
        
    Returns:
        Dictionary with trip information or None if not found
    """
    artifacts_path = Path(artifacts_dir) / f"trip_{trip_id}"
    
    if not artifacts_path.exists():
        return None
    
    # Collect available data files
    data_files = []
    for f in artifacts_path.rglob("*"):
        if f.is_file():
            data_files.append({
                "name": f.name,
                "path": str(f.relative_to(artifacts_path)),
                "size": f.stat().st_size,
                "relative_path": str(f.relative_to(artifacts_path))
            })
    
    return {
        "trip_id": trip_id,
        "path": str(artifacts_path),
        "data_files": data_files,
        "has_gps": any("gps" in f["name"].lower() for f in data_files),
        "has_candidates": any("candidate" in f["name"].lower() for f in data_files),
        "has_route": any("route" in f["name"].lower() for f in data_files),
    }


def recommend_trip(trips: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Recommend the best trip to process based on available data.
    
    Args:
        trips: List of trip dictionaries from list_available_trips()
        
    Returns:
        Recommended trip dictionary or None if no trips available
    """
    # Prioritize trips with GPS data
    trips_with_gps = [t for t in trips if t.get("has_gps_data", False)]
    
    if trips_with_gps:
        # Return the first trip with GPS data, sorted by ID
        trips_with_gps.sort(key=lambda x: int(x["trip_id"]) if x["trip_id"].isdigit() else x["trip_id"])
        return trips_with_gps[0]
    
    # Fall back to any available trip
    if trips:
        return trips[0]
    
    return None


# CLI interface for trip selection
def select_trip_interactive(artifacts_dir: str = "artifacts") -> Optional[str]:
    """Interactively select a trip from available options.
    
    Args:
        artifacts_dir: Path to the artifacts directory
        
    Returns:
        Selected trip ID or None if no trips available
    """
    trips = list_available_trips(artifacts_dir)
    
    if not trips:
        print("No trips found in the artifacts directory.")
        print("Please ensure trip data is in 'artifacts/trip_XXXX/' format.")
        return None
    
    print("\nAvailable Trips:")
    print("=" * 60)
    
    for i, trip in enumerate(trips, 1):
        gps_status = "✓" if trip.get("has_gps_data", False) else "✗"
        print(f"{i}. Trip {trip['trip_id']} - {gps_status} {trip['description']}")
    
    print("=" * 60)
    
    while True:
        try:
            choice = input(f"\nSelect trip (1-{len(trips)}) or enter trip ID: ").strip()
            
            if not choice:
                continue
            
            # Try as number selection
            choice_num = int(choice)
            if 1 <= choice_num <= len(trips):
                selected = trips[choice_num - 1]
                print(f"\nSelected: Trip {selected['trip_id']}")
                return selected["trip_id"]
            
            # Try as direct trip ID
            # Validate the trip ID exists
            trip_info = get_trip_info(choice, artifacts_dir)
            if trip_info:
                print(f"\nSelected: Trip {choice}")
                return choice
            else:
                print(f"Trip ID '{choice}' not found. Please try again.")
                
        except ValueError:
            # Try as direct trip ID
            trip_info = get_trip_info(choice, artifacts_dir)
            if trip_info:
                print(f"\nSelected: Trip {choice}")
                return choice
            else:
                print(f"Invalid input. Please enter a number 1-{len(trips)} or a valid trip ID.")
                
                

if __name__ == "__main__":
    # CLI mode - list trips
    import sys
    
    artifacts_dir = sys.argv[1] if len(sys.argv) > 1 else "artifacts"
    trips = list_available_trips(artifacts_dir)
    
    if trips:
        print(f"Found {len(trips)} trips in {artifacts_dir}/")
        for trip in trips:
            print(f"  - Trip {trip['trip_id']}: {trip['description']}")
    else:
        print(f"No trips found in {artifacts_dir}/")
        print("Expected format: artifacts/trip_XXXX/")
