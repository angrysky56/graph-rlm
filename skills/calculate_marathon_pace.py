def calculate_marathon_pace(marathon_time_str, distance_km=42.195):
    """
    Calculate marathon pace per km and per mile from total time.

    Args:
        marathon_time_str (str): Time in format "HH:MM:SS" or "MM:SS"
        distance_km (float): Marathon distance in kilometers (default 42.195)

    Returns:
        dict: Pace calculations including per km and per mile
    """
    try:
        # Parse time string
        time_parts = marathon_time_str.split(":")
        if len(time_parts) == 3:
            hours, minutes, seconds = map(int, time_parts)
        elif len(time_parts) == 2:
            hours, minutes = 0, int(time_parts[0])
            seconds = int(time_parts[1])
        else:
            raise ValueError("Invalid time format")

        # Convert to total seconds
        total_seconds = hours * 3600 + minutes * 60 + seconds

        # Calculate paces
        pace_per_km = total_seconds / distance_km
        pace_per_mile = total_seconds / 26.2  # 26.2 miles is marathon distance

        # Format as MM:SS
        pace_min_km = int(pace_per_km // 60)
        pace_sec_km = int(pace_per_km % 60)

        pace_min_mile = int(pace_per_mile // 60)
        pace_sec_mile = int(pace_per_mile % 60)

        return {
            "total_time": marathon_time_str,
            "pace_per_km": f"{pace_min_km}:{pace_sec_km:02d}",
            "pace_per_mile": f"{pace_min_mile}:{pace_sec_mile:02d}",
            "pace_seconds_per_km": round(pace_per_km, 2),
            "pace_seconds_per_mile": round(pace_per_mile, 2),
        }

    except Exception as e:
        return {"error": str(e)}


# Example usage:
if __name__ == "__main__":
    # Kipchoge's official world record
    result = calculate_marathon_pace("2:01:09")
    print(f"Kipchoge World Record - Time: {result['total_time']}")
    print(f"Pace: {result['pace_per_km']} per km / {result['pace_per_mile']} per mile")
