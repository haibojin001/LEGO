# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg831::math.atan2+math.cos+math.sin
# name: math_primitive
# summary: Uses math.atan2, math.cos, math.sin, math.sqrt across 2 repos
# anchor_symbols: ['math.atan2', 'math.cos', 'math.sin', 'math.sqrt']
# observed in 2 repos: ['firmai__pandasvault', 'scikit-mobility__scikit-mobility']...

# --- from firmai__pandasvault::pandasvault/__init__.py::haversine_distance ---
def haversine_distance(row, lon="latitude", lat="longitude"):
    c_lat,c_long = radians(52.5200), radians(13.4050)
    R = 6373.0
    long = radians(row['longitude'])
    lat = radians(row['latitude'])
    
    dlon = long - c_long
    dlat = lat - c_lat
    a = sin(dlat / 2)**2 + cos(lat) * cos(c_lat) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c

# --- from scikit-mobility__scikit-mobility::skmob/utils/gislib.py::getDistanceByHaversine ---
def getDistanceByHaversine(loc1, loc2):
    "Haversine formula - give coordinates as (lat_decimal,lon_decimal) tuples"

    lat1, lon1 = loc1
    lat2, lon2 = loc2
    
    # convert to radians
    lon1 = lon1 * pi / 180.0
    lon2 = lon2 * pi / 180.0
    lat1 = lat1 * pi / 180.0
    lat2 = lat2 * pi / 180.0

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2.0))**2
    c = 2.0 * atan2(sqrt(a), sqrt(1.0-a))
    km = earthradius * c
    return km
