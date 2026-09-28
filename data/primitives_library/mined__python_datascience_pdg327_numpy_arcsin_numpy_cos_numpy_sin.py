# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg327::numpy.arcsin+numpy.cos+numpy.sin
# name: numpy_primitive
# summary: Uses numpy.arcsin, numpy.cos, numpy.sin, numpy.sqrt across 3 repos
# anchor_symbols: ['numpy.arcsin', 'numpy.cos', 'numpy.sin', 'numpy.sqrt']
# observed in 3 repos: ['alteryx__featuretools', 'feature-engine__feature_engine', 'meteostat__meteostat']...

# --- from alteryx__featuretools::featuretools/primitives/standard/transform/latlong/utils.py::_haversine_calculate ---
def _haversine_calculate(lat_1s, lon_1s, lat_2s, lon_2s, unit):
    # https://stackoverflow.com/a/29546836/2512385
    lon1, lat1, lon2, lat2 = map(np.radians, [lon_1s, lat_1s, lon_2s, lat_2s])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    radius_earth = 3958.7613
    if unit == "kilometers":
        radius_earth = 6371.0088
    distances = radius_earth * 2 * np.arcsin(np.sqrt(a))
    return distances

# --- from meteostat__meteostat::meteostat/utils/geo.py::get_distance ---
def get_distance(lat1, lon1, lat2, lon2) -> int:
    """
    Calculate distance between two geographical points using the Haversine formula
    """
    # Earth radius in meters
    radius = 6371000

    # Degress to radian
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])

    # Deltas
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Calculate distance
    arch = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    arch_sin = 2 * np.arcsin(np.sqrt(arch))

    return round(radius * arch_sin)

# --- from feature-engine__feature_engine::feature_engine/creation/geo_features.py::GeoDistanceFeatures._haversine_distance ---
def _haversine_distance(
        self,
        lat1: np.ndarray,
        lon1: np.ndarray,
        lat2: np.ndarray,
        lon2: np.ndarray,
    ) -> np.ndarray:
        """Calculate the great-circle distance using the Haversine formula."""

        # Convert to radians
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        lon1_rad = np.radians(lon1)
        lon2_rad = np.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        # Distance in the requested unit
        distance = EARTH_RADIUS[self.output_unit] * c

        return distance
