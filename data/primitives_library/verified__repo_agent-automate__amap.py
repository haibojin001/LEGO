from __future__ import annotations

from .base import BaseIntegration

API = "https://restapi.amap.com/v3"


class AmapIntegration(BaseIntegration):
    name = "amap"
    label = "高德地图 (Amap)"
    env_vars = {
        "AMAP_API_KEY": "API key from lbs.amap.com/dev/key/app",
    }

    def _key(self) -> str:
        return self.env("AMAP_API_KEY")

    def register(self, mcp) -> None:
        integration = self

        @mcp.tool()
        def amap_geocode(address: str, city: str = "") -> str:
            """
            Geocode an address to coordinates using 高德地图.

            Args:
                address: Address string (e.g. "北京市朝阳区望京SOHO").
                city: Optional city hint to improve accuracy (e.g. "北京").
            """
            import urllib.parse

            parameters = {
                "key": integration._key(),
                "address": address,
            }
            if city:
                parameters["city"] = city

            result = integration.get(
                f"{API}/geocode/geo?{urllib.parse.urlencode(parameters)}"
            )
            matches = result.get("geocodes", [])

            if not matches:
                return integration.ok(
                    {
                        "error": "Address not found",
                        "status": result.get("info"),
                    }
                )

            match = matches[0]
            lng, lat = match.get("location", ",").split(",")
            return integration.ok(
                {
                    "address": match.get("formatted_address"),
                    "lng": lng,
                    "lat": lat,
                    "level": match.get("level"),
                }
            )

        @mcp.tool()
        def amap_reverse_geocode(lng: str, lat: str) -> str:
            """
            Reverse geocode coordinates to address using 高德地图.

            Args:
                lng: Longitude (e.g. "116.480053").
                lat: Latitude (e.g. "39.987005").
            """
            import urllib.parse

            parameters = {
                "key": integration._key(),
                "location": f"{lng},{lat}",
                "extensions": "base",
            }
            result = integration.get(
                f"{API}/geocode/regeo?{urllib.parse.urlencode(parameters)}"
            )
            reverse_data = result.get("regeocode", {})
            component = reverse_data.get("addressComponent", {})

            return integration.ok(
                {
                    "address": reverse_data.get("formatted_address"),
                    "province": component.get("province"),
                    "city": component.get("city"),
                    "district": component.get("district"),
                }
            )

        @mcp.tool()
        def amap_search_poi(
            keywords: str,
            city: str = "",
            limit: int = 10,
        ) -> str:
            """
            Search for POIs (Points of Interest) using 高德地图.

            Args:
                keywords: Search keywords (e.g. "星巴克", "加油站").
                city: City name to restrict search (e.g. "上海").
                limit: Max results (default 10).
            """
            import urllib.parse

            parameters = {
                "key": integration._key(),
                "keywords": keywords,
                "offset": limit,
                "extensions": "base",
            }
            if city:
                parameters["city"] = city

            result = integration.get(
                f"{API}/place/text?{urllib.parse.urlencode(parameters)}"
            )
            pois = result.get("pois", [])
            output = [f"Found {len(pois)} POI(s) for '{keywords}':"]

            for poi in pois:
                lng, lat = poi.get("location", ",").split(",")
                output.append(
                    f"  {poi.get('name')} — {poi.get('address')} ({lng},{lat})"
                )

            return "\n".join(output)

        @mcp.tool()
        def amap_route_drive(
            origin_lng: str,
            origin_lat: str,
            dest_lng: str,
            dest_lat: str,
        ) -> str:
            """
            Get driving route between two points using 高德地图.

            Args:
                origin_lng: Origin longitude.
                origin_lat: Origin latitude.
                dest_lng: Destination longitude.
                dest_lat: Destination latitude.
            """
            import urllib.parse

            parameters = {
                "key": integration._key(),
                "origin": f"{origin_lng},{origin_lat}",
                "destination": f"{dest_lng},{dest_lat}",
                "extensions": "base",
            }
            result = integration.get(
                f"{API}/direction/driving?{urllib.parse.urlencode(parameters)}"
            )
            route = result.get("route", {})
            paths = route.get("paths", [])

            if not paths:
                return integration.ok({"error": "No route found"})

            path = paths[0]
            return integration.ok(
                {
                    "distance_m": path.get("distance"),
                    "duration_s": path.get("duration"),
                    "tolls_yuan": path.get("tolls"),
                    "steps": len(path.get("steps", [])),
                }
            )