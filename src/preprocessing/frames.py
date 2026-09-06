from __future__ import annotations

import numpy as np


WGS84_A_M = 6_378_137.0
WGS84_E2 = 6.69437999014e-3


def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, altitude_m: float | np.ndarray = 0.0) -> np.ndarray:
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    altitude = np.broadcast_to(np.asarray(altitude_m, dtype=float), lat.shape)
    prime_vertical = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    x = (prime_vertical + altitude) * np.cos(lat) * np.cos(lon)
    y = (prime_vertical + altitude) * np.cos(lat) * np.sin(lon)
    z = (prime_vertical * (1.0 - WGS84_E2) + altitude) * np.sin(lat)
    return np.column_stack((x, y, z))


def geodetic_to_local_enu(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    origin_index: int = 0,
    *,
    altitude_m: float | np.ndarray = 0.0,
    origin_lat_deg: float | None = None,
    origin_lon_deg: float | None = None,
    origin_altitude_m: float = 0.0,
) -> np.ndarray:
    """Convert WGS84 coordinates to ENU, optionally using an explicit origin."""
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    ecef = geodetic_to_ecef(lat, lon, altitude_m)
    if (origin_lat_deg is None) != (origin_lon_deg is None):
        raise ValueError("origin latitude and longitude must be provided together")
    if origin_lat_deg is None:
        origin_lat = float(lat[origin_index])
        origin_lon = float(lon[origin_index])
        origin_altitude = float(np.broadcast_to(np.asarray(altitude_m), lat.shape)[origin_index])
    else:
        origin_lat = float(origin_lat_deg)
        origin_lon = float(origin_lon_deg)
        origin_altitude = float(origin_altitude_m)
    origin = geodetic_to_ecef(
        np.array([origin_lat]), np.array([origin_lon]), np.array([origin_altitude])
    )[0]
    lat0 = np.deg2rad(origin_lat)
    lon0 = np.deg2rad(origin_lon)
    ecef_to_enu = np.array(
        [
            [-np.sin(lon0), np.cos(lon0), 0.0],
            [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
            [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)],
        ]
    )
    return (ecef - origin) @ ecef_to_enu.T


def bearing_deg_to_enu_velocity(speed_mps: float, bearing_deg: float) -> np.ndarray:
    """Navigation bearing (clockwise from North) -> ENU velocity."""
    bearing = np.deg2rad(bearing_deg)
    return np.array([speed_mps * np.sin(bearing), speed_mps * np.cos(bearing), 0.0])


def vehicle_to_enu_rotation(bearing_deg: float) -> np.ndarray:
    """Vehicle X-forward, Y-left, Z-up -> ENU at a navigation bearing."""
    bearing = np.deg2rad(bearing_deg)
    return np.array(
        [
            [np.sin(bearing), -np.cos(bearing), 0.0],
            [np.cos(bearing), np.sin(bearing), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
