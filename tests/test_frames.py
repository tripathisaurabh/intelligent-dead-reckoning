import numpy as np

from src.preprocessing.frames import (
    bearing_deg_to_enu_velocity,
    geodetic_to_local_enu,
    vehicle_to_enu_rotation,
)


def test_enu_origin_is_zero_and_northward_latitude_is_north():
    enu = geodetic_to_local_enu(np.array([12.0, 12.0001]), np.array([77.0, 77.0]))
    assert np.allclose(enu[0], 0.0)
    assert enu[1, 1] > 10.0
    assert abs(enu[1, 0]) < 0.1


def test_explicit_enu_origin_places_independent_tracks_in_same_frame():
    origin = (12.0, 77.0)
    first = geodetic_to_local_enu(
        np.array([12.0001]), np.array([77.0]),
        origin_lat_deg=origin[0], origin_lon_deg=origin[1],
    )
    second = geodetic_to_local_enu(
        np.array([12.0001]), np.array([77.0]),
        origin_lat_deg=origin[0], origin_lon_deg=origin[1],
    )
    assert np.allclose(first, second)
    assert first[0, 1] > 10.0


def test_navigation_bearing_velocity_convention():
    assert np.allclose(bearing_deg_to_enu_velocity(5.0, 0.0), [0.0, 5.0, 0.0], atol=1e-12)
    assert np.allclose(bearing_deg_to_enu_velocity(5.0, 90.0), [5.0, 0.0, 0.0], atol=1e-12)


def test_vehicle_rotation_maps_forward_to_bearing():
    north = vehicle_to_enu_rotation(0.0) @ np.array([1.0, 0.0, 0.0])
    east = vehicle_to_enu_rotation(90.0) @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(north, [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(east, [1.0, 0.0, 0.0], atol=1e-12)
