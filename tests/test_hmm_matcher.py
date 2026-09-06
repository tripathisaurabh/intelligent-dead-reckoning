import networkx as nx
import numpy as np

from src.mapmatch.hmm_matcher import HMMMapMatcher, OSMRoadNetwork, RoadSegment


def _network():
    graph = nx.Graph()
    segments = []
    for segment_id, (u, v, start, end) in enumerate([
        (1, 2, [0, 0, 0], [100, 0, 0]),
        (3, 4, [0, 15, 0], [100, 15, 0]),
    ]):
        start, end = np.asarray(start, float), np.asarray(end, float)
        graph.add_edge(u, v, length=100.0)
        segments.append(RoadSegment(segment_id, 10 + segment_id, u, v, start, end, 100.0, 90.0, True))
    return OSMRoadNetwork(graph, segments)


def test_hmm_uses_sequence_to_stay_on_consistent_parallel_road():
    matcher = HMMMapMatcher(_network(), candidate_count=2, distance_sigma_m=10, window_size=4)
    trajectory = np.column_stack((np.linspace(5, 95, 12), np.full(12, 2.0), np.zeros(12)))
    trajectory[6, 1] = 9.0  # one point is nearer the other road
    result = matcher.match(trajectory)
    assert len(result.position_enu_m) == len(trajectory)
    assert set(result.segment_id) == {0}
    assert np.allclose(result.position_enu_m[:, 1], 0.0)


def test_candidate_projection_and_validation():
    matcher = HMMMapMatcher(_network(), candidate_count=1)
    candidate = matcher.candidates(np.array([25.0, 4.0, 0.0]), 90.0)[0]
    assert np.allclose(candidate.point_enu_m[:2], [25.0, 0.0])
    assert np.isclose(candidate.distance_m, 4.0)


def test_matcher_accepts_filter_heading_as_separate_observation():
    matcher = HMMMapMatcher(_network(), candidate_count=2, window_size=4)
    trajectory = np.column_stack((np.linspace(5, 95, 8), np.full(8, 2.0), np.zeros(8)))
    result = matcher.match(trajectory, np.full(8, 90.0))
    assert set(result.segment_id) == {0}
