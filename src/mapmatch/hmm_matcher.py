"""Dependency-light OSM road graph and fixed-lag HMM/Viterbi matcher."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as et

import networkx as nx
import numpy as np

from src.preprocessing.frames import geodetic_to_local_enu


DRIVABLE_HIGHWAYS = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
    "residential", "living_street", "service", "motorway_link", "trunk_link",
    "primary_link", "secondary_link", "tertiary_link",
}


@dataclass(frozen=True)
class RoadSegment:
    segment_id: int
    way_id: int
    node_u: int
    node_v: int
    start_enu_m: np.ndarray
    end_enu_m: np.ndarray
    length_m: float
    bearing_deg: float
    bidirectional: bool


@dataclass(frozen=True)
class Candidate:
    segment: RoadSegment
    point_enu_m: np.ndarray
    fraction: float
    distance_m: float
    heading_error_deg: float
    emission_log_probability: float


@dataclass(frozen=True)
class MatchResult:
    position_enu_m: np.ndarray
    segment_id: np.ndarray
    way_id: np.ndarray
    candidate_distance_m: np.ndarray
    heading_error_deg: np.ndarray
    path_log_probability: float


class OSMRoadNetwork:
    def __init__(self, graph: nx.Graph, segments: list[RoadSegment]) -> None:
        if not segments:
            raise ValueError("road network contains no drivable segments")
        self.graph = graph
        self.segments = segments

    @classmethod
    def from_xml(
        cls, path: str | Path, *, origin_lat_deg: float, origin_lon_deg: float
    ) -> "OSMRoadNetwork":
        root = et.parse(path).getroot()
        nodes_ll = {
            int(node.attrib["id"]): (float(node.attrib["lat"]), float(node.attrib["lon"]))
            for node in root.findall("node")
        }
        node_ids = np.asarray(list(nodes_ll), dtype=np.int64)
        coordinates = np.asarray([nodes_ll[int(node_id)] for node_id in node_ids])
        enu = geodetic_to_local_enu(
            coordinates[:, 0], coordinates[:, 1],
            origin_lat_deg=origin_lat_deg, origin_lon_deg=origin_lon_deg,
        )
        node_enu = {int(node_id): point for node_id, point in zip(node_ids, enu)}
        graph = nx.Graph()
        segments: list[RoadSegment] = []
        for way in root.findall("way"):
            tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way.findall("tag")}
            if tags.get("highway") not in DRIVABLE_HIGHWAYS or tags.get("access") == "private":
                continue
            references = [int(node.attrib["ref"]) for node in way.findall("nd")]
            if tags.get("oneway") == "-1":
                references.reverse()
            bidirectional = tags.get("oneway") not in {"yes", "1", "true", "-1"}
            for node_u, node_v in zip(references, references[1:]):
                if node_u not in node_enu or node_v not in node_enu:
                    continue
                start, end = node_enu[node_u], node_enu[node_v]
                delta = end[:2] - start[:2]
                length = float(np.linalg.norm(delta))
                if length < 0.05:
                    continue
                bearing = float(np.rad2deg(np.arctan2(delta[0], delta[1])) % 360.0)
                segment = RoadSegment(
                    len(segments), int(way.attrib["id"]), node_u, node_v,
                    start.copy(), end.copy(), length, bearing, bidirectional,
                )
                segments.append(segment)
                graph.add_node(node_u, position=start)
                graph.add_node(node_v, position=end)
                old = graph.get_edge_data(node_u, node_v)
                if old is None or length < old["length"]:
                    graph.add_edge(node_u, node_v, length=length)
        return cls(graph, segments)


def _angle_difference_deg(first: float, second: float) -> float:
    return float(abs((first - second + 180.0) % 360.0 - 180.0))


def _trajectory_bearing(position: np.ndarray, half_window: int) -> np.ndarray:
    delta = np.empty((len(position), 2))
    for index in range(len(position)):
        start, end = max(0, index - half_window), min(len(position) - 1, index + half_window)
        delta[index] = position[end, :2] - position[start, :2]
    return np.rad2deg(np.arctan2(delta[:, 0], delta[:, 1])) % 360.0


class HMMMapMatcher:
    def __init__(
        self,
        network: OSMRoadNetwork,
        *,
        candidate_count: int = 6,
        maximum_candidate_distance_m: float = 120.0,
        distance_sigma_m: float = 18.0,
        heading_sigma_deg: float = 35.0,
        transition_beta_m: float = 12.0,
        transition_heading_sigma_deg: float = 30.0,
        road_turn_sigma_deg: float = 25.0,
        way_change_penalty: float = 12.0,
        heading_smoothing_half_window: int = 20,
        window_size: int = 30,
    ) -> None:
        if candidate_count < 1 or window_size < 2:
            raise ValueError("candidate_count and window_size are too small")
        if min(maximum_candidate_distance_m, distance_sigma_m, heading_sigma_deg, transition_beta_m, transition_heading_sigma_deg, road_turn_sigma_deg) <= 0 or way_change_penalty < 0:
            raise ValueError("matcher scale parameters must be positive")
        if heading_smoothing_half_window < 1:
            raise ValueError("heading smoothing half-window must be positive")
        self.network = network
        self.candidate_count = candidate_count
        self.maximum_candidate_distance_m = maximum_candidate_distance_m
        self.distance_sigma_m = distance_sigma_m
        self.heading_sigma_deg = heading_sigma_deg
        self.transition_beta_m = transition_beta_m
        self.transition_heading_sigma_deg = transition_heading_sigma_deg
        self.road_turn_sigma_deg = road_turn_sigma_deg
        self.way_change_penalty = way_change_penalty
        self.heading_smoothing_half_window = heading_smoothing_half_window
        self.window_size = window_size

    def _candidate(self, segment: RoadSegment, point: np.ndarray, heading: float) -> Candidate:
        vector = segment.end_enu_m[:2] - segment.start_enu_m[:2]
        fraction = float(np.clip(
            np.dot(point[:2] - segment.start_enu_m[:2], vector) / np.dot(vector, vector), 0.0, 1.0
        ))
        projected = segment.start_enu_m + fraction * (segment.end_enu_m - segment.start_enu_m)
        distance = float(np.linalg.norm(point[:2] - projected[:2]))
        heading_error = _angle_difference_deg(heading, segment.bearing_deg)
        if segment.bidirectional:
            heading_error = min(heading_error, _angle_difference_deg(heading, segment.bearing_deg + 180.0))
        emission = -0.5 * (distance / self.distance_sigma_m) ** 2
        emission -= 0.5 * (heading_error / self.heading_sigma_deg) ** 2
        return Candidate(segment, projected, fraction, distance, heading_error, emission)

    def candidates(self, point: np.ndarray, heading_deg: float) -> list[Candidate]:
        values = [self._candidate(segment, point, heading_deg) for segment in self.network.segments]
        within = [value for value in values if value.distance_m <= self.maximum_candidate_distance_m]
        pool = within if within else values
        return sorted(pool, key=lambda value: -value.emission_log_probability)[:self.candidate_count]

    @lru_cache(maxsize=100_000)
    def _node_distance(self, first: int, second: int) -> float:
        if first == second:
            return 0.0
        try:
            return float(nx.shortest_path_length(
                self.network.graph, first, second, weight="length"
            ))
        except nx.NetworkXNoPath:
            return float("inf")

    def _route_distance(self, first: Candidate, second: Candidate) -> float:
        if first.segment.segment_id == second.segment.segment_id:
            return abs(second.fraction - first.fraction) * first.segment.length_m
        first_ends = (
            (first.segment.node_u, first.fraction * first.segment.length_m),
            (first.segment.node_v, (1.0 - first.fraction) * first.segment.length_m),
        )
        second_ends = (
            (second.segment.node_u, second.fraction * second.segment.length_m),
            (second.segment.node_v, (1.0 - second.fraction) * second.segment.length_m),
        )
        distances = [
            first_offset + self._node_distance(first_node, second_node) + second_offset
            for first_node, first_offset in first_ends
            for second_node, second_offset in second_ends
        ]
        return min(distances)

    def _transition_log_probability(
        self, first: Candidate, second: Candidate, observed_distance_m: float,
        observed_heading_deg: float,
    ) -> float:
        route_distance = self._route_distance(first, second)
        if not np.isfinite(route_distance):
            return -1.0e9
        probability = -abs(route_distance - observed_distance_m) / self.transition_beta_m
        delta = second.point_enu_m[:2] - first.point_enu_m[:2]
        if np.linalg.norm(delta) > 0.2:
            candidate_heading = float(np.rad2deg(np.arctan2(delta[0], delta[1])) % 360.0)
            heading_error = _angle_difference_deg(candidate_heading, observed_heading_deg)
            probability -= 0.5 * (heading_error / self.transition_heading_sigma_deg) ** 2
        if first.segment.way_id != second.segment.way_id:
            probability -= self.way_change_penalty
            road_turn = _angle_difference_deg(
                first.segment.bearing_deg, second.segment.bearing_deg
            )
            road_turn = min(road_turn, abs(180.0 - road_turn))
            probability -= 0.5 * (road_turn / self.road_turn_sigma_deg) ** 2
        return probability

    def _viterbi_window(
        self,
        candidates: list[list[Candidate]],
        observed_steps: np.ndarray,
        observed_headings: np.ndarray,
        previous: Candidate | None,
    ) -> list[Candidate]:
        scores = np.asarray([candidate.emission_log_probability for candidate in candidates[0]])
        if previous is not None:
            scores += np.asarray([
                self._transition_log_probability(
                    previous, candidate, observed_steps[0], observed_headings[0]
                )
                for candidate in candidates[0]
            ])
        backpointers: list[np.ndarray] = []
        for index in range(1, len(candidates)):
            transition = np.asarray([
                [self._transition_log_probability(
                    before, after, observed_steps[index], observed_headings[index]
                )
                 for after in candidates[index]]
                for before in candidates[index - 1]
            ])
            combined = scores[:, None] + transition
            parents = np.argmax(combined, axis=0)
            scores = combined[parents, np.arange(len(candidates[index]))]
            scores += np.asarray([candidate.emission_log_probability for candidate in candidates[index]])
            backpointers.append(parents)
        state = int(np.argmax(scores))
        states = [state]
        for parents in reversed(backpointers):
            state = int(parents[state]); states.append(state)
        states.reverse()
        return [options[state] for options, state in zip(candidates, states)]

    def match(
        self, position_enu_m: np.ndarray, heading_deg: np.ndarray | None = None
    ) -> MatchResult:
        position = np.asarray(position_enu_m, dtype=float)
        if position.ndim != 2 or position.shape[1] != 3 or len(position) < 2:
            raise ValueError("trajectory must have shape (N, 3), N >= 2")
        if not np.isfinite(position).all():
            raise ValueError("trajectory must be finite")
        if heading_deg is None:
            headings = _trajectory_bearing(position, self.heading_smoothing_half_window)
        else:
            headings = np.array(heading_deg, dtype=float, copy=True)
            if headings.shape != (len(position),) or not np.isfinite(headings).all():
                raise ValueError("heading_deg must be a finite array with shape (N,)")
            headings %= 360.0
        all_candidates = [self.candidates(point, heading) for point, heading in zip(position, headings)]
        observed_steps = np.r_[0.0, np.linalg.norm(np.diff(position[:, :2], axis=0), axis=1)]
        committed: list[Candidate] = []
        last_path: list[Candidate] = []
        for end in range(len(position)):
            start = max(0, end - self.window_size + 1)
            previous = committed[-1] if committed else None
            last_path = self._viterbi_window(
                all_candidates[start:end + 1], observed_steps[start:end + 1],
                headings[start:end + 1], previous,
            )
            if len(last_path) == self.window_size:
                committed.append(last_path[0])
        selected = committed + last_path[1:] if committed else last_path
        if len(selected) != len(position):
            raise RuntimeError("fixed-lag Viterbi produced an invalid path length")
        matched = np.asarray([candidate.point_enu_m for candidate in selected])
        matched[:, 2] = position[:, 2]
        return MatchResult(
            matched,
            np.asarray([candidate.segment.segment_id for candidate in selected]),
            np.asarray([candidate.segment.way_id for candidate in selected]),
            np.asarray([candidate.distance_m for candidate in selected]),
            np.asarray([candidate.heading_error_deg for candidate in selected]),
            float(sum(candidate.emission_log_probability for candidate in selected)),
        )
