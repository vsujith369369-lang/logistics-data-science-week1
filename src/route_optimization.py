"""Phase 4-5: delivery territory clustering and capacitated vehicle routing."""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


def haversine_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Great-circle distance matrix in km (fallback when OSRM is unavailable)."""
    lat = np.radians(lats)[:, None]
    lon = np.radians(lons)[:, None]
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def cluster_territories(stops: pd.DataFrame, k_range=range(4, 16)) -> pd.DataFrame:
    """K-Means territories, k chosen by silhouette score."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    coords = StandardScaler().fit_transform(stops[["lat", "lon"]])
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(coords)
        scores[k] = silhouette_score(coords, labels)

    best_k = max(scores, key=scores.get)
    out = stops.copy()
    out["zone"] = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(coords)
    return out


def solve_vrp(distance_matrix, demands, vehicle_capacities, depot: int = 0, seconds: int = 30):
    """Capacitated VRP solved with Google OR-Tools guided local search."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    mgr = pywrapcp.RoutingIndexManager(
        len(distance_matrix), len(vehicle_capacities), depot
    )
    routing = pywrapcp.RoutingModel(mgr)

    def dist_cb(i, j):
        return int(distance_matrix[mgr.IndexToNode(i)][mgr.IndexToNode(j)])

    transit = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    def demand_cb(i):
        return int(demands[mgr.IndexToNode(i)])

    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(demand_cb),
        0,
        list(vehicle_capacities),
        True,
        "Capacity",
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.FromSeconds(seconds)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("No feasible routing solution found")

    routes = []
    for v in range(len(vehicle_capacities)):
        idx, route = routing.Start(v), []
        while not routing.IsEnd(idx):
            route.append(mgr.IndexToNode(idx))
            idx = solution.Value(routing.NextVar(idx))
        route.append(mgr.IndexToNode(idx))
        routes.append(route)

    return routes, solution.ObjectiveValue()


if __name__ == "__main__":
    # Small self-contained demonstration
    rng = np.random.default_rng(42)
    lats = np.r_[12.97, 12.97 + rng.normal(0, 0.05, 9)]
    lons = np.r_[77.59, 77.59 + rng.normal(0, 0.05, 9)]
    matrix = np.round(haversine_matrix(lats, lons) * 1000)  # metres
    demands = [0] + [10] * 9
    routes, cost = solve_vrp(matrix, demands, [40, 40])
    print("Routes:", routes)
    print("Total distance (m):", cost)
