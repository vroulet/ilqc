"""Tracks for autonomous car racing."""

import math
import numpy as np
import os
import pandas as pd
import torch
from envs.tracks.torch_spline import (
    natural_cubic_spline_coeffs,
    NaturalCubicSpline,
)


def get_track(
    track: str,
) -> tuple[NaturalCubicSpline, NaturalCubicSpline, NaturalCubicSpline]:
    """Select track among: a line, a bend, a simple track or a complex track.

    Complex track is from [Optimization-Based Autonomous Racing of 143 Scale RC Cars, Liniger et al 2017])
    Returns three splines one for the center, one for the inner border, one for the outer border of the track
    The splines are encoded as NaturalCubicSpline (from Patrick Ridge,
    https://github.com/patrick-kidger/torchcubicspline). The spliens are functions of time, which given a continuous
    time t outputs the corresponding point on the track.

    Args:
      track: choice of the track in ['line', 'bend', 'circle', 'simple', complex']

    Returns:
        - center: a spline that approximates the center of the track
        - inner: a spline that approximates the inner border of the track
        - outer: a spline that approximates the outer border of the track
    """
    tracks_folder = os.path.dirname(os.path.abspath(__file__))
    track_file = os.path.join(tracks_folder, track + "_track.json")
    if not os.path.exists(track_file):
        make_track(track)
    track_data = pd.read_json(
        os.path.join(tracks_folder, track + "_track.json")
    )

    time = None
    lines = []
    starting_point = None
    for coord in ["", "_i", "_o"]:
        line = track_data[["X" + coord, "Y" + coord]].to_numpy(dtype=float)
        if starting_point is None:
            starting_point = line[0]
        line = line - starting_point
        line = torch.from_numpy(line)
        if time is None:
            time = np.sqrt(np.sum(np.diff(line, axis=0) ** 2, axis=1))
            time = np.insert(time, 0, 0).cumsum()
            time = torch.from_numpy(time)
        coeffs = natural_cubic_spline_coeffs(time, line)
        if track in ["simple", "complex", "circle"]:
            line = NaturalCubicSpline(coeffs, handle_termination="loop")
        else:
            line = NaturalCubicSpline(
                coeffs, handle_termination="repeat_end_point"
            )

        lines.append(line)
    center, inner, outer = lines
    return center, inner, outer


def make_track(track: str) -> None:
    if track == "line":
        make_line()
    elif track == "bend":
        make_bend()
    elif track == "circle":
        make_circle()
    elif track == "simple":
        make_simple_track()
    elif track == "complex":
        # The coordinates of the complex track have already been recorded and are taken from
        # [Optimization-Based Autonomous Racing of 143 Scale RC Cars, Liniger et al 2017]
        pass
    else:
        raise NotImplementedError(f"Track {track} not implemented")


def make_line(border_length: float = 0.37, nb_points: int = 100) -> None:
    # border length chosen such that it matches the one of the complex circuit after rescaling
    x_center = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0])
    y_center = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    center_track = np.stack((x_center, y_center)).transpose() / 172

    make_track_from_np_array(
        center_track, "line_track", border_length, nb_points
    )


def make_bend(border_length: float = 0.37, nb_points: int = 100) -> None:
    R = 50.0
    short = 50
    pts_bend = math.ceil(R * math.pi / 2)

    cx = []
    cy = []
    # border length chosen such that it matches the one of the complex circuit after rescaling

    # 1st straight line
    cx = cx + [float(t) for t in range(short + 1)]
    cy = cy + [0.0 for t in range(short + 1)]

    # 1st bend
    cx = cx + [
        cx[-1] + R * math.cos(-math.pi / 2 + t / pts_bend * math.pi / 2)
        for t in range(1, pts_bend + 1)
    ]
    cy = cy + [
        cy[-1] + R + R * math.sin(-math.pi / 2 + t / pts_bend * math.pi / 2)
        for t in range(1, pts_bend + 1)
    ]

    # 2nd straight line
    cx = cx + [cx[-1] for t in range(1, short + 1)]
    cy = cy + [cy[-1] + t for t in range(1, short + 1)]

    center_track = np.array([cx, cy]).transpose() / 172

    make_track_from_np_array(
        center_track, "bend_track", border_length, nb_points
    )


def make_circle(border_length: float = 0.37, nb_points: int = 100) -> None:
    R = 50.0
    pts_bend = math.ceil(R * math.pi / 2)
    cx = [0.0]
    cy = [0.0]

    cx = cx + [
        cx[-1] + R * math.cos(-math.pi / 2 + t / pts_bend * math.pi / 2)
        for t in range(1, 4 * pts_bend + 1)
    ]
    cy = cy + [
        cy[-1] + R + R * math.sin(-math.pi / 2 + t / pts_bend * math.pi / 2)
        for t in range(1, 4 * pts_bend + 1)
    ]

    center_track = np.array([cx, cy]).transpose() / 172

    make_track_from_np_array(
        center_track, "circle_track", border_length, nb_points
    )


def make_simple_track(
    border_length: float = 0.37, nb_points: int = 100
) -> None:
    # border length chosen such that it matches the one of the complex circuit after rescaling
    R = 50.0
    # points per bend
    pts_bend = math.ceil(R * math.pi / 2)

    long = 100
    short = 50
    cx = []
    cy = []
    # 1st straight line
    cx = cx + [float(t) for t in range(long + 1)]
    cy = cy + [0.0 for t in range(long + 1)]

    # 1st bend
    cx = cx + [
        cx[-1] + R * math.cos(-math.pi / 2 + t / pts_bend * math.pi / 2)
        for t in range(1, pts_bend + 1)
    ]
    cy = cy + [
        cy[-1] + R + R * math.sin(-math.pi / 2 + t / pts_bend * math.pi / 2)
        for t in range(1, pts_bend + 1)
    ]

    # 2nd straight line
    cx = cx + [cx[-1] for t in range(1, long + 1)]
    cy = cy + [cy[-1] + t for t in range(1, long + 1)]

    # 2nd bend
    cx = cx + [
        cx[-1] - R + R * math.cos(t / pts_bend * math.pi / 2)
        for t in range(1, pts_bend + 1)
    ]
    cy = cy + [
        cy[-1] + R * math.sin(t / pts_bend * math.pi / 2)
        for t in range(1, pts_bend + 1)
    ]

    # 3rd straight line
    cx = cx + [cx[-1] - t for t in range(1, short + 1)]
    cy = cy + [cy[-1] for t in range(1, short + 1)]

    # 4th bend
    cx = cx + [
        cx[-1] + R * math.cos(math.pi / 2 + t / pts_bend * math.pi / 2)
        for t in range(1, pts_bend + 1)
    ]
    cy = cy + [
        cy[-1] - R + R * math.sin(math.pi / 2 + t / pts_bend * math.pi / 2)
        for t in range(1, pts_bend + 1)
    ]

    # 5th bend
    cx = cx + [
        cx[-1] + -R + R * math.cos(-t / pts_bend * math.pi / 2)
        for t in range(1, pts_bend + 1)
    ]
    cy = cy + [
        cy[-1] + R * math.sin(-t / pts_bend * math.pi / 2)
        for t in range(1, pts_bend + 1)
    ]

    # 5th straight line
    cx = cx + [cx[-1] - t for t in range(1, long + 1)]
    cy = cy + [cy[-1] for t in range(1, long + 1)]

    # 6th bend
    cx = cx + [
        cx[-1] + R * math.cos(math.pi / 2 + t / pts_bend * math.pi / 2)
        for t in range(1, 2 * pts_bend + 1)
    ]
    cy = cy + [
        cy[-1] - R + R * math.sin(math.pi / 2 + t / pts_bend * math.pi / 2)
        for t in range(1, 2 * pts_bend + 1)
    ]

    cx = cx + [cx[-1] + t for t in range(1, long + short + 1)]
    cy = cy + [cy[-1] for t in range(1, long + short + 1)]
    center_track = np.array([cx, cy]).transpose() / 172

    make_track_from_np_array(
        center_track, "simple_track", border_length, nb_points
    )


def make_track_from_np_array(
    center_track: np.ndarray,
    name_track: str,
    border_length: float,
    nb_points: int,
) -> None:
    """Create track and record it in a json file.

    Given a list of points recorded in a numpy array create the spline corresponding to this track and record
    nb_points of this spline as the definition of this track, the resulting file is saved for further use.

    Args:
      center_track: array
      name_track: Name of the track to be saved
      border_length: width of the track
      nb_points: number of points used to define the spline. More points give a more precise curve
        but slow down the computations

    """
    l = border_length
    time = np.sqrt(np.sum(np.diff(center_track, axis=0) ** 2, axis=1))
    time = np.insert(time, 0, 0).cumsum()
    time, center_track = torch.from_numpy(time), torch.from_numpy(center_track)
    coeffs = natural_cubic_spline_coeffs(time, center_track)
    spline = NaturalCubicSpline(coeffs)

    time = torch.linspace(0, max(spline._t), nb_points)
    center_track = spline.evaluate(time)
    dtrack = spline.derivative(time)

    vs = torch.sqrt(torch.sum(dtrack**2, dim=1))
    normals = torch.tensor(
        [[dpos[1] / v, -dpos[0] / v] for dpos, v in zip(dtrack, vs)]
    )
    inner_border = center_track - l / 2 * normals
    outer_border = center_track + l / 2 * normals

    track = np.concatenate(
        (center_track.numpy(), inner_border.numpy(), outer_border.numpy()),
        axis=1,
    )
    tracks_folder = os.path.dirname(os.path.abspath(__file__))
    pd.DataFrame(
        track, columns=["X", "Y", "X_i", "Y_i", "X_o", "Y_o"]
    ).to_json(
        os.path.join(tracks_folder, name_track) + ".json",
        orient="columns",
        indent=1,
    )
