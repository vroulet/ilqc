import math
import numpy as np
import os
import pandas as pd
import json
import torch
from envs.utils.torch_spline import natural_cubic_spline_coeffs, NaturalCubicSpline


def get_track(track):
    """
    Select track among: a line, a bend, a simple track or a complex track
    (complex track from [Optimization-Based Autonomous Racing of 143 Scale RC Cars, Liniger et al 2017])
    :param track: (string) choice of the track in ['line', 'bend', 'simple', complex']
    :param scale: (boolean) scale the track such that it is in approx. [-5, 5]
    :param loop: (boolean) if True, the index
    :return: spline: (NaturalCubicSpline) a spline that approximates the track
    """
    dir_name = os.path.dirname(os.path.abspath(__file__))
    track_data = pd.read_json(os.path.join(dir_name, track + "_track.json"))

    time = None
    lines = []
    starting_point = None
    for coord in ['', '_i', '_o']:
        line = track_data[['X' + coord, 'Y' + coord]].to_numpy(dtype=float)
        if starting_point is None:
            starting_point = line[0]
        line = line - starting_point
        line = torch.from_numpy(line)
        if time is None:
            time = np.sqrt(np.sum(np.diff(line, axis=0) ** 2, axis=1))
            time = np.insert(time, 0, 0).cumsum()
            time = torch.from_numpy(time)
        coeffs = natural_cubic_spline_coeffs(time, line)
        if track in ['simple', 'complex', 'circle']:
            line = NaturalCubicSpline(coeffs, handle_termination='loop')
        else:
            line = NaturalCubicSpline(coeffs, handle_termination='repeat_end_point')

        lines.append(line)
    center, inner, outer = lines
    return center, inner, outer


def make_track_from_np_array(center_track, name_track, border_length, nb_points):
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
    normals = torch.tensor([[dpos[1]/v, -dpos[0]/v] for dpos, v in zip(dtrack, vs)])
    inner_border = center_track - l/2*normals
    outer_border = center_track + l/2*normals

    track = np.concatenate((center_track.numpy(), inner_border.numpy(), outer_border.numpy()), axis=1)
    dir_name = os.path.dirname(os.path.abspath(__file__))
    pd.DataFrame(track, columns=['X', 'Y', 'X_i', 'Y_i', 'X_o', 'Y_o']).to_json(os.path.join(dir_name, name_track)
                                                                                + '.json', orient='columns', indent=1)


def make_line(border_length=0.37, nb_points=100):
    # border length chosen such that it matches the one of the complex circuit after rescaling
    x_center = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0])
    y_center = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    center_track = np.stack((x_center, y_center)).transpose()/172

    make_track_from_np_array(center_track, 'line_track', border_length, nb_points)


def make_bend(border_length=0.37, nb_points=100):
    R = 50.
    short = 50
    pts_bend = math.ceil(R * math.pi / 2)

    cx = []
    cy = []
    # border length chosen such that it matches the one of the complex circuit after rescaling

    # 1st straight line
    cx = cx + [float(t) for t in range(short + 1)]
    cy = cy + [0. for t in range(short + 1)]

    # 1st bend
    cx = cx + [cx[-1] + R * math.cos(-math.pi / 2 + t / pts_bend * math.pi / 2) for t in range(1, pts_bend + 1)]
    cy = cy + [cy[-1] + R + R * math.sin(-math.pi / 2 + t / pts_bend * math.pi / 2) for t in range(1, pts_bend + 1)]

    # 2nd straight line
    cx = cx + [cx[-1] for t in range(1, short + 1)]
    cy = cy + [cy[-1] + t for t in range(1, short + 1)]

    # x_center = np.array([0.0, 50, 60.0, 70.0, 80.0, 90.0, 100.0])
    # y_center = np.array([0.0, 0.0, 2., 8., 18., 32., 50.])
    # center_track = np.stack((x_center, y_center)).transpose()/172
    center_track = np.array([cx, cy]).transpose()/172

    make_track_from_np_array(center_track, 'bend_track', border_length, nb_points)


def make_circle(border_length=0.37, nb_points=100):
    R = 50.
    pts_bend = math.ceil(R * math.pi / 2)
    cx = [0.]
    cy = [0.]

    cx = cx + [cx[-1] + R * math.cos(-math.pi / 2 + t / pts_bend * math.pi / 2) for t in range(1, 4*pts_bend + 1)]
    cy = cy + [cy[-1] + R + R * math.sin(-math.pi / 2 + t / pts_bend * math.pi / 2) for t in range(1, 4*pts_bend + 1)]

    center_track = np.array([cx, cy]).transpose()/172

    make_track_from_np_array(center_track, 'circle_track', border_length, nb_points)


def make_simple_track(border_length=0.37, nb_points=100):
    # border length chosen such that it matches the one of the complex circuit after rescaling
    R = 50.
    # points per bend
    pts_bend = math.ceil(R * math.pi / 2)

    long = 100
    short = 50
    cx = []
    cy = []
    # 1st straight line
    cx = cx + [float(t) for t in range(long + 1)]
    cy = cy + [0. for t in range(long + 1)]

    # 1st bend
    cx = cx + [cx[-1] + R * math.cos(-math.pi / 2 + t / pts_bend * math.pi / 2) for t in range(1, pts_bend + 1)]
    cy = cy + [cy[-1] + R + R * math.sin(-math.pi / 2 + t / pts_bend * math.pi / 2) for t in range(1, pts_bend + 1)]

    # 2nd straight line
    cx = cx + [cx[-1] for t in range(1, long + 1)]
    cy = cy + [cy[-1] + t for t in range(1, long + 1)]

    # 2nd bend
    cx = cx + [cx[-1] - R + R * math.cos(t / pts_bend * math.pi / 2) for t in range(1, pts_bend + 1)]
    cy = cy + [cy[-1] + R * math.sin(t / pts_bend * math.pi / 2) for t in range(1, pts_bend + 1)]

    # 3rd straight line
    cx = cx + [cx[-1] - t for t in range(1, short + 1)]
    cy = cy + [cy[-1] for t in range(1, short + 1)]

    # 4th bend
    cx = cx + [cx[-1] + R * math.cos(math.pi / 2 + t / pts_bend * math.pi / 2) for t in range(1, pts_bend + 1)]
    cy = cy + [cy[-1] - R + R * math.sin(math.pi / 2 + t / pts_bend * math.pi / 2) for t in range(1, pts_bend + 1)]

    # 5th bend
    cx = cx + [cx[-1] + -R + R * math.cos(-t / pts_bend * math.pi / 2) for t in range(1, pts_bend + 1)]
    cy = cy + [cy[-1] + R * math.sin(-t / pts_bend * math.pi / 2) for t in range(1, pts_bend + 1)]

    # 5th straight line
    cx = cx + [cx[-1] - t for t in range(1, long + 1)]
    cy = cy + [cy[-1] for t in range(1, long + 1)]

    # 6th bend
    cx = cx + [cx[-1] + R * math.cos(math.pi / 2 + t / pts_bend * math.pi / 2) for t in range(1, 2 * pts_bend + 1)]
    cy = cy + [cy[-1] - R + R * math.sin(math.pi / 2 + t / pts_bend * math.pi / 2) for t in range(1, 2 * pts_bend + 1)]

    cx = cx + [cx[-1] + t for t in range(1, long + short + 1)]
    cy = cy + [cy[-1] for t in range(1, long + short + 1)]
    center_track = np.array([cx, cy]).transpose()/172

    make_track_from_np_array(center_track, 'simple_track', border_length, nb_points)





# dir_name = os.path.dirname(os.path.abspath(__file__))
# tracks_data = pd.read_json(os.path.join(dir_name, "simple_track.json"))
# time = None
# tracks = []
# for coord in ['', 'i', 'o']:
#     track = tracks_data[['X' + coord, 'Y' + coord]].to_numpy(dtype=float)
#     track = torch.from_numpy(track)
#     if time is None:
#         time = np.sqrt(np.sum(np.diff(track, axis=0) ** 2, axis=1))
#         time = np.insert(time, 0, 0).cumsum()
#         time = torch.from_numpy(time)
#     coeffs = natural_cubic_spline_coeffs(time, track)
#     track = NaturalCubicSpline(coeffs)
#     tracks.append(track)
# center, inner, outer = tracks
#
# from matplotlib import pyplot as plt
# s = torch.linspace(0, max(center._t), 1000)
# plt.figure()
# track_plots = []
# for track in [center, inner, outer]:
#     track_plot = track.evaluate(s)
#     track_plots.append(track_plot)
#     plt.plot(track_plot[:, 0], track_plot[:, 1], 'k-')
# plt.title('Track')
# plt.show()
# plt.figure()
# center_track = track_plots[0]
# for track_plot in track_plots[1:]:
#     dist_tracks = [torch.norm(a-b)-5 for a, b in zip(center_track, track_plot)]
#     plt.plot(dist_tracks)
#     print(sum(dist_tracks))
#     plt.show()