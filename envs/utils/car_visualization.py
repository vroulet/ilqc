import torch
import math
import numpy as np
import colorsys
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection


light_green, orange, light_gray, dark_gray, dark_red = [[144 / 255, 238 / 255, 144 / 255],
                                                        [204 / 255, 85 / 255, 0.],
                                                        [119 / 255, 136 / 255, 153 / 255],
                                                        [47 / 255, 79 / 255, 79 / 255],
                                                        [.8, .3, .3]]
pastel_light_green, pastel_light_gray = [list(colorsys.rgb_to_hls(*color)) for color in [light_green, light_gray]]
pastel_light_green[1] = 0.8
pastel_light_gray[1] = 0.85
pastel_light_green, pastel_light_gray = [list(colorsys.hls_to_rgb(*color))
                                         for color in [pastel_light_green, pastel_light_gray]]

#########################################
# Viewer functionalities
#########################################


def set_window_viewer(min_x, max_x, min_y, max_y, border):
    from envs.utils import rendering

    width, height = max_x - min_x + 2*border, max_y - min_y + 2*border
    if width > height:
        viewer = rendering.Viewer(800., 800. * height / width, bg_color=pastel_light_green + [1.])
    else:
        viewer = rendering.Viewer(800. * width / height, 800., bg_color=pastel_light_green + [1.])
    viewer.set_bounds(min_x - border, max_x + border,
                      min_y - border, max_y + border)
    return viewer


def add_track_to_viewer(viewer, track, inner_track, outer_track, nb_points=500):
    from envs.utils import rendering
    time = torch.linspace(0, max(track._t), nb_points)

    # Fill up the track
    lines = [track.evaluate(time).numpy() for track in [track, inner_track, outer_track]]
    for i in range(len(lines[1]) - 1):
        filled_track = rendering.make_polygon([lines[1][i], lines[1][i + 1], lines[2][i]])
        filled_track.set_color(*pastel_light_gray)
        viewer.add_geom(filled_track)
        filled_track = rendering.make_polygon([lines[1][i + 1], lines[2][i], lines[2][i + 1]])
        filled_track.set_color(*pastel_light_gray)
        viewer.add_geom(filled_track)

    # Plot center line, inner and outer borders
    for i, line in enumerate(lines):
        track_vis = rendering.make_polyline(line)
        if i == 0:
            track_vis.set_linewidth(3)
            track_vis.set_color(*orange)
            track_vis.add_attr(rendering.LineStyle(0x00FF))
        else:
            track_vis.set_linewidth(10)
            track_vis.set_color(*dark_gray)
        viewer.add_geom(track_vis)


def add_obstacle_to_viewer(viewer, point, radius):
    from envs.utils import rendering

    obstacle = rendering.make_circle(radius=math.sqrt(radius))
    translate = rendering.Transform()
    obstacle.add_attr(translate)
    viewer.add_geom(obstacle)
    translate.set_translation(*point)


def add_car_to_viewer(viewer, carlength, carwidth):
    from envs.utils import rendering

    cartrans = rendering.Transform()

    l, r, t, b = [-carlength / 2, carlength / 2, carwidth / 2, -carwidth / 2]
    lw, rw, tw, bw = [ i /3 for i in [l, r, t, b]]
    for wheel_center in [(l, b), (l, t), (r, t), (r, b)]:
        x, y = wheel_center
        x, y = x/ 1.3, y / 1.1
        wheel = rendering.FilledPolygon([(x + lw, y + bw), (x + lw, y + tw), (x + rw, y + tw), (x + rw, y + bw)])
        wheel.add_attr(cartrans)
        viewer.add_geom(wheel)

    car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    car.set_color(*dark_red)
    car.add_attr(cartrans)
    viewer.add_geom(car)
    return cartrans


#########################################
# Plot functionalities
#########################################


def plot_track(track, inner_track, outer_track, nb_loops=1, nb_points=500, fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_facecolor(color=pastel_light_green)
    time = torch.linspace(0, nb_loops*max(track._t), nb_points)
    lines = [track.evaluate(time).numpy() for track in [track, inner_track, outer_track]]

    patches = []
    # Fill track with light gray
    for i in range(len(lines[1])-1):
        polygon = Polygon([lines[1][i], lines[1][i + 1], lines[2][i]], color=pastel_light_gray)
        patches.append(polygon)
        polygon = Polygon([lines[1][i + 1], lines[2][i], lines[2][i + 1]], color=pastel_light_gray)
        patches.append(polygon)
    p = PatchCollection(patches, match_original=True, zorder=2)
    ax.add_collection(p)

    # Plot center track, inner vorder and outer border
    for i, line in enumerate(lines):
        if i == 0:
            ax.plot(line[:, 0], line[:, 1], '--', color=orange, linewidth=1, zorder=3)
        else:
            ax.plot(line[:, 0], line[:, 1], '-', color=dark_gray, linewidth=3, zorder=4)
    ax.axis('equal')
    ax.axis('off')
    ax.add_artist(ax.patch)
    return fig, ax


def plot_traj(traj, fig, ax, model, add_colorbar=True):
    # Plot starting point
    plt.plot(traj[0][0].item(), traj[0][1].item(), 'k+', markersize=10)

    # Gather traj
    xs = [state[0].item() for state in traj]
    ys = [state[1].item() for state in traj]

    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Gather speeds
    if model == 'simple':
        vs = [state[3].item() for state in traj]
    elif model == 'real':
        vs = [np.sqrt(state[3].item()**2 + state[4].item()**2) for state in traj]
    else:
        raise NotImplementedError

    # Plot traj with speeds
    cmap = plt.get_cmap('viridis')
    min_v, max_v = 1., 3.
    colors = []
    for i in range(len(vs)-1):
        v = (vs[i] + vs[i+1])/2
        idx = (v - min_v) / (max_v - min_v)
        idx = int(idx * cmap.N)
        colors.append(cmap(idx))
    lc = LineCollection(segments, colors=colors, linewidths=3, cmap='viridis', norm=plt.Normalize(min_v, max_v),
                        zorder=4)
    ax.add_collection(lc)
    if add_colorbar:
        axcb = fig.colorbar(lc)
        axcb.set_label('Speed m/s')

    # if self.qO > 0.:
    #     plt.plot(self.obstacle[0], self.obstacle[1], 'ko')
    # ax.patch.set_facecolor(color='green')
    return fig
