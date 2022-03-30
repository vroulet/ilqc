import torch
from matplotlib import pyplot as plt
from envs.utils.tracks import get_track, make_line, make_bend, make_circle, make_simple_track
from envs.utils.car_visualization import plot_track
from copy import deepcopy

make_line()
make_bend()
make_circle()
make_simple_track()

# Get a spline approx. of the track
center, inner, outer = get_track('simple')
fig, axs = plot_track(center, inner, outer)
plt.show()

# Plot distance ot inner border
fig, ax = plt.subplots()
time = torch.linspace(0, 1 * max(center._t), 500)

track_plots = []
for track in [center, inner, outer]:
    track_plot = track.evaluate(time)
    track_plots.append(track_plot)
dist_tracks = [torch.norm(a-b) for a, b in zip(track_plots[0], track_plots[1])]
plt.plot(dist_tracks)
plt.title('Distance to inner border')
plt.show()

# Get the derivatives along the spline to deduce the yaw
dtrack = center.derivative(time)
yaw_ref = torch.atan2(dtrack[:, 1], dtrack[:, 0])
fig, axs = plt.subplots(1, 2, squeeze=False, figsize=(8, 4))
axs[0,0].plot(yaw_ref)
axs[0,0].title.set_text('Yaw')
# Compute the acceleration of the yaw (and check btw if they are well computed)
dyaws = []
for t in time:
    s = deepcopy(t)
    s.requires_grad = True
    dpos = center.derivative(s)
    yaw = torch.atan2(dpos[1], dpos[0])
    dyaw = torch.autograd.grad(yaw, s)[0]
    dyaws.append(dyaw.item())
axs[0,1].plot(dyaws)
axs[0,1].title.set_text('Yaw rate')
plt.show()



