from matplotlib import pyplot as plt
import seaborn as sns
from copy import deepcopy
import colorsys


def set_plt_params():
    params = {'axes.labelsize': 28,
              'font.size': 26,
              'legend.fontsize': 24,
              'xtick.labelsize': 28,
              'ytick.labelsize': 28,
              'lines.linewidth': 5,
              'lines.markersize': 20,
              'text.usetex': True,
              }
    plt.rcParams.update(params)


def get_nice_writing(alternative=False):
    nice_writing = dict(iteration='Iterations', cost='Cost', stepsize='Stepsize', time='Time',
                        pendulum='Swinging up Pendulum',
                        cart_pendulum='Swinging up Pendulum on a Cart',
                        simple_car='Simple Model of Car with Tracking Cost',
                        real_car='Bicycle Model of Car with Contouring Cost',
                        gd='GD',
                        )
    algo_names = dict(classic_linquad='GN', classic_quad='NE', ddp_linquad='DDP-LQ', ddp_quad='DDP-Q')
    step_mode_names = dict(dir='dir', reg='reg', dirvar='dir var', regvar='reg var')
    nice_writing.update({algo + '_' + step_mode: algo_name + ' ' + step_mode_name
                         for (algo, algo_name) in algo_names.items()
                         for (step_mode, step_mode_name) in step_mode_names.items()})
    if alternative:
        nice_writing.update(classic_linquad_reg='ILQR', ddp_linquad_reg='IDDP')
        nice_writing.update(simple_car='Simple Model of Car\nwith Tracking Cost')
        nice_writing.update(real_car='Bicycle Model of Car\nwith Tracking Cost')
        nice_writing.update(pendulum='Swinging up Pendulum\n')

    return nice_writing


def cmap_linestyles_ref():
    cmap = sns.color_palette("colorblind")
    # line_styles_list = [(0, (1, 5)), (0, (1, 1)),
    #                     (0, (5, 5)), (0, (5, 1)),
    #                     (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)),
    #                     (0, (3, 5, 1, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1))]
    marker_styles_list = ['v', '^', 's', 'D', 'P', 'X', 'p', '*']
    return cmap, marker_styles_list


def get_palette_line_styles():
    algos_refs = [algo_type + '_' + approx + '_' + step_mode_name
                  for approx in ['linquad', 'quad']
                  for algo_type in ['classic', 'ddp']
                  for step_mode_name in ['dir', 'reg']
                  ]
    cmap, marker_styles_list = cmap_linestyles_ref()
    palette = {key: value for key, value in zip(algos_refs, cmap)}
    palette.update({'gd': cmap[len(palette)]})
    # line_styles = {key: value for key, value in zip(algos_refs, line_styles_list)}
    # line_styles.update({'gd': (0, ())})
    marker_styles = {key: value for key, value in zip(algos_refs, marker_styles_list)}
    marker_styles.update({'gd': 'o'})
    return palette, marker_styles


# Variation on the stepsizes search
def get_var_palette_line_styles():
    var_algos_refs = [algo_type + '_' + approx + '_' + step_mode_name
                      for algo_type in ['classic', 'ddp']
                      for approx in ['linquad', 'quad']
                      for step_mode_name in ['dirvar', 'regvar']]

    cmap, marker_styles_list = cmap_linestyles_ref()

    var_palette, var_marker_styles = get_palette_line_styles()
    var_palette.update({var_algos_refs[i]: cmap[(i+2)%len(var_algos_refs)]
                        for i in range(len(var_algos_refs))})
    # var_line_styles.update({var_algos_refs[i]: line_styles_list[(i+2)%len(var_algos_refs)]
    #                         for i in range(len(var_algos_refs))})
    var_marker_styles.update({var_algos_refs[i]: marker_styles_list[(i+2)%len(var_algos_refs)]
                            for i in range(len(var_algos_refs))})
    return var_palette, var_marker_styles


def set_linestyles(ax, algos, line_styles):
    for line, algo in zip(ax.lines, algos):
        line.set_linestyle(line_styles[algo])


def nice_ax(ax, x_axis, y_axis, logscale):
    nice_writing = get_nice_writing()
    handles, labels = ax.get_legend_handles_labels()
    for i in range(len(labels)):
        labels[i] = nice_writing[labels[i]]
        handles[i] = ax.lines[i]
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_ylabel(nice_writing[y_axis])
    if logscale:
        ax.set(yscale='log')
    return handles, labels


def format_plot(fig, title, add_legend=False):
    axes = fig.axes
    nice_writing = get_nice_writing()
    for i, ax in enumerate(axes):
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        x_axis = ax.xaxis.get_label().get_text()
        ax.set_xlabel(nice_writing[x_axis])
        if i == 0:
            y_axis = ax.yaxis.get_label().get_text()
            ax.set_ylabel(nice_writing[y_axis])
        else:
            ax.set(ylabel=None)
        ax.set(yscale='log')

    fig.suptitle(title, y=1.)
    fig.tight_layout()
    if add_legend:
        handles, labels = axes[0].get_legend_handles_labels()
        labels = [nice_writing[label] for label in labels]
        fig.legend(handles=handles, labels=labels, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=len(handles))


def plot_conv_costs(info_exps, start_plot_after=0):
    nice_writing = get_nice_writing()
    fig = plt.figure()
    if start_plot_after > 0:
        info_exps = info_exps[info_exps['iteration'] >= start_plot_after]
    x_axis = 'iteration'
    y_axis = 'cost'
    ax = sns.lineplot(x='iteration', y='cost', hue='algo', data=info_exps)
    handles, labels = ax.get_legend_handles_labels()
    for i in range(len(labels)):
        labels[i] = nice_writing[labels[i]]
    ax.legend(handles=handles, labels=labels)
    ax.set_xlabel(nice_writing[x_axis])
    ax.set_ylabel(nice_writing[y_axis])
    return fig


