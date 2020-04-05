import numpy as np
import matplotlib.pyplot as plt


class LivePlot:
    """
    Renders a plot that keeps updating as more data becomes available.
    """
    def __init__(self, figures):
        fig, axes = plt.subplots(len(figures), 1)
        self.fig = fig
        self.figures = figures
        if len(figures) == 1:
            self.axes = [axes]
        else:
            self.axes = axes
        self.fig.show()
        self.x_range = 2001
        self.plot_lines = [[None for _ in figure["plots"]] for figure in figures]

    def update_plot(self):
        for figure_idx, (figure, ax) in enumerate(zip(self.figures, self.axes)):
            stats = figure["source"]
            # min limits of y axis for all plots in this figure
            min_y = float("inf")
            max_y = float("-inf")
            for plot in figure["plots"]:
                stat = plot["stat"]
                min_y = min(min_y, stats.min[stat])
                max_y = max(max_y, stats.max[stat])
            padding_y = (max_y - min_y) * 0.05
            min_y = min_y - padding_y
            max_y = max_y + padding_y

            min_x = max(0, stats.upper_bound - self.x_range)
            max_x = stats.upper_bound
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)

            # draw all plots in this figure
            for plot_idx, plot in enumerate(figure["plots"]):
                stat = plot["stat"]
                if self.plot_lines[figure_idx][plot_idx]:
                    self.plot_lines[figure_idx][plot_idx].set_xdata(np.arange(stats.lower_bound, stats.upper_bound + 1))
                    self.plot_lines[figure_idx][plot_idx].set_ydata(stats.data[stat])
                else:
                    self.plot_lines[figure_idx][plot_idx] = ax.plot(stats.data[stat], color=plot.get("color", "b"))[0]
        self.fig.canvas.draw()