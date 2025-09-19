from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns
from scipy.interpolate import interpn
from scipy.stats import linregress


class get_plots_global :
    def __init__(self, save_dir, plot_set, model_name) :
        self.save_dir = Path(save_dir).resolve()
        if self.save_dir.exists() :
            shutil.rmtree(self.save_dir)
        self.save_dir.mkdir(parents=True)
        
        self.plot_set = plot_set
        self.model_name = model_name

    def __call__(self, metrics_global):
        for plot_name_base, plot in self.plot_set.items():
            fig = plt.figure(figsize=(plot.size_plot_width, plot.size_plot_height))
            plot.create_plot(metrics_global)
            plot_path = self.save_dir / f"{plot_name_base}_{self.model_name}_plot.jpg"
            fig.savefig(
                plot_path, bbox_inches="tight", pad_inches=0
            )
            plt.close()

class plot_model :
    def __init__(self, graph_list, nb_row, nb_col, size_plot_width, size_plot_height):
        self.graph_list = graph_list
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.size_plot_width = size_plot_width
        self.size_plot_height = size_plot_height
    
    def create_plot(self, metrics_global):
        for graph in self.graph_list :
            ax = plt.subplot2grid((self.nb_row, self.nb_col), (graph.idx_row, graph.idx_col), rowspan=graph.rowspan, colspan=graph.colspan)
            graph.create_graph(metrics_global, ax)


        
class graph_model :
    def __init__(self, idx_row, idx_col, graph_title, method_graph, rowspan= 1, colspan = 1):
        self.idx_row = idx_row
        self.idx_col = idx_col
        self.graph_title = graph_title
        self.method_graph = method_graph
        self.rowspan = rowspan
        self.colspan= colspan

    def create_graph(self, metrics_global, ax):
        self.method_graph(metrics_global, ax)
        if self.graph_title :
            graph_title = self.graph_title
            ax.set_title(graph_title, fontsize=16)


class method_bar :
    def __init__(self, metrics_list, y_min, y_max):
        self.metrics_list = metrics_list
        self.y_min = y_min
        self.y_max = y_max

    def __call__(self, metrics_global, ax) :
        for key in self.metrics_list :
            if key not in metrics_global :
                Exception(f"You must first compute metric {key}")
        ax.bar(self.metrics_list, [metrics_global[key] for key in self.metrics_list])
        ax.set_ylim(self.y_min, self.y_max)
        ax.grid(color="gray", linestyle="dashed", axis="y")
        plt.xticks(rotation=40)


class method_table :
    def __init__(self, metrics_list, font_size, table_width_scale, table_height_scale):
        self.metrics_list = metrics_list
        self.font_size = font_size
        self.table_width_scale = 0.7
        self.table_height_scale = 1.8

    def __call__(self, metrics_global, ax) :
        for key in self.metrics_list :
            if key not in metrics_global :
                Exception(f"You must first compute metric {key}")
        
        # Create table data with metric names and values
        table_data = []
        for metric_name in self.metrics_list:
            value = metrics_global[metric_name]
            # Format the value to 3 decimal places if it's a float
            if isinstance(value, float):
                formatted_value = f"{value:.3f}"
            else:
                formatted_value = str(value)
            table_data.append([metric_name, formatted_value])
        
        # Create the table
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='center',
                        loc='center')
        table.set_fontsize(self.font_size)
        table.scale(self.table_width_scale, self.table_height_scale)
        
        ax.axis('off')


class method_scatter_density:
    def __init__(
        self, 
        metric_name, 
        x_label, 
        y_label, 
        min_range,
        max_range,
        bins, 
        metrics_plot,
        max_points_on_scatter,
    ):
        
        self.metric_name = metric_name
        self.x_label = x_label
        self.y_label = y_label
        self.range_fig = (min_range, max_range)
        self.bins = bins
        self.metrics_plot = metrics_plot
        self.max_points_on_scatter = max_points_on_scatter

    def __call__(self, metrics_global, ax):
        if self.metric_name not in metrics_global:
            raise Exception(f"Metric '{self.metric_name}' not found in metrics_global.")
        
        y, x = metrics_global[self.metric_name]
        
        sns.set_style("ticks")
        ax.set_xlim(self.range_fig)
        ax.set_ylim(self.range_fig)
        ax.set_aspect('equal', adjustable='box')

        # Ensure input is a numpy array
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Remove NaNs
        valid_indices = ~np.isnan(x) & ~np.isnan(y)
        x = x[valid_indices]
        y = y[valid_indices]
        
        if len(x) == 0 or len(y) == 0:
            return # No data to plot

        # Calculate density using a 2D histogram
        hist, x_edges, y_edges = np.histogram2d(
            x, y, bins=self.bins, range=[self.range_fig, self.range_fig], density=True
        )
        z = interpn(
            (x_edges[:-1], y_edges[:-1]),
            hist,
            np.vstack([x, y]).T,
            method="splinef2d",
            bounds_error=False,
            fill_value=0,
        )
        z = np.nan_to_num(z)  # Replace NaNs with 0
        idx = z.argsort()  # Sort points by density
        x, y, z = x[idx], y[idx], z[idx]

        norm = colors.LogNorm(
            vmin=z[z > 0].min(),
            vmax=z.max(),
            clip=True,
        )
        
    
        # Perform linear regression
        slope, intercept, _, _, _ = linregress(x, y)

        # Generate the fit line
        fit_line = slope * np.array(self.range_fig) + intercept
        
        if self.max_points_on_scatter is not None:
            idx = np.random.randint(len(x), size=self.max_points_on_scatter)
            x_scatter = x[idx]
            y_scatter = y[idx]
            z_scatter = z[idx]
        else :
            x_scatter = x
            y_scatter = y
            z_scatter = z

        ax.scatter(
            x_scatter,
            y_scatter,
            c=z_scatter,
            s=1,
            cmap="magma",
            norm=norm,
            edgecolor="none",
        )

        # Draw the x=y line
        ax.plot(
            self.range_fig,
            self.range_fig,
            color="black",
            linestyle="dashed",
            linewidth=1,
            label="1:1",
        )

        # Plot the fit line
        ax.plot(
            self.range_fig,
            fit_line,
            color="blue",
            linestyle="dotted",
            linewidth=1.5,
            label="Linear fit",
        )

        ax.legend(loc="upper left")
        # Add axis labels
        if self.x_label:
            ax.set_xlabel(self.x_label, fontsize=10)
        if self.y_label:
            ax.set_ylabel(self.y_label, fontsize=10)

        # Prepare metrics for display
        metrics = []
        if "mae" in self.metrics_plot:
            mae = mean_absolute_error(x, y)
            metrics.append(f"MAE = {mae:.2f} m")
        if "me" in self.metrics_plot:
            me = np.mean(y - x)
            metrics.append(f"ME = {me:.2f} m")
        if "mape" in self.metrics_plot:
            safe_mask = x != 0  # in case x=0
            mape = np.mean(np.abs((y[safe_mask] - x[safe_mask]) / x[safe_mask])) * 100
            metrics.append(f"MAPE = {mape:.2f}%")
        if "r2" in self.metrics_plot:
            r2 = r2_score(x, y)
            metrics.append(f"R² = {r2:.2f}")
        if "r2_pearson" in self.metrics_plot:
            r2_p = np.corrcoef(x, y)[0, 1] ** 2
            metrics.append(f"r² = {r2_p:.2f}")

        if metrics:
            ax.text(
                self.range_fig[1] - 0.04 * (self.range_fig[1] - self.range_fig[0]),
                self.range_fig[0] + 0.07 * (self.range_fig[1] - self.range_fig[0]),
                "\n".join(metrics),
                fontsize=10,
                ha="right",
                backgroundcolor="white",
            )

class method_boxplot_by_bins:
    def __init__(self, metric_name, bins, x_label, y_label, y_min=None, y_max=None):
        self.metric_name = metric_name
        self.bins = bins
        self.x_label = x_label
        self.y_label = y_label
        self.y_min = y_min
        self.y_max = y_max


    def __call__(self, metrics_global, ax):
                
        if self.metric_name not in metrics_global:
            raise Exception(f"Metric '{self.metric_name}' not found.")
            
        data_to_plot = metrics_global[self.metric_name]
        labels = self.bins

        # Create boxplot
        ax.boxplot(data_to_plot, showfliers=False)
        
        ax.set_xticklabels(labels, rotation=40, ha="right")
        
        ax.set_xlabel(self.x_label, fontsize=14)
        ax.set_ylabel(self.y_label, fontsize=14)
        if self.y_min is not None and self.y_max is not None:
            ax.set_ylim(self.y_min, self.y_max)
        ax.grid(color="gray", linestyle="dashed", axis="y")
        
        # Add proportion histogram on secondary y-axis
        ax2 = ax.twinx()
        
        mean = [np.mean(data) for data in data_to_plot]
        
        # Calculate proportions (number of values in each bin)
        proportions = [len(data) for data in data_to_plot]
        total_values = sum(proportions)
        proportions_percentage = [prop / total_values * 100 for prop in proportions]
        
        # Plot proportions as histogram bars
        x_positions = range(1, len(proportions) + 1)
        ax2.bar(x_positions, proportions_percentage, alpha=0.3, color='red', width=0.6)
        
        ax2.set_ylabel('Proportion (%)', fontsize=14, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, max(proportions_percentage) * 1.1)


