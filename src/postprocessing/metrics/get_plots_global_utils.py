from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd

class get_plots_global :
    def __init__(self, save_dir, plot_set, model_name) :
        self.save_dir = Path(save_dir).resolve()
        if self.save_dir.exists() :
            shutil.rmtree(self.save_dir)
        self.save_dir.mkdir(parents=True)
        
        self.plot_set = plot_set
        self.model_name = model_name

    def __call__(self, metrics_local):
        for plot_name_base, plot in self.plot_set.items():
            fig = plt.figure(figsize=(plot.size_plot_width, plot.size_plot_height))
            plot.create_plot(None, metrics_local, None)
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
    def __init__(self, metric_name, x_label, y_label, x_min, x_max, y_min, y_max, metrics_to_show=None, bins=100, cmap='inferno'):
        self.metric_name = metric_name
        self.x_label = x_label
        self.y_label = y_label
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.metrics_to_show = metrics_to_show if metrics_to_show is not None else {}
        self.bins = bins
        self.cmap = cmap

    def __call__(self, metrics_global, ax):
        if self.metric_name not in metrics_global:
            Exception(f"You must first compute metric {self.metric_name}")
        # Extract x and y values from tuples in the dataframe
        x, y = metrics_global[self.metric_name]
        # Remove NaNs
        valid_indices = ~np.isnan(x) & ~np.isnan(y)

        x = np.concatenate([arr for arr in x[valid_indices]])
        y = np.concatenate([arr for arr in y[valid_indices]])
        
        if len(x) == 0 or len(y) == 0:
            return # No data to plot

        # Create the 2D histogram
        ax.hist2d(x, y, bins=self.bins, cmap=self.cmap, norm=colors.LogNorm())

        # Add 1:1 line
        line_min = max(self.x_min, self.y_min)
        line_max = min(self.x_max, self.y_max)
        ax.plot([line_min, line_max], [line_min, line_max], 'k--')

        ax.set_xlabel(self.x_label, fontsize=14)
        ax.set_ylabel(self.y_label, fontsize=14)
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect('equal', adjustable='box')

        # Add metrics text
        metrics_text = ""
        for metric_name, unit in self.metrics_to_show.items():
            if metric_name not in metrics_global:
                Exception(f"You must first compute metric {metric_name}")
            
            value = metrics_global[metric_name]
            if isinstance(value, float):
                metrics_text += f"{metric_name} = {value:.2f}{unit}\n"
            else:
                metrics_text += f"{metric_name} = {value}{unit}\n"
        
        if metrics_text:
            ax.text(0.95, 0.1, metrics_text.strip(),
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=ax.transAxes,
                    color='black', fontsize=14)


class method_boxplot_by_bins:
    def __init__(self, metric_name, bins, x_label, y_label, y_min=None, y_max=None):
        self.metric_name = metric_name
        self.bins = bins
        self.x_label = x_label
        self.y_label = y_label
        self.y_min = y_min
        self.y_max = y_max


    def __call__(self, images, metrics, ax, row):
                
        if self.metric_name not in metrics:
            raise Exception(f"Metric '{self.metric_name}' not found.")
            
        data_to_plot = metrics[self.metric_name]
        labels = self.bins

        ax.boxplot(data_to_plot)
        
        ax.set_xticklabels(labels, rotation=40, ha="right")
        
        ax.set_xlabel(self.x_label, fontsize=14)
        ax.set_ylabel(self.y_label, fontsize=14)
        if self.y_min is not None and self.y_max is not None:
            ax.set_ylim(self.y_min, self.y_max)
        ax.grid(color="gray", linestyle="dashed", axis="y")


