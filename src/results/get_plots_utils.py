import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

class get_plots :
    def __init__(self, save_dir, plot_set, nb_plots) :
        self.save_dir = save_dir
        if os.path.exists(self.save_dir) :
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)
        
        self.plot_set = plot_set
        self.nb_plots = nb_plots

    def __call__(self, images, metrics, idx_plot):
        for plot_name, plot in self.plot_set.items():
            fig = plt.figure(figsize=(plot.size_plot, plot.size_plot))
            plot.create_plot(images, metrics)
            fig.savefig(
                os.path.join(self.save_dir, plot_name + f"_{idx_plot}_plot.jpg"), bbox_inches="tight", pad_inches=0
            )
            plt.close()

class plot_model :
    def __init__(self, graph_list, nb_row, nb_col, size_plot = 10):
        self.graph_list = graph_list
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.size_plot = size_plot
    
    def create_plot(self, images, metrics):
        for graph in self.graph_list :
            ax = plt.subplot2grid((self.nb_row, self.nb_col), (graph.idx_row, graph.idx_col), rowspan=graph.rowspan, colspan=graph.colspan)
            graph.create_graph(images, metrics, ax)

        
class graph_model :
    def __init__(self, idx_row, idx_col, graph_title, method_graph, rowspan= 1, colspan = 1):
        self.idx_row = idx_row
        self.idx_col = idx_col
        self.graph_title = graph_title
        self.method_graph = method_graph
        self.rowspan = rowspan
        self.colspan= colspan

    def create_graph(self, images, metrics, ax):
        self.method_graph(images, metrics, ax)
        if self.graph_title :
            ax.set_title(self.graph_title, fontsize=16)


class method_imshow :
    def __init__(self, image_name, cmap=None, norm=None, channels_to_keep= None, patch_size_percentage=None):
        self.image_name = image_name
        self.cmap = cmap
        self.norm = norm
        self.channels_to_keep = channels_to_keep
        self.patch_size_percentage = patch_size_percentage 

    def __call__(self, images, metrics, ax):
        if self.image_name not in images :
            Exception(f"You must first load {self.image_name}")
        image = images[self.image_name]
        
        if self.patch_size_percentage :
            height, width = image.shape[-2], image.shape[-1]
            x_start = int(height*(1-self.patch_size_percentage)/2)
            x_stop = int(x_start + height*self.patch_size_percentage)
            y_start = int(width*(1-self.patch_size_percentage)/2)
            y_stop = int(x_start + width*self.patch_size_percentage)

            if len(image.shape) > 2 :
                image = image[..., x_start:x_stop, y_start:y_stop ]
            else :
                image = image[x_start:x_stop, y_start:y_stop ]
            
        if self.channels_to_keep is not None :
            image = np.take(image, indices=self.channels_to_keep, axis=0)
        
        if len(image.shape) == 3 :
            image = np.transpose(image, (1, 2, 0))
            image = (image - image.min()) / (image.max() - image.min())
            gain = 0.4/image.mean()
            image = np.clip(image*gain, 0, 1)
        ax.imshow(image, cmap=self.cmap, norm=self.norm)
        ax.axis("off")
        


class method_bar :
    def __init__(self, metrics_list, y_min, y_max):
        self.metrics_list = metrics_list
        self.y_min = y_min
        self.y_max = y_max

    def __call__(self, images, metrics, ax) :
        for key in self.metrics_list :
            if key not in metrics :
                Exception(f"You must first compute metric {key}")
        ax.bar(self.metrics_list, [metrics[key] for key in self.metrics_list])
        ax.set_ylim(self.y_min, self.y_max)
        ax.grid(color="gray", linestyle="dashed", axis="y")
        plt.xticks(rotation=40)