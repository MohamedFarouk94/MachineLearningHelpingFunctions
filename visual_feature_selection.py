import matplotlib.pyplot as plt
import numpy as np
from math import sqrt


# Plotting visual approach for categorical feature selection
def plot_cat_selection(df, cols, target, cols_per_plot=20):
    n_rows = int(len(cols) / cols_per_plot) + bool(len(cols) % cols_per_plot)
    plt.rcParams['figure.figsize'] = (cols_per_plot, 16 * n_rows)
    global_average = df[df[target] == 1].shape[0] / df.shape[0]
    plt.subplots(n_rows, 1)

    for i in range(n_rows):
        cols_collection = cols[i * cols_per_plot: (i + 1) * cols_per_plot]
        mins, maxes, min_class, max_class = [], [], [], []
        plt.subplot(n_rows, 1, i + 1)
        for col in cols_collection:
            classes = list(df[col].unique())
            classes_ratios = [(cl, df.groupby(col).get_group(cl)[target].mean()) for cl in classes]
            classes_ratios = sorted(classes_ratios, key=lambda x: x[1])
            mins.append(classes_ratios[0][1])
            maxes.append(classes_ratios[-1][1])
            min_class.append(classes_ratios[0][0])
            max_class.append(classes_ratios[-1][0])

        rectangles = plt.bar(cols_collection, maxes, color='b', label='Class with Maximum Target Ratio')
        plt.bar(cols_collection, mins, color='r', label='Class with Minimum Target Ratio')
        plt.axhline(y=global_average, color='g', linewidth=4, label='Dummy Ratio')

        for rect, min_cl, max_cl in zip(rectangles, min_class, max_class):
            y_rect = rect.get_height()
            x_rect = rect.get_x()
            w_rect = rect.get_width()
            plt.text(x_rect + w_rect * 0.2, y_rect + 0.002, f'Max:{max_cl}\nMin:{min_cl}')

        plt.xticks(rotation=90)
        plt.legend()
    plt.show()


# Plotting visual approach for numerical feature selection
def plot_num_selection(df, cols, target, rolling_length=1000, graph_capacity=6):
    n_cols = len(cols)
    n_graphs = int(n_cols // graph_capacity) + bool(n_cols % graph_capacity)
    n_graph_sqrt = int(sqrt(n_graphs) + 1)
    plt.rcParams['figure.figsize'] = (5 + 4 * n_graph_sqrt, 5 + 4 * n_graph_sqrt)
    noise = 'GeneratedRandomValues'
    colors = ['r', 'g', 'b', 'y', 'k', 'c', 'm']
    df_copy = df[cols + [target]].copy()
    df_copy[noise] = np.random.rand(df.shape[0])
    plt.subplots(n_graph_sqrt, n_graph_sqrt)

    for i in range(n_graphs):
        cols_collection = [noise] + cols[i * graph_capacity: (i + 1) * graph_capacity]
        plt.subplot(n_graph_sqrt, n_graph_sqrt, i + 1)

        for column, color in zip(cols_collection, colors):
            df_temp = df_copy[[column, target]].copy().sort_values(by=column).reset_index().drop('index', axis=1)
            plt.scatter(df_temp.index, df_temp[target].rolling(rolling_length).mean(), color=color, label=column)

        plt.legend()
    plt.show()
