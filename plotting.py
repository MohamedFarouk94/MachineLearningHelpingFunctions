from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns


# Plotting counts
def cat_plots(df, cols, n_rows=1, n_cols=1, target=None):
    plt.rcParams['figure.figsize'] = (5 + 4 * n_rows, 5 + 4 * n_cols)

    if isinstance(cols, str):
        cols = [cols]

    plt.subplots(n_rows, n_cols)
    for i, col in enumerate(cols):
        plt.subplot(n_rows, n_cols, i + 1)
        if target:
            order = list(df[col].unique())
            order = sorted(order)
            hue_order = list(df[target].unique())
            hue_order = sorted(hue_order)
            bar_order = product(order, hue_order)
            catp = sns.countplot(data=df, x=col, hue=target, order=order, hue_order=hue_order)

            bar_order = list(bar_order)
            bar_order0 = [bar for j, bar in enumerate(bar_order) if j % 2 == 0]
            bar_order1 = [bar for j, bar in enumerate(bar_order) if j % 2 == 1]
            bar_order = bar_order0 + bar_order1
            assert len(catp.patches) == len(bar_order), (len(catp.patches), len(bar_order))

            spots = zip(catp.patches, bar_order)
            for spot in spots:
                this_class = spot[1][0]
                this_target = spot[1][1]
                df_by_class = df[df[col] == this_class].reset_index()
                df_target_ratio = df_by_class[df_by_class[target] == this_target]
                ratio = df_target_ratio.shape[0] / df_by_class.shape[0]
                if ratio == 0:
                    continue
                height = spot[0].get_height()
                width = spot[0].get_x()
                catp.text(width, height + 3, f'{ratio:.2f}')
        else:
            ax = sns.countplot(data=df, x=col)
            for patch in ax.patches:
                height = patch.get_height()
                ratio = height / df.shape[0] * 100
                ax.text(patch.get_x() + 0.01, height * 1.01, f'{ratio:.2f}%')
        plt.xticks(rotation=90)
    plt.show()


# Showing plots for numerical data
def num_plots(df, cols, target, n_rows=1, n_cols=1, rolling_length=1000):
    plt.rcParams['figure.figsize'] = (5 + 4 * n_rows, 5 + 4 * n_cols)

    if isinstance(cols, str):
        cols = [cols]

    plt.subplots(n_rows, n_cols)
    for i, col in enumerate(cols):
        plt.subplot(n_rows, n_cols, i + 1)
        df_temp = df[[col, target]].copy()
        df_temp_sorted = df_temp.sort_values(by=col).reset_index()
        plt.scatter(df_temp_sorted.index, df_temp_sorted[target].rolling(rolling_length).mean())
        plt.title(col)
    plt.show()


# Showing correlations heatmap
def show_corr(df, cols, cmap='YlGnBu', fmt='.2f'):
    plt.rcParams['figure.figsize'] = (20, 20)
    sns.heatmap(df[cols].corr(), cmap=cmap, annot=True, fmt=fmt)
    plt.show()


# Showing correlations in bars
def show_target_corr(df, cols, target):
    plt.rcParams['figure.figsize'] = (20, 7)
    corrs = df[cols + [target]].corr()[target]
    plt.bar(cols, corrs[:-1])
    plt.xticks(rotation=90)
    plt.xlabel('columns')
    plt.ylabel('Correlation with target column')
    plt.axhline(y=0, c='k')
    plt.show()
