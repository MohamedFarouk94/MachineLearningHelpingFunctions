import pandas as pd
from IPython.display import display


# Information printing function
def print_info(df, unique_threshold=20):
    info_dict = {'Column': [], 'nan_values': [], 'Type': [], 'n_unique': [], 'values': [],
                 'Min': [], 'Max': [], 'Mean': [], 'STD': []}
    for col in df.columns:
        info_dict['Column'].append(col)

        nan_values = df[col].shape[0] - df[col].dropna().shape[0]
        info_dict['nan_values'].append(nan_values)

        col_type = df[col].dtype
        info_dict['Type'].append(col_type)

        n_unique = df[col].nunique()
        info_dict['n_unique'].append(n_unique)

        values = 'too many' if n_unique > unique_threshold else list(df[col].unique())
        info_dict['values'].append(values)

        min_value = df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A'
        max_value = df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A'
        mean_value = df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A'
        std_value = df[col].std() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A'

        info_dict['Min'].append(min_value)
        info_dict['Max'].append(max_value)
        info_dict['Mean'].append(mean_value)
        info_dict['STD'].append(std_value)

    info_dict['Total'] = [df.shape[0]] * df.shape[1]
    display(pd.DataFrame(info_dict))
