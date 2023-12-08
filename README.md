# My Kaggle code for Open Problems – Single-Cell Perturbations competition

This is my codes for the [Open Problems – Single-Cell Perturbations](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations) on Kaggle.

My kaggle profile: [https://www.kaggle.com/masaishi](https://www.kaggle.com/masaishi)

## Highlighted Notebook

**Notebook: [opscp-m-nb019.ipynb](opscp-m-nb019.ipynb)**  
This notebook is my personal favorite. This code convert table data to image format and then use image-based models for training

My one approach is to convert feature-added table data to image data and use PyTorch Image Models (Timm).

### Feature Engineering:
My initial step involved enriching the dataset with calculated statistics for each gene and per 'sm_name' and 'cell_type'. This included mean, standard deviation, minimum, maximum, median, skewness, kurtosis, and mean-to-standard deviation ratios. These new features significantly increased the data dimensions.
```python
def calculate_statistic(df, group_col, cols, stat_func, stat_name):
    """
    Function to calculate statistics for given columns grouped by a specific column.
    """
    stat_df = df.groupby(group_col)[cols].apply(stat_func)
    stat_df.columns = [f'{group_col}_{stat_name}_{col}' for col in cols]
    return stat_df.reset_index().astype({group_col: str})

# Example usage for calculating various statistics for 'cell_type'
cell_type_mean = calculate_statistic(all_de_train, "cell_type", genes, lambda x: x.mean(), 'mean')
cell_type_std = calculate_statistic(all_de_train, "cell_type", genes, lambda x: x.std(), 'std')
# Additional statistics like min, max, median, skew, kurtosis, and ratio_mean_std are also calculated
```

### Converting Data to Image Format:
I converted the data into an image format to handle this massive feature space. This conversion was necesally for applying image-based models.
```python
def convert_to_image_format(data, output_size=(224, 224)):
		square_side = int(math.ceil(math.sqrt(data.shape[1])))
		padding_size = square_side ** 2 - data.shape[1]
		
		data_padded = np.pad(data, ((0, 0), (0, padding_size)), 'constant', constant_values=0)
		
		# We are reshaping to [N, H, W, C] because the final torch tensor needs to be [N, C, H, W]
		data_reshaped = data_padded.reshape(-1, square_side, square_side, 1)
		
		# Expand the last dimension to three channels by repeating the data
		data_rgb = np.repeat(data_reshaped, 3, -1)
		
		return data_rgb
```
![Sample image](output.png)

### Fine-Tuning PyTorch Image Models (timm):
I utilized the PyTorch Image Models library, which offers many pre-trained models. I fine-tuned these models, such as efficientnet_v2_l and regnet_y_800mf.
