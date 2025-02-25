# NER and POS Tagging for TREC Dataset

## Prerequisites
- Python 3.x
- TensorFlow 2.x
- NumPy
- Pandas
- scikit-learn
- Gensim
- Jupyter Notebook or JupyterLab

## Installation
To set up the project environment, run the following commands:
```bash
pip install tensorflow numpy pandas scikit-learn gensim jupyter
```

## Usage
To train the model or make predictions, navigate to the directory containing the Jupyter notebooks and launch Jupyter Notebook:
```bash
cd path/to/notebooks
jupyter notebook
```
Open the `model.ipynb` or `TREC_pretrain.ipynb` notebook and follow the instructions within.

For using the pre-trained models, load them using TensorFlow's `load_model` function:
```python
import tensorflow as tf
model = tf.keras.models.load_model('path/to/model_directory')
```

## Additional Notes
- Adjust the paths in the scripts according to your directory structure.
- The dataset is located in the `TREC_dataset` directory.
- Model training logs can be found in the `model_logs` directory.
- For detailed model architecture, refer to the `model.png` file.
- Ensure that the `.gitignore` file includes all the necessary patterns to avoid uploading large files or sensitive information to version control.
