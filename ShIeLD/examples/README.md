# Inputs to SHIELD

---

## Input Format

Shield requires two main files : 

- A CSV file with marker expression, spatial coordinates, and complementary sample information (see below)

- A requirements.pt file, referencing the CSV file of interest and specifying important hyperparameters of the model. 
SHIELD calls directly on this file for all downstream analyses.

---

## CSV file

Here is an example of how to set up your input CSV : 

| Cell    | Marker 1 | Marker 2 | Marker 3 | Marker 4 | Marker 5 | Marker 6 | Marker 7 | Marker 8 | Marker 9 | X     | Y     | Cell Type   | Condition | Patient | Fold      |
|---------|----------|----------|----------|----------|----------|----------|----------|----------|----------|-------|-------|-------------|-----------|---------|-----------|
| Cell 1  | 1.23     | 0.88     | 2.10     | 1.02     | 0.55     | 3.12     | 1.99     | 0.44     | 2.78     | 120.4 | 305.7 | T cell      | Tumour    | P01     | 0         |
| Cell 2  | 0.92     | 1.10     | 1.78     | 0.87     | 0.66     | 2.45     | 1.65     | 0.39     | 2.11     | 118.9 | 310.2 | B cell      | Normal    | P02     | 1         |
| Cell 3  | 2.34     | 0.76     | 2.44     | 1.55     | 0.48     | 3.55     | 2.10     | 0.50     | 3.01     | 130.1 | 298.4 | Fibroblast  | Tumour    | P01     | 1         |
| Cell 4  | 1.01     | 0.67     | 1.55     | 0.92     | 0.70     | 2.89     | 1.34     | 0.29     | 2.45     | 125.8 | 315.9 | Macrophage  | Normal    | P03     | 2         |
| Cell n  | 1.87     | 1.45     | 2.01     | 1.20     | 0.60     | 3.00     | 1.90     | 0.40     | 2.90     | 140.2 | 290.5 | Endothelial | Tumour    | P04     | 4         |

Important things to note : 

- **X** and **Y** are the **spatial coordinates** of your cells
- The names of these columns **are NOT fixed**, and can be specified in the requirements folder

⸻

## Requirements file

Based on the CSV file, here is an example of how to set up the requirements.pt (see also the .ipynb notebooks for more examples).

```python

RAW_INPUT = 'path/to/the/csv/file.csv'

requirements = {
    "path_raw_data": RAW_INPUT,
    "path_to_data_set": Path.cwd() / 'results' / 'data_set',
    "path_training_results": Path.cwd() / 'results' / 'training_results',
    "path_to_model": Path.cwd() / 'results' / 'model',
    "path_to_interaction_plots": Path.cwd() / 'results' / 'cTc_interactions',

    'cell_type_names': ['T cell',
                        'B cell',
                        'Fibroblast',
                        'Macrophage',
                        'Endothelial'
                        ],

    'markers': ['Marker 1', ...., 'Marker 9'],

    'label_dict': {'Normal': 0,
                   'Tumour': 1},

    'label_column': 'Condition',

    'eval_columns': ['Condition', 'Cell Type'],

    # These are tunable hyperparameters ##########################################################
    'col_of_interest': ['anker_value', 'radius_distance', 'fussy_limit',
                        'droupout_rate', 'comment', 'comment_norm', 'model_no', 'split_number'],
    'col_of_variables': ['fussy_limit', 'anker_value', 'radius_distance'],

    'minimum_number_cells': 25,
    'radius_distance_all': [9, 27],
    'fussy_limit_all': [0.4, 0.8],
    'anker_value_all': [0.01, 0.2],

    'filter_column': None,
    'filter_value': None,
    'filter_cells': False,

    'anker_cell_selction_type': '%',  # either % or absolut
    'multiple_labels_per_subSample': False,

    'droupout_rate': [0.2],
    'input_layer': 100,
    'batch_size': 32,
    'learning_rate': 1e-2,

    'output_layer': 2,
    'layer_1': 38,
    'attr_bool': False,
    'comment_norm': 'no_norm',
    'databased_norm': None,
    #################################################################################################

    'augmentation_number': 5,
    'X_col_name': 'X',
    'Y_col_name': 'Y',
    'measument_sample_name': 'Patient',

    'validation_split_column': 'Fold',
    'number_validation_splits': [0, 1, 2, 3, 4],
    'test_set_fold_number': [5],
    'voro_neighbours': 50,
}

```


