#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Nov 2024

@author: Vivek
"""
import argparse
import pickle
from pathlib import Path
import torch

import utils.evaluation_utils as evaluation_utils


torch.multiprocessing.set_start_method('fork', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-req_path", "--requirements_file_path",
                        default=Path.cwd() / 'examples' / 'diabetes' / 'requirements.pt')

    args = parser.parse_args()
    print(args)

    with open(args.requirements_file_path, 'rb') as file:
        requirements = pickle.load(file)

    col_of_interest = ['anker_value', 'radius_neibourhood', 'fussy_limit',
                        'dp', 'comment', 'comment_norm', 'model_no','split_number']
    col_of_variables = ['dropout', 'fussy_limit','anker_value','radius_distance']

    hyper_search_results = evaluation_utils.get_hypersear_results(requirements_dict = requirements,
                                                                  col_of_interest = col_of_interest,
                                                                  col_of_variables = col_of_variables)
    hyper_search_results.to_csv(Path(requirements['path_training_results'] / 'hyper_search_results.csv'), index=False)

    melted_results = hyper_search_results.melt(id_vars=['total_acc_balanced_mean'], var_name='hyperparameter')


    save_path_folder = Path(requirements['path_training_results'] / 'hyper_search_plots')
    save_path_folder.mkdir(parents=True, exist_ok=True)

    for observable_of_interest in col_of_variables:

        evaluation_utils.create_parameter_influence_plots(df = melted_results,
                                                          observed_variable = observable_of_interest,
                                                          save_path=save_path_folder / f'{observable_of_interest}.png')


if __name__ == "__main__":
    main()
