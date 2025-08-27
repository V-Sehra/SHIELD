import argparse
import pickle
from utils import optuna_utils
from pathlib import Path

import torch
import optuna

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
parser = argparse.ArgumentParser()

parser.add_argument("-req_path", "--requirements_file_path", default=None)
parser.add_argument("-conf_model", "--config_file_path_model", default=None)
parser.add_argument("-b_path", "--best_conf_path", default=None)

parser.add_argument("-model_type", "--model_type", type=str, default='GAT',
                    choices=['GAT', 'GNN', 'MLP']
                    )
parser.add_argument("-val_split", "--validationSplit", default=1)
parser.add_argument("-fix_nodeSize", "--fix_nodeSize", default=132)
parser.add_argument("-scoring", "--scoring", type=str, default='f1_weighted',
                    choices=['f1_weighted', 'balanced_accuracy', 'accuracy'])

parser.add_argument("-reRun_bool", "--reRun_bool", type=bool, default=False,
                    choices=[False, True],
                    help="if the hyperparameter search should be rerun or not. If False, and the results .csv already exists the configuration will be skipped.")


def main():
    args, unknown = parser.parse_known_args()
    print(args)

    val_split = int(args.validationSplit)

    requirements = pickle.load(open(args.requirements_file_path, 'rb'))
    requirements['path_training_results'] = Path(str(requirements['path_training_results']) + f'_{args.model_type}')
    requirements['path_to_model'] = Path(str(requirements['path_to_model']) + f'_{args.model_type}')

    model_config = pickle.load(open(args.config_file_path_model, 'rb'))
    best_config_dict = pickle.load(open(args.best_conf_path, 'rb'))

    path_to_graphs = Path(requirements['path_to_data_set'] /
                          f'anker_value_{best_config_dict["anker_value"]}'.replace('.', '_') /
                          f"min_cells_{requirements['minimum_number_cells']}" /
                          f"{best_config_dict['fussy_limit']}" /
                          f"radius_{best_config_dict['radius_distance']}")
    data_loader_train, data_loader_validation = optuna_utils.load_data_loader(model_type=args.model_type,
                                                                              path_to_graphs=path_to_graphs,
                                                                              split_number=val_split,
                                                                              requirements=requirements,
                                                                              fix_nodeSize=args.fix_nodeSize)

    requirements['path_training_results'].mkdir(parents=True, exist_ok=True)
    db_path = Path(
        f"{requirements['path_training_results']}/tuning_v{args.validationSplit}.db").resolve()
    storage_uri = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=f"{args.model_type}_tuning_v{args.validationSplit}_Scoring{args.scoring}",
        storage=storage_uri,
        direction="maximize",
        load_if_exists=True
    )
    
    input_dim = requirements['input_layer']
    if args.model_type == 'MLP':
        input_dim = input_dim * 132

        
    output_dim = requirements['output_layer']

    # hypersearch:
    study.optimize(lambda trial: optuna_utils.objective(trial, data_loader_train=data_loader_train,
                                                        data_loader_test=data_loader_validation,
                                                        model_type=args.model_type, input_dim=input_dim,
                                                        output_dim=output_dim,
                                                        config_file_train=model_config[args.model_type],
                                                        args=args, device=device), n_trials=50)


if __name__ == "__main__":
    main()
