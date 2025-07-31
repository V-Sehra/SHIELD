import pickle
from pathlib import Path
import argparse

from torch_geometric.loader import DataListLoader

from utils import train_utils, model_utils
from utils.data_class import graph_dataset
from model import ShIeLD

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("-req_path", "--requirements_file_path",
                    default=None)

parser.add_argument("-aug", "--augmentation",
                    default='noisyLabel_prob')
parser.add_argument("-noisy_edge", "--noisy_edge",
                    default=False)
parser.add_argument("-noise_yLabel", "--noise_yLabel",
                    default=False)
parser.add_argument("-c", "--comment",
                    default=False)

args = parser.parse_args()

if args.augmentation == 'noisyLabel_prob':
    args.noise_yLabel = 'prob'

elif args.augmentation == 'noisyLabel_even':
    args.noise_yLabel = 'even'

if args.augmentation == 'edge_augmentation_percent':
    args.noisy_edge = 'percent'
elif args.augmentation == 'edge_augmentation_sameCon':
    args.noisy_edge = 'sameCon'

if args.requirements_file_path is None:
    args.requirements_file_path = Path.cwd() / 'rebuttle' / f'req_HCC_{args.augmentation}.pt'
requirements = pickle.load(open(args.requirements_file_path, 'rb'))

# need to overwrite the batch size:
requirements['batch_size'] = 32

if args.comment is not False:
    requirements['path_training_results'] = Path(str(requirements['path_training_results']) + f'{args.comment}')
    requirements['path_to_model'] = Path(str(requirements['path_to_model']) + f'{args.comment}')

print(args)

if 'sampleing' in requirements.keys():
    if requirements['sampleing'] == 'random':
        fussy_vector = ['randomSampling']
        fussy_folder_name = 'random_sampling'
    elif requirements['sampleing'] == 'voronoi':
        fussy_folder_name = True
        fussy_vector = ['fussy_limit_all']
else:
    fussy_vector = ['fussy_limit_all']
    fussy_folder_name = True

best_model_specs = {'number_of_anker_cells': 500,
                    'radius_neibourhood': 530,
                    'fussy_limit': 0.8,
                    'fussy_reference_beak': 'True'}

model_specs = {'layer_1': 23,
               'input_layer': 23,
               'droupout_rate': 0.2,
               'final_layer': 3,
               'attr_bool': False,
               'output_layer': 3,
               'comment_norm': 'No_norm'
               }
anker_number = best_model_specs['number_of_anker_cells']
fussy_reference_beak = best_model_specs['fussy_reference_beak']
radius_distance = best_model_specs['radius_neibourhood']
fussy_limit = best_model_specs['fussy_limit']

minimum_number_cells = 25

if 'sampleing' in requirements.keys():
    if requirements['sampleing'] == 'random':
        path_to_graphs = Path(requirements['path_to_data_set'] /
                              f'anker_value_{anker_number}'.replace('.', '_') /
                              f"min_cells_{requirements['minimum_number_cells']}" /
                              'random_sampling' /
                              f'radius_{radius_distance}')
    elif requirements['sampleing'] == 'voronoi':
        path_to_graphs = Path(requirements['path_to_data_set'] /
                              f'anker_value_{anker_number}'.replace('.', '_') /
                              f"min_cells_{requirements['minimum_number_cells']}" /
                              f"fussy_limit_{fussy_limit}".replace('.', '_') /
                              f'radius_{radius_distance}')
else:
    path_to_graphs = Path(requirements['path_to_data_set'] /
                          f'anker_value_{anker_number}'.replace('.', '_') /
                          f"min_cells_{requirements['minimum_number_cells']}" /
                          f"fussy_limit_{fussy_limit}".replace('.', '_') /
                          f'radius_{radius_distance}')

data_loader_train = DataListLoader(
    graph_dataset(
        root=str(path_to_graphs / 'train' / 'graphs'),
        path_to_graphs=path_to_graphs,
        fold_ids=requirements['number_validation_splits'],
        requirements_dict=requirements,
        graph_file_names=f'train_set_file_names.pkl'
    ),
    batch_size=requirements['batch_size'], shuffle=True, num_workers=8, prefetch_factor=50
)

data_loader_test = DataListLoader(
    graph_dataset(
        root=str(path_to_graphs / 'test' / 'graphs'),
        path_to_graphs=path_to_graphs,
        fold_ids=requirements['test_set_fold_number'],
        requirements_dict=requirements,
        graph_file_names=f'test_set_file_names.pkl'
    ),
    batch_size=requirements['batch_size'], shuffle=True, num_workers=8, prefetch_factor=50
)

loss_init_path = path_to_graphs / 'train' / 'graphs' if args.noise_yLabel is not False else path_to_graphs / 'train_set_file_names.pkl'
loss_fkt = train_utils.initiaize_loss(
    path=Path(loss_init_path),
    tissue_dict=requirements['label_dict'],
    device=device,
    noise_yLabel=args.noise_yLabel
)
requirements['path_to_model'].mkdir(parents=True, exist_ok=True)
requirements['path_training_results'].mkdir(parents=True, exist_ok=True)

for repeat in range(requirements['augmentation_number']):
    print(f"Repeat number: {repeat}")
    model = ShIeLD(
        num_of_feat=int(model_specs['input_layer']),
        layer_1=model_specs['layer_1'],
        layer_final=model_specs['output_layer'],
        dp=model_specs['droupout_rate'],
        self_att=False, attr_bool=model_specs['attr_bool'],
        norm_type=model_specs['comment_norm'],
        noisy_edge=args.noisy_edge
    ).to(device)

    model.train()

    model, train_loss = train_utils.train_loop_shield(
        optimizer=torch.optim.Adam(model.parameters(), lr=requirements['learning_rate']),
        model=model,
        data_loader=data_loader_train,
        loss_fkt=loss_fkt,
        attr_bool=model_specs['attr_bool'],
        device=device,
        patience=model_specs['patience'] if 'patience' in model_specs.keys() else 5,
        noise_yLabel=args.noise_yLabel
    )

    model.eval()
    print('start validation')
    train_bal_acc, train_f1_score, train_cm = model_utils.get_acc_metrics(
        model=model, data_loader=data_loader_train, device=device
    )

    val_bal_acc, val_f1_score, test_cm = model_utils.get_acc_metrics(
        model=model, data_loader=data_loader_test, device=device
    )

    print(f"Train Balanced Accuracy: {train_bal_acc}, Train F1 Score: {train_f1_score}")
    print(f"Val Balanced Accuracy: {val_bal_acc}, Val F1 Score: {val_f1_score}")
    pickle.dump({'train_bal_acc': train_bal_acc,
                 'train_f1_score': train_f1_score,
                 'train_cm': train_cm,
                 'val_bal_acc': val_bal_acc,
                 'val_f1_score': val_f1_score,
                 'val_cm': test_cm, },
                open(requirements['path_training_results'] / f"repeat_{repeat}_results.pkl", 'wb'))
    torch.save(model.state_dict(), requirements['path_to_model'] / f"model_repeat_{repeat}.pt")
