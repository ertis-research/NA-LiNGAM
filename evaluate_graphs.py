import cdt

from cdt.metrics import precision_recall, SID, SHD

from utils.graph_functions import load_graph_from_json, is_path, check_graph, evaluate_graphs

import json
import os

#########################################################
#########################################################

test_name = 'results_all_20_noise_20_iter'
is_real_data = False
max_noise = 4

#########################################################
#########################################################

if is_real_data:
    graphs_path = 'results_real/' + test_name + '/'
else:
    graphs_path = 'results_synthetic/' + test_name + '/'

folder_list = os.listdir(graphs_path)
graphs_models_dict = {}

true_graphs = {}

for folder in folder_list:

    if folder == 'real_graph':
        if is_real_data:
            true_graphs['real'] = load_graph_from_json(graphs_path + folder + '/' + os.listdir(graphs_path + folder)[0])

    elif folder != 'GrangerAlgorithm':  # Exclude GrangerAlgorithm results
        graphs_dict = {}

        for noise_level in range(max_noise + 1):
            graphs_dict[str(noise_level)] = None
            if not is_real_data:
                true_graphs[str(noise_level)] = None


        file_list = os.listdir(graphs_path + folder)
        for file in file_list:
            try:
                name_dir = graphs_path + folder + '/' + file

                graph = load_graph_from_json(name_dir)

                noise_level = file.split('_')[2]
                if graphs_dict[noise_level] is None:
                    graphs_dict[noise_level] = []
                graphs_dict[noise_level].append(graph)

                if not is_real_data:
                    true_graph = load_graph_from_json(graphs_path + 'real_graph/' + file)
                    if true_graphs[noise_level] is None:
                        true_graphs[noise_level] = []
                    true_graphs[noise_level].append(true_graph)

            except:
                print(f'Error loading {file}')
                continue

        for noise_level in graphs_dict:
            if graphs_dict[noise_level] is None:
                graphs_dict[noise_level] = []

        graphs_models_dict[folder] = graphs_dict
        

results = evaluate_graphs(graphs_models_dict, true_graphs, is_real_data)

# Save results to a JSON file
if is_real_data:
    save_path = f'graph_metrics/real_{test_name}.json'
else:
    save_path = f'graph_metrics/synthetic_{test_name}.json'

with open(save_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f'Results saved to {save_path}')