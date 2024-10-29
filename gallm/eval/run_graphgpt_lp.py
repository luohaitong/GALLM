import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from gallm.conversation import conv_templates, SeparatorStyle
from gallm.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from gallm.model import *
from gallm.model.utils import KeywordsStoppingCriteria
from gallm.model.GraphLlama import GraphLlamaForCausalLM_iaware
from torch_geometric.data import Data
import json
import copy

import os
import requests
from PIL import Image
from io import BytesIO

from tqdm import tqdm
import json
import os.path as osp

import ray

# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"
DEFAULT_LABEL_TOKEN = "<g_label>"


def load_graph_LP(node_idx_group, path_prefix, dataset ,template): 

    emb_path = f"{path_prefix}/{dataset}/sbert_x.pt"
    pretrained_embs = torch.tensor(torch.load(emb_path))
    structure_emb = torch.load(f'{path_prefix}/laplacian_2_10.pt')


    node_idx_1 = int(node_idx_group[0])
    node_idx_2 = int(node_idx_group[1])
    label_idx = int(node_idx_group[2])
    sample_neighbors = json.load(open(f"{path_prefix}/{dataset}/sample_neighbors.json", "r"))

    if template == 'nd':
        graph_dict = {'node_idx_1':node_idx_1, 'edge_index_1':sample_neighbors[str(node_idx_1)]['edge_index'], 'node_list_1': sample_neighbors[str(node_idx_1)]['pad_node_list'],
                        'node_idx_2':node_idx_2, 'edge_index_2':sample_neighbors[str(node_idx_2)]['edge_index'], 'node_list_2': sample_neighbors[str(node_idx_2)]['pad_node_list']
                    }
    else:
        graph_dict = {'node_idx_1':node_idx_1, 'edge_index_1':sample_neighbors[str(node_idx_1)]['edge_index'], 'node_list_1': sample_neighbors[str(node_idx_1)]['node_list'],
                    'node_idx_2':node_idx_2, 'edge_index_2':sample_neighbors[str(node_idx_2)]['edge_index'], 'node_list_2': sample_neighbors[str(node_idx_2)]['node_list']
                    }
                        
    graph_edge_index_1 = torch.Tensor(copy.deepcopy(graph_dict['edge_index_1'])).long()
    target_node_1 = copy.deepcopy(graph_dict['node_idx_1'])
    
    if template == 'nd':
        graph_node_list = copy.deepcopy(sample_neighbors[str(target_node_1)]['pad_node_list'])
        graph = torch.LongTensor(graph_node_list)
        mask = (graph != -500)
        masked_graph_emb = pretrained_embs[graph[mask]]
        n, d = graph.shape[0], masked_graph_emb.shape[1]
        graph_emb = torch.zeros((n, d))
        graph_emb[mask] = masked_graph_emb
        if structure_emb is not None:
            graph_emb = torch.cat([graph_emb, structure_emb], dim=-1)
        graph_node_rep_1 = graph_emb
    else:
        graph_node_list = copy.deepcopy(sample_neighbors[str(target_node_1)]['node_list'])
        graph_node_rep_1 = pretrained_embs[graph_node_list] ## 

    cur_token_len_1 = len(graph_node_rep_1)   # FIXME: 14 is hardcoded patch size

    graph_edge_index_2 = torch.Tensor(copy.deepcopy(graph_dict['edge_index_2'])).long()
    target_node_2 = copy.deepcopy(graph_dict['node_idx_2'])
    
    if template == 'nd':
        graph_node_list = copy.deepcopy(sample_neighbors[str(target_node_2)]['pad_node_list'])
        graph = torch.LongTensor(graph_node_list)
        mask = (graph != -500)
        masked_graph_emb = pretrained_embs[graph[mask]]
        n, d = graph.shape[0], masked_graph_emb.shape[1]
        graph_emb = torch.zeros((n, d))
        graph_emb[mask] = masked_graph_emb
        if structure_emb is not None:
            graph_emb = torch.cat([graph_emb, structure_emb], dim=-1)
        graph_node_rep_2 = graph_emb
    else:
        graph_node_list = copy.deepcopy(sample_neighbors[str(target_node_2)]['node_list'])
        graph_node_rep_2 = pretrained_embs[graph_node_list] ## 

    cur_token_len_2 = len(graph_node_rep_2)   # FIXME: 14 is hardcoded patch 

    graph_ret = {
        'graph_1': Data(graph_node = graph_node_rep_1, edge_index=graph_edge_index_1, target_node = torch.tensor([target_node_1])), 
        'graph_2': Data(graph_node = graph_node_rep_2, edge_index=graph_edge_index_2, target_node = torch.tensor([target_node_2]))
        }

    return {
        'graph_data': graph_ret, 
        'graph_token_len_1': cur_token_len_1, 
        'graph_token_len_2': cur_token_len_2 
    }

def construct_conversation(task, num_labels, node_idx, y, label_texts, title, abs, maunal_prompt, dataset):
    if task == "nc":
        label_text_tem = []
        for i in range(num_labels):
            label_text_tem.append(f"category: {label_texts[i]}")
        label_text_str_tem = '; '.join(label_text_tem).lower()
        user_prompt = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, each node represents a paper. The first token represents the central node of the subgraph. The remaining represent the neighbors. we need to classify the center node into {num_labels} classes: {label_text_str_tem}. Please tell me which class the center node belongs?"
        assistant_prompt = label_texts[y[node_idx]].lower()
    elif task == "nd":
        user_prompt = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, each node represents a paper. The first graph token represents the central node of the subgraph and the remaining represent the neighbor nodes. Please briefly describe the center node."
        assistant_prompt = process_title(title[node_idx])
    elif task == 'tm':
        candidate_file = title_match_file[dataset][str(node_idx)]
        candidate_list = candidate_file['candidate_list']
        node_posit_index = candidate_file['node_posit_index']
        candiate_titles = [process_title(title[i]) for i in candidate_list]
        num_labels = len(candidate_list)
        label_text_tem = []
        for i in range(num_labels):
            label_text_tem.append(f"title: {candiate_titles[i]}")
        label_text_str_tem = '; '.join(label_text_tem).lower()

        user_prompt = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, each node represents a paper. The first token represents the central node of the subgraph. The remaining represent the neighbors. we need to classify the center node into {num_labels} titles: {label_text_str_tem}. please tell me which title the center node belongs to?"
        assistant_prompt = process_title(title[node_idx])
    elif task == 'nc_sp':
        label_text_tem = []
        for i in range(num_labels):
            label_text_tem.append(f"category: {label_texts[i]}, <g_label>")
        label_text_str_tem = '; '.join(label_text_tem).lower()
        user_prompt = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, each node represents a paper. The first token represents the central node of the subgraph. The remaining represent the neighbors. We need to classify the center node into {num_labels} classes: {label_text_str_tem}. Please tell me which class the center node belongs to?"
        assistant_prompt = label_texts[y[node_idx]].lower()
    elif task == 'nc_mp':
        label_text_tem = []
        for i in range(num_labels):
            label_text_tem.append(f"category: {label_texts[i]}, {maunal_prompt[dataset][label_texts[i].lower()]}")
        label_text_str_tem = '; '.join(label_text_tem).lower()
        user_prompt = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, each node represents a paper. The first token represents the central node of the subgraph. The remaining represent the neighbors. We need to classify the center node into {num_labels} classes: {label_text_str_tem}. Please tell me which class the center node belongs to?"
        assistant_prompt = label_texts[y[node_idx]].lower()
    elif task == "nc_graphgpt":
        label_text_tem = []
        for i in range(num_labels):
            label_text_tem.append(f"category: {label_texts[i]}")
        label_text_str_tem = '; '.join(label_text_tem).lower()
        user_prompt = f"Given a citation graph: {DEFAULT_GRAPH_TOKEN} where the 0th node is the target paper, and other nodes are its one-hop or multi-hop neighbors. Question: Which arXiv CS sub-category does this paper belong to? Give the most likely arXiv CS sub-categories of this paper directly, in the form \"cs.XX\" with full name of the category."
        assistant_prompt = label_texts[y[node_idx]].lower()
    elif task == 'lp':
        label_text_tem = []
        for i in range(num_labels):
            label_text_tem.append(f"category: {label_texts[i]}")
        label_text_str_tem = '; '.join(label_text_tem).lower()
        user_prompt = f"Given two node-centered graphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, each node represents a paper. Please tell me whether two center nodes in the subgraphs should connect to each other. Answer yes or no."
        if node_idx[2] == 0:
            assistant_prompt = "no"
        elif node_idx[2] == 1:
            assistant_prompt = "yes"
        else:
            print("label error")
            raise ValueError
    elif task == 'lp_sp':
        label_text_tem = []
        for i in range(num_labels):
            label_text_tem.append(f"category: {label_texts[i]}, <g_label>")
        label_text_str_tem = '; '.join(label_text_tem).lower()
        user_prompt = f"Given two node-centered graphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, each node represents a paper. The first graph token represents the central node of the subgraph and the remaining represent the neighbor nodes. We need to classify the connectivity of these two center nodes into 2 classes: connected, <g_label>; category: unconnected, <g_label>. Please tell me which class the connectivity of these two center nodes belongs?"
        if node_idx[2] == 0:
            assistant_prompt = "unconnected"
        elif node_idx[2] == 1:
            assistant_prompt = "connected"
        else:
            print("label error")
            raise ValueError
    elif task == 'lp_mp':
        label_text_tem = []
        for i in range(num_labels):
            label_text_tem.append(f"category: {label_texts[i]}")
        label_text_str_tem = '; '.join(label_text_tem).lower()
        user_prompt = f"Given two node-centered graphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, each node represents a paper. The first graph token represents the central node of the subgraph and the remaining represent the neighbor nodes. We need to classify the connectivity of these two center nodes into 2 classes: connected, which means a paper citation relationship exists; category: unconnected, which means they not have a paper citation relationship. Please tell me which class the connectivity of these two center nodes belongs?"
        if node_idx[2] == 0:
            assistant_prompt = "unconnected"
        elif node_idx[2] == 1:
            assistant_prompt = "connected"
        else:
            print("label error")
            raise ValueError
    else:
        print(f"{task} not exist!!!")
        raise ValueError
    return user_prompt, assistant_prompt

def run_eval(args, num_gpus):

    data_path = f"{args.path_prefix}/{args.dataset}/processed_data.pt"
    data = torch.load(data_path)
    label_texts = data.label_texts
    if 'title' in data:
        title = data.title
    else:
        title = data.raw_texts
    #abs = data.abs
    y = data.y
    num_labels = len(label_texts)
    label_texts_str = ','.join(label_texts).lower()
    task = args.task

    prompt_file = torch.load(f'{args.path_prefix}/{args.dataset}/lp_test.pt')
    if args.end_id == -1:
        args.end_id = len(prompt_file)-1
    prompt_file = prompt_file[args.start_id:args.end_id]
    chunk_size = len(prompt_file) // num_gpus
    ans_handles = []
    split_list = list(range(args.start_id, args.end_id, chunk_size))
    idx_list = list(range(0, len(prompt_file), chunk_size))
    if len(split_list) == num_gpus: 
        split_list.append(args.end_id)
        idx_list.append(len(prompt_file))
    elif len(split_list) == num_gpus + 1: 
        split_list[-1] = args.end_id
        idx_list[-1] = len(prompt_file)
    else: 
        raise ValueError('error in the number of list')

    if osp.exists(args.output_res_path) is False: 
        os.mkdir(args.output_res_path)
    
    maunal_prompt = json.load(open(f'{args.path_prefix}/label_manual_prompt.json', "r"))

    for idx in range(len(idx_list) - 1):
        start_idx = idx_list[idx]
        end_idx = idx_list[idx + 1]
        
        start_split = split_list[idx]
        end_split = split_list[idx + 1]
        ans_handles.append(
            eval_model.remote(
                args, prompt_file[start_idx:end_idx], start_split, end_split, label_texts, title, abs, y, num_labels, label_texts_str, task, args.dataset, maunal_prompt
            )
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))

    # with open(args.output_res_path, "w") as ans_file:
    #     for line in ans_jsons:
    #         ans_file.write(json.dumps(line) + "\n")


@ray.remote(num_gpus=1)
@torch.inference_mode()
def eval_model(args, prompt_file, start_idx, end_idx, label_texts, title, abs, y, num_labels, label_texts_str, task, dataset, maunal_prompt):

    # Model
    disable_torch_init()
    print('start loading')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print('finish loading')

    print('start loading')
    if task == 'nc_sp' or task == 'lp_sp':
        model = GraphLlamaForCausalLM_iaware.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, use_cache=True, low_cpu_mem_usage=True).cuda()
    else:
        model = GraphLlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, use_cache=True, low_cpu_mem_usage=True).cuda()
    print('finish loading')
    use_graph_start_end = getattr(model.config, "use_graph_start_end", False)
    tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
    tokenizer.add_tokens([DEFAULT_LABEL_TOKEN], special_tokens=True)
    if use_graph_start_end:
        tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)

    graph_tower = model.get_model().graph_tower
    
    clip_graph, args_graph= load_model_pretrained(CLIP, args.pretra_gnn)
    graph_tower = graph_transformer(args_graph)
    graph_tower = transfer_param_tograph(clip_graph, graph_tower)
    
    model.get_model().graph_tower = graph_tower.cuda()

    graph_tower.to(device='cuda', dtype=torch.bfloat16)
    graph_config = graph_tower.config
    graph_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]
    graph_config.graph_label_token = tokenizer.convert_tokens_to_ids([DEFAULT_LABEL_TOKEN])[0]
    graph_config.use_graph_start_end = use_graph_start_end
    if use_graph_start_end:
        graph_config.graph_start_token, graph_config.graph_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])

    res_data = []
    print(f'total: {len(prompt_file)}')
    predict_list = []

    for idx, instruct_item in tqdm(enumerate(prompt_file)):

        graph_dict = load_graph_LP(instruct_item, args.path_prefix, dataset, args.template)
        node_idx = instruct_item
        graph_token_len_1 = graph_dict['graph_token_len_1']
        graph_token_len_2 = graph_dict['graph_token_len_2']
        graph_data = graph_dict['graph_data']

        qs, gt = construct_conversation(task, num_labels, node_idx, y, label_texts, title, abs, maunal_prompt, dataset)

        replace_token_1 = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len_1
        replace_token_2 = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len_2

        if use_graph_start_end:
            replace_token_1 = DEFAULT_G_START_TOKEN + replace_token_1 + DEFAULT_G_END_TOKEN
            replace_token_2 = DEFAULT_G_START_TOKEN + replace_token_2 + DEFAULT_G_END_TOKEN

        if DEFAULT_GRAPH_TOKEN in qs:
            first_index = qs.find(DEFAULT_GRAPH_TOKEN)
            qs = qs[:first_index] + replace_token_1 + qs[first_index+len(DEFAULT_GRAPH_TOKEN):]

            second_index = qs.find(DEFAULT_GRAPH_TOKEN)
            qs = qs[:second_index] + replace_token_2 + qs[second_index+len(DEFAULT_GRAPH_TOKEN):]

        if  task == 'lp_sp':
            label_token_len = 3
            replace_token = DEFAULT_LABEL_TOKEN * label_token_len
            qs = qs.replace(DEFAULT_LABEL_TOKEN, replace_token)
        conv_mode = "graphchat_v1"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])

        
        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        graph_data['graph_1'].graph_node = graph_data['graph_1'].graph_node.to(torch.bfloat16)
        graph_data['graph_2'].graph_node = graph_data['graph_2'].graph_node.to(torch.bfloat16)

        graph_data['graph_1'] = graph_data['graph_1'].cuda()
        graph_data['graph_2'] = graph_data['graph_2'].cuda()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                graph_data=graph_data,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()


        res_data.append({"res": outputs, "gt":gt}.copy())
        with open(osp.join(args.output_res_path, '{}_test_res_{}_{}.json'.format(args.dataset, start_idx, end_idx)), "w") as fout:
            json.dump(res_data, fout, indent=4)
    return res_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--pretra_gnn", type=str, default=None)

    parser.add_argument("--output_res_path", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=4)

    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=20567)
    parser.add_argument("--dataset", type=str, default='cora')
    parser.add_argument("--task", type=str, default='nc')
    parser.add_argument("--template", type=str, default='nd')
    parser.add_argument("--path_prefix", type=str, default="/path/to/GALLM/dataset")

    args = parser.parse_args()

    # eval_model(args)

    ray.init()
    run_eval(args, args.num_gpus)
