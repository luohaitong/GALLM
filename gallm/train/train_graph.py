# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import random
import torch
import pickle as pk
import numpy as np
import transformers
from torch.utils.data import Dataset
from graphchat_trainer import GraphChatTrainer
import gallm.conversation as conversation_lib
from gallm.model.GraphLlama import GraphLlamaForCausalLM, GraphLlamaForCausalLM_iaware

from PIL import Image
import torch.nn as nn
from torch_geometric.data import Data

# TODO: import and use code from ../data/dataset.py

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"
DEFAULT_LABEL_TOKEN = "<g_label>"

def process_title(title):
    title = title.lower()
    if title.startswith('title: '):
        # Remove the begining Title: '
        new_string = title[len('title: '):]
    else:
        # Keep unchanged
        new_string = title
    if new_string.endswith(' '):
        new_string = new_string[:-2]
    if new_string.endswith('.'):
        new_string = new_string[:-1]
    return new_string.lower()

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_graph_mlp_adapter: bool = field(default=False)
    graph_tower: Optional[str] = field(default=None)
    graph_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_graph_mlp_adapter: Optional[str] = field(default=None)
    use_graph_start_end: bool = field(default=False)
    label_iaware: bool = field(default=False)


@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_graph: bool = False
    sep_graph_conv_front: bool = False
    graph_token_len: int = 0
    data_path_prefix: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    use_task: Optional[str] = field(default="nc")
    use_dataset: Optional[str] = field(default="cora")
    template: Optional[str] = field(default="nd")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_graph_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    disable_tqdm: bool =False


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_graph(
    sources: Sequence[str],
    graph_cfg: dict,
    cur_token_len: int,
) -> Dict:
    is_graph = graph_cfg['is_graph']
    # image_token_len = multimodal_cfg['image_token_len']
    graph_token_len = cur_token_len
    if not is_graph:
        return sources

    for source in sources:
        if graph_cfg['sep_graph_conv_front']:
            assert DEFAULT_GRAPH_TOKEN in source[0]['value']
            source[0]['value'] = source[0]['value'].replace(DEFAULT_GRAPH_TOKEN, '').strip()
            source[0]['value'] = DEFAULT_GRAPH_TOKEN + conversation_lib.default_conversation.sep + conversation_lib.default_conversation.roles[0] + ": " + source[0]['value']
        for sentence in source:
            replace_token = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len
            if graph_cfg['use_graph_start_end']:
                replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_GRAPH_TOKEN, replace_token)

    return sources

def preprocess_graph_LP(
    sources: Sequence[str],
    graph_cfg: dict,
    cur_token_len_1: int,
    cur_token_len_2: int,
) -> Dict:
    is_graph = graph_cfg['is_graph']
    # image_token_len = multimodal_cfg['image_token_len']
    graph_token_len_1 = cur_token_len_1
    graph_token_len_2 = cur_token_len_2

    if not is_graph:
        return sources

    for source in sources:
        if graph_cfg['sep_graph_conv_front']:
            assert DEFAULT_GRAPH_TOKEN in source[0]['value']
            source[0]['value'] = source[0]['value'].replace(DEFAULT_GRAPH_TOKEN, '').strip()
            source[0]['value'] = DEFAULT_GRAPH_TOKEN + conversation_lib.default_conversation.sep + conversation_lib.default_conversation.roles[0] + ": " + source[0]['value']
        for sentence in source:
            replace_token_1 = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len_1
            replace_token_2 = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len_2
            if graph_cfg['use_graph_start_end']:
                replace_token_1 = DEFAULT_G_START_TOKEN + replace_token_1 + DEFAULT_G_END_TOKEN
                replace_token_2 = DEFAULT_G_START_TOKEN + replace_token_2 + DEFAULT_G_END_TOKEN

            if DEFAULT_GRAPH_TOKEN in sentence["value"]:
                first_index = sentence["value"].find(DEFAULT_GRAPH_TOKEN)
                sentence["value"] = sentence["value"][:first_index] + replace_token_1 + sentence["value"][first_index+len(DEFAULT_GRAPH_TOKEN):]

                # Replace the second <graph> with B
                second_index = sentence["value"].find(DEFAULT_GRAPH_TOKEN)
                sentence["value"] = sentence["value"][:second_index] + replace_token_2 + sentence["value"][second_index+len(DEFAULT_GRAPH_TOKEN):]


            # sentence["value"] = sentence["value"].replace(DEFAULT_GRAPH_TOKEN, replace_token)

    # print(sources)

    return sources

def preprocess_label(
    sources: Sequence[str],
    cur_token_len: int,
) -> Dict:

    for source in sources:
        for sentence in source:
            replace_token = DEFAULT_LABEL_TOKEN * cur_token_len
            sentence["value"] = sentence["value"].replace(DEFAULT_LABEL_TOKEN, replace_token)

    return sources

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids) + len(tokenizer(conv.sep).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids)
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "v1":
        return preprocess_v1(sources, tokenizer)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...")
        sources = [example["conversations"] for example in list_data_dict]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer,
                 graph_cfg: dict, 
                 **kwargs,):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.graph_cfg = graph_cfg
        self.graph_data_all = {}
        self.graph_emb_all = {}
        self.template = kwargs.get('template')
        self.index={}
        list_data_dict = []
        self.use_dataset = kwargs.get('use_dataset').split('-')
        self.path_prefix = kwargs.get('data_path_prefix')
        print("use dataset:", self.use_dataset)
        for d, dataset in enumerate(self.use_dataset):
            
            # load original data
            data_path = f"{self.path_prefix}/{dataset}/processed_data.pt"
            data = torch.load(data_path)
            self.graph_data_all[dataset]=data

            # load necessary data for construct conversations
            label_texts = data.label_texts
            raw_texts = data.raw_texts
            if 'title' in data:
                title = data.title
            else:
                title = data.raw_texts
            #abs = data.abs
            y = data.y
            num_labels = len(label_texts)
            title_match_file = json.load(open(f'{self.path_prefix}/{dataset}/text_match_candidate.json', "r"))
            maunal_prompt = json.load(open(f'{self.path_prefix}/label_manual_prompt.json', "r"))
            train_mask = data.train_mask
            
            # load pretrained embedding
            emb_path = f"{self.path_prefix}/{dataset}/sbert_x.pt"
            pretrained_embs = torch.load(emb_path)
            if self.template == 'nd':
                self.structure_emb = torch.load(f'{self.path_prefix}/laplacian_2_10.pt')
            self.graph_emb_all[dataset] = torch.tensor(pretrained_embs)

            # load node index list and sampled neighbors
            sample_neighbors = json.load(open(f'{self.path_prefix}/{dataset}/sample_neighbors.json', "r"))

            self.use_task = kwargs.get('use_task').split('-')
            print("task:", self.use_task)
            node_idx_list = torch.arange(len(y))[data.train_mask]
            if 'nd' in self.use_task or 'tm' in self.use_task:
                node_idx_list = torch.load(f'{self.path_prefix}/{dataset}/stage1_id.pt')

            list_data_dict_tem = []
            # for every node, construct every task conversations
            for node_idx in node_idx_list:
                l = {}
                node_idx = int(node_idx)
                if self.template == 'nd':
                    l['graph'] = {'node_idx':node_idx, 'edge_index':sample_neighbors[str(node_idx)]['edge_index'], 'node_list': sample_neighbors[str(node_idx)]['pad_node_list']}
                else:
                    l['graph'] = {'node_idx':node_idx, 'edge_index':sample_neighbors[str(node_idx)]['edge_index'], 'node_list': sample_neighbors[str(node_idx)]['node_list']}
                for task in self.use_task:
                    if task == "nc":
                        label_text_tem = []
                        for i in range(num_labels):
                            label_text_tem.append(f"category: {label_texts[i]}")
                        label_text_str_tem = '; '.join(label_text_tem).lower()
                        if dataset == 'instagram':
                            user_prompt = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, each node represents a user. The first token represents the central node of the subgraph. The remaining represent the neighbors. we need to classify the center node into {num_labels} classes: {label_text_str_tem}. Please tell me which class the center node belongs?"
                        else:
                            user_prompt = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, each node represents a paper. The first token represents the central node of the subgraph. The remaining represent the neighbors. we need to classify the center node into {num_labels} classes: {label_text_str_tem}. Please tell me which class the center node belongs?"
                        assistant_prompt = label_texts[y[node_idx]].lower()
                    elif task == "nd":
                        user_prompt = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, each node represents a paper. The first graph token represents the central node of the subgraph and the remaining represent the neighbor nodes. Please briefly describe the center node."
                        assistant_prompt = process_title(title[node_idx])
                    elif task == 'tm':
                        candidate_file = title_match_file[str(node_idx)]
                        candidate_list = candidate_file['candidate_list']
                        node_posit_index = candidate_file['node_posit_index']
                        candiate_titles = [process_title(title[i]) for i in candidate_list]
                        num_labels = len(candidate_list)
                        label_text_tem = []
                        for i in range(num_labels):
                            label_text_tem.append(f"title: {candiate_titles[i]}")
                        label_text_str_tem = '; '.join(label_text_tem).lower()
                        if dataset == 'instagram':
                            user_prompt = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, each node represents a user. The first token represents the central node of the subgraph. The remaining represent the neighbors. we need to classify the center node into {num_labels} profiles: {label_text_str_tem}. please tell me which profile the center node belongs to?"
                        else:
                            user_prompt = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, each node represents a paper. The first token represents the central node of the subgraph. The remaining represent the neighbors. we need to classify the center node into {num_labels} titles: {label_text_str_tem}. please tell me which title the center node belongs to?"
                        assistant_prompt = process_title(title[node_idx])
                    elif task == 'nc_sp':
                        label_text_tem = []
                        for i in range(num_labels):
                            label_text_tem.append(f"category: {label_texts[i]}, <g_label>")
                        label_text_str_tem = '; '.join(label_text_tem).lower()
                        if dataset == 'instagram':
                            user_prompt = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, each node represents a user. The first token represents the central node of the subgraph. The remaining represent the neighbors. We need to classify the center node into {num_labels} classes: {label_text_str_tem}. Please tell me which class the center node belongs to?"
                        else:
                            user_prompt = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, each node represents a paper. The first token represents the central node of the subgraph. The remaining represent the neighbors. We need to classify the center node into {num_labels} classes: {label_text_str_tem}. Please tell me which class the center node belongs to?"
                        assistant_prompt = label_texts[y[node_idx]].lower()
                    elif task == 'nc_mp':
                        label_text_tem = []
                        for i in range(num_labels):
                            label_text_tem.append(f"category: {label_texts[i]}, {maunal_prompt[dataset][label_texts[i].lower()]}")
                        label_text_str_tem = '; '.join(label_text_tem).lower()
                        if dataset == 'instagram':
                            user_prompt = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, each node represents a user. The first token represents the central node of the subgraph. The remaining represent the neighbors. We need to classify the center node into {num_labels} classes: {label_text_str_tem}. Please tell me which class the center node belongs to?"
                        else:
                            user_prompt = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, each node represents a paper. The first token represents the central node of the subgraph. The remaining represent the neighbors. We need to classify the center node into {num_labels} classes: {label_text_str_tem}. Please tell me which class the center node belongs to?"
                        assistant_prompt = label_texts[y[node_idx]].lower()
                    elif task == 'gm':
                        candidate_list = sample_neighbors[str(node_idx)]['node_list']
                        candiate_titles = [process_title(title[i]) for i in candidate_list]
                        candiate_titles_shuffle = candiate_titles.copy()
                        random.shuffle(candiate_titles_shuffle)
                        num_labels = len(candidate_list)
                        label_text_tem = []
                        for i in range(num_labels):
                            label_text_tem.append(f"{i}. {candiate_titles_shuffle[i]}")
                        label_text_str_tem = '. '.join(label_text_tem).lower()

                        if dataset == 'instagram':
                            user_prompt = f"Given a sequence of graph tokens {DEFAULT_GRAPH_TOKEN} that constitute a subgraph of a social network graph, where the first token represents the central node of the subgraph, and the remaining nodes represent the first and second order neighbors of the central node. Each graph token contains the profile information of the user at this node. Here is a list of user profiles: {label_text_str_tem},  please reorder the list of users according to the order of graph tokens (i.e., complete the matching of graph tokens and users)."
                        else:
                            user_prompt = f"Given a sequence of graph tokens {DEFAULT_GRAPH_TOKEN} that constitute a subgraph of a citation graph, where the first token represents the central node of the subgraph, and the remaining nodes represent the first and second order neighbors of the central node. Each graph token contains the title and abstract information of the paper at this node. Here is a list of paper titles: {label_text_str_tem},  please reorder the list of papers according to the order of graph tokens (i.e., complete the matching of graph tokens and papers)."
                        gt_text_tem = []
                        for i in range(num_labels):
                            gt_text_tem.append(f"Graph token {i} corresponds to {process_title(candiate_titles[i])}")
                        gt_text_tem = '. '.join(gt_text_tem).lower()
                        assistant_prompt = f"Based on the given graph tokens and the list of paper titles, we obtain the matching of graph tokens and papers as follows: {gt_text_tem}."
                    elif task == "nc_graphgpt":
                        label_text_tem = []
                        for i in range(num_labels):
                            label_text_tem.append(f"category: {label_texts[i]}")
                        label_text_str_tem = '; '.join(label_text_tem).lower()
                        if dataset == 'cora':
                            user_prompt = f"Given a citation graph: {DEFAULT_GRAPH_TOKEN} where the 0th node is the target paper, and other nodes are its one-hop or multi-hop neighbors. Question: Which arXiv CS sub-category does this paper belong to? Give the most likely arXiv CS sub-categories of this paper directly, in the form \"cs.XX\" with full name of the category."
                        assistant_prompt = label_texts[y[node_idx]].lower()
                    else:
                        if 'lp' in task:
                            continue
                        else:
                            print(f"{task} not exist!!!")
                            raise ValueError
                    l['id'] = f'{dataset}-train-{task}-{node_idx}'
                    l["conversations"] = [{'from': 'human', 'value': user_prompt},
                                        {'from': 'gpt', 'value': assistant_prompt}]
                    list_data_dict_tem.append(l)
            
            if 'lp' in self.use_task or 'lp_mp' in self.use_task or 'lp_sp' in self.use_task:
                lp_node_idx_list = torch.load(f'{self.path_prefix}/{dataset}/lp_train.pt')

                for node_idx_group in lp_node_idx_list:
                    l = {}
                    node_idx_1 = int(node_idx_group[0])
                    node_idx_2 = int(node_idx_group[1])
                    label_idx = int(node_idx_group[2])
                    if self.template == 'nd':
                        l['graph'] = {'node_idx_1':node_idx_1, 'edge_index_1':sample_neighbors[str(node_idx_1)]['edge_index'], 'node_list_1': sample_neighbors[str(node_idx_1)]['pad_node_list'],
                                        'node_idx_2':node_idx_2, 'edge_index_2':sample_neighbors[str(node_idx_2)]['edge_index'], 'node_list_2': sample_neighbors[str(node_idx_2)]['pad_node_list']
                                    }
                    else:
                        l['graph'] = {'node_idx_1':node_idx_1, 'edge_index_1':sample_neighbors[str(node_idx_1)]['edge_index'], 'node_list_1': sample_neighbors[str(node_idx_1)]['node_list'],
                                    'node_idx_2':node_idx_2, 'edge_index_2':sample_neighbors[str(node_idx_2)]['edge_index'], 'node_list_2': sample_neighbors[str(node_idx_2)]['node_list']
                                    }
                    for task in self.use_task:
                        if task == 'lp_mp':
                            label_text_tem = []
                            for i in range(num_labels):
                                label_text_tem.append(f"category: {label_texts[i]}")
                            label_text_str_tem = '; '.join(label_text_tem).lower()
                            user_prompt = f"Given two node-centered graphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, each node represents a paper. The first graph token represents the central node of the subgraph and the remaining represent the neighbor nodes. We need to classify the connectivity of these two center nodes into 2 classes: connected, which means a paper citation relationship exists; category: unconnected, which means they don't have a paper citation relationship. Please tell me which class the connectivity of these two center nodes belongs?"
                            if label_idx == 0:
                                assistant_prompt = "unconnected"
                            elif label_idx == 1:
                                assistant_prompt = "connected"
                            else:
                                print("label error")
                                raise ValueError
                        elif task == 'lp_sp':
                            label_text_tem = []
                            for i in range(num_labels):
                                label_text_tem.append(f"category: {label_texts[i]}, <g_label>")
                            label_text_str_tem = '; '.join(label_text_tem).lower()
                            user_prompt = f"Given two node-centered graphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, each node represents a paper. The first graph token represents the central node of the subgraph and the remaining represent the neighbor nodes. We need to classify the connectivity of these two center nodes into 2 classes: connected, <g_label>; category: unconnected, <g_label>. Please tell me which class the connectivity of these two center nodes belongs?"
                            #assistant_prompt = label_texts[y[node_idx]].lower()
                            if label_idx == 0:
                                assistant_prompt = "unconnected"
                            elif label_idx == 1:
                                assistant_prompt = "connected"
                            else:
                                print("label error")
                        elif task == 'lp':
                            label_text_tem = []
                            for i in range(num_labels):
                                label_text_tem.append(f"category: {label_texts[i]}")
                            label_text_str_tem = '; '.join(label_text_tem).lower()
                            user_prompt = f"Given two node-centered graphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, each node represents a paper. Please tell me whether two center nodes in the subgraphs should connect to each other. Answer yes or no."
                            if label_idx == 0:
                                assistant_prompt = "no"
                            elif label_idx == 1:
                                assistant_prompt = "yes"
                            else:
                                print("label error")
                                raise ValueError
                        else:
                            continue
                        l['id'] = f'{dataset}-train-{task}-{node_idx_1}'
                        l["conversations"] = [{'from': 'human', 'value': user_prompt},
                                            {'from': 'gpt', 'value': assistant_prompt}]
                        list_data_dict_tem.append(l)
                     
            list_data_dict.extend(list_data_dict_tem)


        random.shuffle(list_data_dict)
        print(f"Formatting inputs...Skip in lazy mode, size {len(list_data_dict)}")
        self.list_data_dict = list_data_dict


    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        task_type = self.list_data_dict[i]['id'].split("-")[2]
        if 'lp' not in task_type: 
            if 'graph' in sources[0]:
                graph_dict = self.list_data_dict[i]['graph']
                graph_edge_index = torch.Tensor(copy.deepcopy(graph_dict['edge_index'])).long()
                graph_node_list = copy.deepcopy(graph_dict['node_list'])
                target_node = copy.deepcopy(graph_dict['node_idx'])
                graph_type = copy.deepcopy(self.list_data_dict[i]['id']).split('-')[0]

                if self.template == 'nd':
                    graph = torch.LongTensor(graph_node_list)
                    mask = (graph != -500)
                    masked_graph_emb = self.graph_emb_all[graph_type][graph[mask]]
                    n, d = graph.shape[0], masked_graph_emb.shape[1]
                    graph_node_rep = torch.zeros((n, d))
                    graph_node_rep[mask] = masked_graph_emb
                    if self.structure_emb is not None:
                        graph_node_rep = torch.cat([graph_node_rep, self.structure_emb], dim=-1)
                else:
                    graph_node_rep = self.graph_emb_all[graph_type][graph_node_list] ## 
                
                cur_token_len = len(graph_node_rep)   # FIXME: 14 is hardcoded patch size
                sources = preprocess_graph(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.graph_cfg, cur_token_len)
            else:
                sources = copy.deepcopy([e["conversations"] for e in sources])
        else: 
            if 'graph' in sources[0]:
                graph_dict = self.list_data_dict[i]['graph']
                graph_edge_index_1 = torch.Tensor(copy.deepcopy(graph_dict['edge_index_1'])).long()
                graph_node_list_1 = copy.deepcopy(graph_dict['node_list_1'])
                target_node_1 = copy.deepcopy(graph_dict['node_idx_1'])
                graph_type = copy.deepcopy(self.list_data_dict[i]['id']).split('-')[0]

                if self.template == 'nd':
                    graph_1 = torch.LongTensor(graph_node_list_1)
                    mask_1 = (graph_1 != -500)
                    masked_graph_emb_1 = self.graph_emb_all[graph_type][graph_1[mask_1]]
                    n, d = graph_1.shape[0], masked_graph_emb_1.shape[1]
                    graph_node_rep_1 = torch.zeros((n, d))
                    graph_node_rep_1[mask_1] = masked_graph_emb_1
                    if self.structure_emb is not None:
                        graph_node_rep_1 = torch.cat([graph_node_rep_1, self.structure_emb], dim=-1)
                elif self.template == 'ho':
                    center_id = graph_node_list[0]
                    graph = torch.LongTensor([center_id]*(4+1))
                    center_id = self.index[graph_type][graph[0]]
                    graph_node_rep = torch.stack([emb[center_id] for emb in self.pretrained_embs[graph_type]], dim=1)
                else:
                    graph_node_rep_1 = self.graph_emb_all[graph_type][graph_node_list_1] ##

                cur_token_len_1 = len(graph_node_rep_1)   # FIXME: 14 is hardcoded patch size

                graph_edge_index_2 = torch.Tensor(copy.deepcopy(graph_dict['edge_index_2'])).long()
                graph_node_list_2 = copy.deepcopy(graph_dict['node_list_2'])
                target_node_2 = copy.deepcopy(graph_dict['node_idx_2'])

                if self.template == 'nd':
                    graph_2 = torch.LongTensor(graph_node_list_2)
                    mask_2 = (graph_2 != -500)
                    masked_graph_emb_2 = self.graph_emb_all[graph_type][graph_2[mask_2]]
                    n, d = graph_2.shape[0], masked_graph_emb_2.shape[1]
                    graph_node_rep_2 = torch.zeros((n, d))
                    graph_node_rep_2[mask_2] = masked_graph_emb_2
                    if self.structure_emb is not None:
                        graph_node_rep_2 = torch.cat([graph_node_rep_2, self.structure_emb], dim=-1)
                elif self.template == 'ho':
                    center_id = graph_node_list[0]
                    graph = torch.LongTensor([center_id]*(4+1))
                    center_id = self.index[graph_type][graph[0]]
                    graph_node_rep = torch.stack([emb[center_id] for emb in self.pretrained_embs[graph_type]], dim=1)
                else:
                    graph_node_rep_2 = self.graph_emb_all[graph_type][graph_node_list_2] ## 
                cur_token_len_2 = len(graph_node_rep_2)   # FIXME: 14 is hardcoded patch size
                
                sources = preprocess_graph_LP(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.graph_cfg, cur_token_len_1, cur_token_len_2)
            else:
                sources = copy.deepcopy([e["conversations"] for e in sources])
        if task_type == 'nc_sp' or task_type == 'lp_sp':
            sources = preprocess_label(
                sources, 3)
        data_dict = preprocess(
            sources,
            self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'lp' not in task_type: 
            if 'graph' in self.list_data_dict[i]:
                data_dict['graph_data'] = Data(graph_node = graph_node_rep, edge_index=graph_edge_index, target_node = torch.tensor([target_node]))

            elif self.graph_cfg['is_graph']:
                # image does not exist in the data, but the model is multimodal
                node_feas = self.graph_cfg['graph_processor'].node_feas
                data_dict['graph_data'] = Data(graph_node = torch.zeros(3, node_feas), edge_index=torch.zeros(2, 3), target_node = torch.tensor([0]))
        else: 
            if 'graph' in self.list_data_dict[i]:
                data_dict['graph_data'] = {
                    'graph_1': Data(graph_node = graph_node_rep_1, edge_index=graph_edge_index_1, target_node = torch.tensor([target_node_1])), 
                    'graph_2': Data(graph_node = graph_node_rep_2, edge_index=graph_edge_index_2, target_node = torch.tensor([target_node_2]))
                    }

            elif self.graph_cfg['is_graph']:
                # image does not exist in the data, but the model is multimodal
                node_feas = self.graph_cfg['graph_processor'].node_feas
                data_dict['graph_data'] = Data(graph_node = torch.zeros(3, node_feas), edge_index=torch.zeros(2, 3), target_node = torch.tensor([0]))
        return data_dict
    
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'graph_data' in instances[0]:
            graph_data_batch = [instance['graph_data'] for instance in instances]
        batch['graph_data'] = graph_data_batch

        return batch
    
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (LazySupervisedDataset
                   if data_args.lazy_preprocess else SupervisedDataset)
    train_dataset = dataset_cls(tokenizer=tokenizer,
                                graph_cfg=dict(
                                    is_graph=data_args.is_graph,
                                    sep_graph_conv_front=data_args.sep_graph_conv_front,
                                    graph_token_len=data_args.graph_token_len,
                                    use_graph_start_end=getattr(data_args, 'use_graph_start_end', False)
                                    ), 
                                    data_path_prefix = data_args.data_path_prefix,
                                    use_dataset = data_args.use_dataset,
                                    use_task = data_args.use_task,
                                    template = data_args.template)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}

    ## load 4 8 bit 
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_int8_training
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))


    if model_args.graph_tower is not None:
    #if model_args.graph_token:
        if model_args.label_iaware:
            model = GraphLlamaForCausalLM_iaware.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                )
        else:
            model = GraphLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )

    model.config.pretrain_graph_model_path = model_args.graph_tower
    model.config.use_cache = False
    model.config.graph_hidden_size = 384
    if data_args.template == 'nd':
        model.config.graph_hidden_size += 111
    model.config.label_iaware = model_args.label_iaware

    if model_args.label_iaware:
        num_label_dict = {'cora':7, 'pubmed':3, 'arxiv':40, 'instagram':2}
        model.config.num_label = num_label_dict[data_args.use_dataset]
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_int8_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing and model_args.graph_tower is None:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        logging.warning("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    if model_args.graph_tower is not None:
    #if model_args.graph_token:
        model_graph_dict = model.get_model().initialize_graph_modules(
            graph_tower=model_args.graph_tower,
            graph_select_layer=model_args.graph_select_layer,
            pretrain_graph_mlp_adapter=model_args.pretrain_graph_mlp_adapter,
            fsdp=training_args.fsdp
        )
        
        if model_args.graph_tower is not None:
            model.get_graph_tower().to(dtype=torch.float16, device=training_args.device)

        data_args.is_graph = True

        model.config.tune_graph_mlp_adapter = training_args.tune_graph_mlp_adapter = model_args.tune_graph_mlp_adapter
        if model_args.tune_graph_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().graph_projector.parameters():
                p.requires_grad = True
            if model_args.label_iaware:
                for p in model.get_model().label_soft_prompt.parameters():
                    p.requires_grad = True
                for p in model.get_model().label_projector.parameters():
                    p.requires_grad = True

        model.config.freeze_graph_mlp_adapter = training_args.freeze_graph_mlp_adapter
        if training_args.freeze_graph_mlp_adapter:
            for p in model.get_model().graph_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().graph_projector.to(dtype=compute_dtype, device=training_args.device)
            if model_args.label_iaware:
                model.get_model().label_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.use_graph_start_end = data_args.use_graph_start_end = model_args.use_graph_start_end
        training_args.use_graph_start_end = model_args.use_graph_start_end
        model.config.sep_graph_conv_front = data_args.sep_graph_conv_front
        model.initialize_graph_tokenizer(use_graph_start_end=model_args.use_graph_start_end, tokenizer=tokenizer, device=training_args.device,
                                          tune_graph_mlp_adapter=model_args.tune_graph_mlp_adapter, pretrain_graph_mlp_adapter=model_args.pretrain_graph_mlp_adapter)
        params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
        if len(params_no_grad) > 0:
            if training_args.fsdp is not None and len(training_args.fsdp) > 0:
                if len(params_no_grad) < 10:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
                else:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'. format(len(params_no_grad), ', '.join(params_no_grad[:10])))
                print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
                print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
                def patch_FSDP_use_orig_params(func):
                    def wrap_func(*args, **kwargs):
                        use_orig_params = kwargs.pop('use_orig_params', True)
                        return func(*args, **kwargs, use_orig_params=use_orig_params)
                    return wrap_func

                FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                            data_args=data_args)
    trainer = GraphChatTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    
    print('************************** parameters: #', sum(p.numel() for p in model.parameters() if p.requires_grad))
    tuned_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            tuned_params.append(name)
    print(tuned_params)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    seed = 0
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ['OMP_NUM_THREADS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    train()
