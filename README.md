## 1. Environment Preparation: 

```shell
# create a new environment
conda create -n gallm python=3.10
conda activate gallm

# install pytorch. Modify the command to align with your own CUDA version.
pip3 install torch  --index-url https://download.pytorch.org/whl/cu118

# install related libraries
pip install -r requirements.txt

# install flash-attn
pip install flash-attn --no-build-isolation

# install pyg
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

## 2. Download Data

Download our datasets from [Box](https://utexas.box.com/s/i7y03rzm40xt9bjbaj0dfdgxeyjx77gb). Move the processed data to `./dataset`.


## 3. Tuning Models
**Template preparation:**

For Text-Graph Grounding template, to obtain the pre-trained GNNs , you can run the commands as follow (this is optional):

```shell
cd path/to/GALLM
python ./text-graph-grounding/main_train.py
```

For model tuning, you can run the scripts as follow:

**Stage 1 tuning:**

```shell
cd path/to/GALLM
sh ./scripts/tune_script/stage1.sh
```

**Stage 2 tuning:**

```shell
cd path/to/GALLM
sh ./scripts/tune_script/stage2.sh
```


## 3. Evaluation
for node classification tasks
 ```shell
cd path/to/GALLM
sh ./scripts/eval_script/eval.sh
python ./gallm/eval/eval_metrics.py
```
for link prediction tasks
```shell
cd path/to/GALLM
sh ./scripts/eval_script/eval_lp.sh
python ./gallm/eval/eval_metrics.py
```














