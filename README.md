# REOrdering Patches Improves Vision Models

<p align="center">
  <img src="./docs/static/images/jigsaw.gif" alt="Duck Jigsaw Puzzle Gif" />
</p>


<div align="center">
  <a href="https://d3tk.github.io/REOrder"><img src="https://img.shields.io/badge/Homepage-blue" alt="Homepage"/></a>
  <a href="https://arxiv.org/abs/2505.23751"><img src="https://img.shields.io/badge/arXiv-2505.23751-b31b1b.svg" alt="arXiv"/></a>
  <a href="https://github.com/d3tk/REOrder/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-red" alt="License Badge"/></a>
</div>
</br>

Transformers for vision typically flatten images in a fixed row-major order, but this choice can significantly impact performance due to architectural approximations that are sensitive to patch order.

This repo introduces _REOrder_, a framework that discovers task-specific patch orderings by combining compressibility-based priors with learned permutation policies. _REOrder_ boosts accuracy on datasets like ImageNet-1K and Functional Map of the World, demonstrating that smarter patch sequencing can meaningfully improve transformer performance.

## Setup

With `conda`:

```shell
conda create -n reorder python=3.11
conda activate reorder
```

or with `pip`:

```shell
python3 -m venv .reorder
source .reorder/bin/activate
```

then install our required packages:

```shell
pip3 install torch torchvision torchaudio pyyaml omegaconf wandb gpustat transformers timm matplotlib numpy ninja pytest torchinfo
pip install "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@v1.5.0.post8"
pip install "git+https://github.com/hustvl/Vim.git@main#egg=mamba_ssm&subdirectory=mamba-1p1p1"
```

An `environment.yml` is also provided as well as a `requirements.txt`.

## Configs

`./configs/` contains all the configs used in training models for the paper. Configurations are set up using OmegaConf. All configurations needed to reproduce the experiments in the paper are provided.

### Paths

In `./configs/` there is a dir for a paths yaml. This yaml allows you define different paths for different hosts so that training can be launched on many devices. For each hostname you can define a path yaml. In `src/config/utils.py::get_path_config_for_hostname()` you can add a statement to resolve the new hostname to your new path yaml file.  

## Running Training

Distributed training can be launched with:
`torchrun --nproc_per_node=N main.py --config=path/to/config.yaml`

Set `--nproc_per_node` to the number of nodes. In our experiments, we use either 4x 40GB A100s or 8x 80GB A100s. Details are in the Appendix for each experiment.

We have tested this repo with AMD MI250 and MI300 GPUs. It works with the caveat that model compilation has to be turned off. Perhaps a future version of TorchDynamo will address this issue. None of our experiments rely on AMD GPUs, but if you do, this repo will work!

### Launching on Slurm

`./launch` conatains a set of [submitit](https://github.com/facebookincubator/submitit) scripts that are used to launch experiments on Slurm clusters. You will have to change Slurm arguments to ones that fit your cluster's setup.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{kutscher2025REOrder,
      title={REOrdering Patches Improves Vision Models}, 
      author={Declan Kutscher and David M. Chan and Yutong Bai and Trevor Darrell and Ritwik Gupta},
      year={2025},
      eprint={2505.23751},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.23751}, 
}
```
