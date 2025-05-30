import json
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


# Accuracy computation function for Top-K from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/metrics.py#L25
def accuracy(output, target, top_k=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(top_k), output.size()[1])

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[: min(k, maxk)].reshape(-1).float().sum(0) for k in top_k]


__opt_dict__ = {
    "AdamW": torch.optim.AdamW,
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}

__sch_dict__ = {
    "LinearLR": torch.optim.lr_scheduler.LinearLR,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    # Some other popular schedulers I found
    "StepLR": torch.optim.lr_scheduler.StepLR,
    "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
    "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "CyclicLR": torch.optim.lr_scheduler.CyclicLR,
    "OneCycleLR": torch.optim.lr_scheduler.OneCycleLR,
    "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "PolynomialLR": torch.optim.lr_scheduler.PolynomialLR,
}


class PolicyWeightScheduler(nn.Module):
    """
    Adjusts the policy weight during training. Go from starting_weight to ending_weight over the
    course of warmup_epochs, and then stay there until the end.
    """

    def __init__(
        self,
        starting_weight: float,
        ending_weight: float,
        steps_per_epoch: int,
        num_epochs: int,
        warmup_epochs: int,
    ):
        super().__init__()
        self.starting_weight = starting_weight
        self.ending_weight = ending_weight
        self.steps_per_epoch = steps_per_epoch
        self.num_epochs = num_epochs

        self.total_steps = num_epochs * steps_per_epoch

        self.register_buffer("current_step", torch.tensor(0, dtype=torch.long))

        self.warmup_epochs = warmup_epochs
        self.warmup_steps = warmup_epochs * steps_per_epoch

        if self.warmup_steps > self.total_steps:
            raise ValueError("Warmup steps must be <= total training steps.")

        # Build taus with warmup (exponential increase) + hold
        self.weights = []
        for t in range(self.total_steps + 1):
            if t <= self.warmup_steps:
                # Exponential warmup from starting to ending temp
                weight = self.starting_weight * math.exp(
                    t
                    * math.log(self.ending_weight / self.starting_weight)
                    / self.warmup_steps
                )
            else:
                # Hold ending temp after warmup
                weight = self.ending_weight
            self.weights.append(weight)
        self.weights = torch.tensor(self.weights)

        self.current_weight = self.weights[self.current_step]

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1

    def get_current_weight(self) -> float:
        step = torch.clamp(self.current_step, 0, self.total_steps).item()
        return self.weights[step].item()


class PolicyGumbelTempScheduler(nn.Module):
    """
    Adjusts the policy Gumbel noise temperature over time.
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        starting_temp: float,
        ending_temp: float,
        steps_per_epoch: int,
        running_epochs: int,
        num_epochs: int,
        dist: bool,
        decay_type: str = "exponential",
    ):
        super().__init__()
        self.policy_ref = policy
        self.starting_temp = starting_temp
        self.ending_temp = ending_temp
        self.steps_per_epoch = steps_per_epoch
        self.running_epochs = running_epochs
        self.num_epochs = num_epochs
        self.dist = dist
        self.decay_type = decay_type

        self.decay_steps = self.running_epochs * self.steps_per_epoch
        self.total_defined_steps = self.num_epochs * self.steps_per_epoch

        self.register_buffer("current_step", torch.tensor(0, dtype=torch.long))

        taus_tensor = self._build_schedule()
        self.register_buffer("taus", taus_tensor)

        self._set_policy_tau(self.get_current_tau())

    def _build_schedule(self) -> torch.Tensor:
        total_steps_for_schedule = self.total_defined_steps
        if self.decay_steps == 0:
            val = 0.0 if self.ending_temp == 0.0 else self.ending_temp
            return torch.full((total_steps_for_schedule + 1,), val, dtype=torch.float32)

        decay_taus_list = []
        if self.decay_type == "triangular":
            peak = self.starting_temp
            half = self.decay_steps // 2
            for t in range(self.decay_steps):
                if t < half:
                    frac = t / max(half - 1, 1)
                    tau_t = peak * frac
                else:
                    frac = (t - half) / max(half - 1, 1)
                    tau_t = peak * (1.0 - frac)
                decay_taus_list.append(tau_t)

        elif self.decay_type == "plateau":
            peak = self.starting_temp
            plateau_steps = int(0.7 * self.decay_steps)
            current_decay_steps = self.decay_steps - plateau_steps
            decay_taus_list.extend([peak] * plateau_steps)
            for t in range(current_decay_steps):
                frac = t / max(current_decay_steps - 1, 1)
                tau_t = peak * (1.0 - frac)
                decay_taus_list.append(tau_t)

        elif self.decay_type == "exponential":
            if self.ending_temp == self.starting_temp:
                decay_taus_list = [self.starting_temp] * self.decay_steps
            elif self.ending_temp != 0.0:
                for t in range(self.decay_steps):
                    frac = t / max(self.decay_steps - 1, 1)
                    tau_t = self.starting_temp * math.exp(
                        frac * math.log(self.ending_temp / self.starting_temp)
                    )
                    decay_taus_list.append(tau_t)
            else:
                eps = 1.0e-12
                actual_starting_temp = max(self.starting_temp, eps)
                for t in range(self.decay_steps):
                    frac = t / max(self.decay_steps - 1, 1)
                    tau_t = actual_starting_temp * math.exp(
                        frac * math.log(eps / actual_starting_temp)
                    )
                    decay_taus_list.append(tau_t)

        elif self.decay_type == "linear":
            for t in range(self.decay_steps):
                frac = t / max(self.decay_steps - 1, 1)
                tau_t = self.starting_temp + frac * (
                    self.ending_temp - self.starting_temp
                )
                decay_taus_list.append(tau_t)
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")

        remaining_steps = total_steps_for_schedule - self.decay_steps
        if remaining_steps < 0:
            pass

        if len(decay_taus_list) < total_steps_for_schedule + 1:
            num_to_pad = (total_steps_for_schedule + 1) - len(decay_taus_list)
            padding_value = (
                decay_taus_list[-1]
                if (self.decay_steps > 0 and remaining_steps <= 0)
                else self.ending_temp
            )
            decay_taus_list.extend([padding_value] * num_to_pad)

        return torch.tensor(
            decay_taus_list[: total_steps_for_schedule + 1], dtype=torch.float32
        )

    def step(self):
        if self.current_step < self.total_defined_steps:
            self.current_step += 1
        self._set_policy_tau(self.get_current_tau())

    def get_current_tau(self) -> float:
        step = torch.clamp(self.current_step, 0, self.total_defined_steps).item()
        return self.taus[step].item()

    def _set_policy_tau(self, tau_value: float):
        if hasattr(self, "policy_ref") and self.policy_ref is not None:
            if self.dist:
                if hasattr(self.policy_ref, "module"):
                    self.policy_ref.module.temperature = tau_value
                else:
                    self.policy_ref.temperature = tau_value
            else:
                self.policy_ref.temperature = tau_value


def build_optimizer(opt_config: dict, params) -> torch.optim.Optimizer:
    optimizer = __opt_dict__[opt_config["name"]]
    lr = opt_config["base_lr"] * math.sqrt(get_world_size())

    return optimizer(
        params,
        lr=lr,
        weight_decay=opt_config["weight_decay"],
    )


def build_scheduler(config, optimizer, steps_per_epoch):
    sched_cfg = config["scheduler"]
    warm_epochs = sched_cfg["warmup_epochs"]
    num_epochs = config["training"]["num_epochs"]

    if config.get("reinforce", None):
        start_after = config["reinforce"].get("start_after", warm_epochs)
        run_epochs = (
            config["reinforce"]
            .get("policy_gumbel_temp_scheduler", {})
            .get("running_epochs", 0)
        )
    else:
        start_after = warm_epochs  # decay right after warm-up
        run_epochs = 0

    total_steps = num_epochs * steps_per_epoch
    warm_steps = warm_epochs * steps_per_epoch
    flat1_steps = max(0, start_after - warm_epochs) * steps_per_epoch
    flat2_steps = max(0, run_epochs) * steps_per_epoch
    decay_steps = total_steps - warm_steps - flat1_steps - flat2_steps

    base_lr = optimizer.param_groups[0]["lr"]
    min_lr = sched_cfg["min_lr_ratio"] * base_lr

    lr_values = []
    lr_values += [base_lr * (t + 1) / warm_steps for t in range(warm_steps)]  # Warmup
    lr_values += [base_lr] * flat1_steps  # Hold until policy starts policy
    lr_values += [base_lr] * flat2_steps  # Hold while policy is running
    # cosine decay afterwards
    for t in range(decay_steps):
        cos = 0.5 * (1 + math.cos(math.pi * t / max(1, decay_steps - 1)))
        lr_values.append(min_lr + (base_lr - min_lr) * cos)

    lr_tensor = torch.tensor(lr_values, dtype=torch.float32, device="cpu")
    assert len(lr_tensor) == total_steps, "schedule length mismatch"

    def lr_lambda(step):
        # Adjust index to be 0-based (step is 1-based)
        # Clamp step to avoid negative index if step is 0 (though LambdaLR starts step at 1)
        safe_step = max(1, step)
        return lr_tensor[safe_step - 1].item() / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_policy_weight_scheduler(
    starting_weight: float,
    ending_weight: float,
    steps_per_epoch: int,
    num_epochs: int,
    warmup_epochs: int,
) -> PolicyWeightScheduler:
    return PolicyWeightScheduler(
        starting_weight=starting_weight,
        ending_weight=ending_weight,
        steps_per_epoch=steps_per_epoch,
        num_epochs=num_epochs,
        warmup_epochs=warmup_epochs,
    )


def build_policy_gumbel_temp_scheduler(
    policy: torch.nn.Module,
    starting_temp: float,
    ending_temp: float,
    steps_per_epoch: int,
    running_epochs: int,
    num_epochs: int,
    dist: bool,
    decay_type: str = "exponential",
) -> PolicyGumbelTempScheduler:
    return PolicyGumbelTempScheduler(
        policy=policy,
        starting_temp=starting_temp,
        ending_temp=ending_temp,
        steps_per_epoch=steps_per_epoch,
        running_epochs=running_epochs,
        num_epochs=num_epochs,
        dist=dist,
        decay_type=decay_type,
    )


def build_policy_schedulers(
    policy: torch.nn.Module,
    steps_per_epoch: int,
    config: dict,
) -> Tuple[PolicyWeightScheduler, PolicyGumbelTempScheduler]:
    num_epochs = config["training"]["num_epochs"] - config["reinforce"]["start_after"]
    if config["reinforce"].get("policy_weight_scheduler", None):
        policy_weight_scheduler = build_policy_weight_scheduler(
            starting_weight=config["reinforce"]["policy_weight_scheduler"][
                "starting_weight"
            ],
            ending_weight=config["reinforce"]["policy_weight_scheduler"][
                "ending_weight"
            ],
            steps_per_epoch=steps_per_epoch,
            num_epochs=num_epochs,
            warmup_epochs=config["reinforce"]["policy_weight_scheduler"][
                "warmup_epochs"
            ],
        )
    else:
        policy_weight_scheduler = None
    policy_gumbel_temp_scheduler = build_policy_gumbel_temp_scheduler(
        policy=policy,
        starting_temp=config["reinforce"]["policy_gumbel_temp_scheduler"][
            "starting_temp"
        ],
        ending_temp=config["reinforce"]["policy_gumbel_temp_scheduler"]["ending_temp"],
        steps_per_epoch=steps_per_epoch,
        running_epochs=int(
            config["reinforce"]["policy_gumbel_temp_scheduler"]["running_epochs"]
        ),
        num_epochs=num_epochs,
        dist=config["training"]["dist"],
        decay_type=config["reinforce"]["policy_gumbel_temp_scheduler"]["decay_type"],
    )

    return policy_weight_scheduler, policy_gumbel_temp_scheduler


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


# from: https://github.com/bair-climate-initiative/xT/blob/main/xt/utils.py
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


# from: https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/utils/log_uniform_sampler.py
class LogUniformSampler(object):
    def __init__(self, range_max, n_sample):
        """
        Reference : https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/candidate_sampling_ops.py
            `P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)`

        expected count can be approximated by 1 - (1 - p)^n
        and we use a numerically stable version -expm1(num_tries * log1p(-p))

        Our implementation fixes num_tries at 2 * n_sample, and the actual #samples will vary from run to run
        """
        with torch.no_grad():
            self.range_max = range_max
            log_indices = torch.arange(1.0, range_max + 2.0, 1.0).log_()
            self.dist = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]
            # print('P', self.dist.numpy().tolist()[-30:])

            self.log_q = (
                (-(-self.dist.double().log1p_() * 2 * n_sample).expm1_()).log_().float()
            )

        self.n_sample = n_sample

    def sample(self, labels):
        """
            labels: [b1, b2]
        Return
            true_log_probs: [b1, b2]
            samp_log_probs: [n_sample]
            neg_samples: [n_sample]
        """

        # neg_samples = torch.empty(0).long()
        n_sample = self.n_sample
        n_tries = 2 * n_sample

        with torch.no_grad():
            neg_samples = torch.multinomial(
                self.dist, n_tries, replacement=True
            ).unique()
            device = labels.device
            neg_samples = neg_samples.to(device)
            true_log_probs = self.log_q[labels].to(device)
            samp_log_probs = self.log_q[neg_samples].to(device)
            return true_log_probs, samp_log_probs, neg_samples


def sample_logits(embedding, bias, labels, inputs, sampler):
    """
        embedding: an nn.Embedding layer
        bias: [n_vocab]
        labels: [b1, b2]
        inputs: [b1, b2, n_emb]
        sampler: you may use a LogUniformSampler
    Return
        logits: [b1, b2, 1 + n_sample]
    """
    true_log_probs, samp_log_probs, neg_samples = sampler.sample(labels)
    n_sample = neg_samples.size(0)
    b1, b2 = labels.size(0), labels.size(1)
    all_ids = torch.cat([labels.view(-1), neg_samples])
    all_w = embedding(all_ids)
    true_w = all_w[:-n_sample].view(b1, b2, -1)
    sample_w = all_w[-n_sample:].view(n_sample, -1)

    all_b = bias[all_ids]
    true_b = all_b[:-n_sample].view(b1, b2)
    sample_b = all_b[-n_sample:]

    hit = (labels[:, :, None] == neg_samples).detach()

    true_logits = (
        torch.einsum("ijk,ijk->ij", [true_w, inputs]) + true_b - true_log_probs
    )
    sample_logits = (
        torch.einsum("lk,ijk->ijl", [sample_w, inputs]) + sample_b - samp_log_probs
    )
    sample_logits.masked_fill_(hit, -1e30)
    logits = torch.cat([true_logits[:, :, None], sample_logits], -1)

    return logits


# from: https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/utils/proj_adaptive_softmax.py
class ProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, keep_order=False):
        super(ProjectedAdaptiveLogSoftmax, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(
                torch.zeros(self.n_clusters, self.d_embed)
            )
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()

        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    self.out_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
                else:
                    self.out_projs.append(None)

            self.out_layers.append(nn.Linear(d_embed, n_token))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val**i)

                self.out_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

                self.out_layers.append(nn.Linear(d_emb_i, r_idx - l_idx))

        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            # if CUDA_MAJOR <= 9 and CUDA_MINOR <= 1:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logit = F.linear(proj_hid, weight, bias=bias)
            # else:
            #     logit = torch.einsum('bd,de,ev->bv', (hidden, proj, weight.t()))
            #     if bias is not None:
            #         logit = logit + bias

        return logit

    def forward(self, hidden, target, keep_order=False):
        """
        hidden :: [len*bsz x d_proj]
        target :: [len*bsz]
        """

        if hidden.size(0) != target.size(0):
            raise RuntimeError(
                "Input and target should have the same size " "in the batch dimension."
            )

        if self.n_clusters == 0:
            logit = self._compute_logit(
                hidden,
                self.out_layers[0].weight,
                self.out_layers[0].bias,
                self.out_projs[0],
            )
            nll = (
                -F.log_softmax(logit, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
            )
        else:
            # construct weights and biases
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias

                if i == 0:
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]

            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            head_logprob = F.log_softmax(head_logit, dim=1)

            nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)

            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

                mask_i = (target >= l_idx) & (target < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                target_i = target.index_select(0, indices_i) - l_idx
                head_logprob_i = head_logprob.index_select(0, indices_i)

                if i == 0:
                    logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]

                    hidden_i = hidden.index_select(0, indices_i)

                    tail_logit_i = self._compute_logit(
                        hidden_i, weight_i, bias_i, proj_i
                    )
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)

                    logprob_i = head_logprob_i[:, -i] + tail_logprob_i.gather(
                        1, target_i[:, None]
                    ).squeeze(1)

                if (hasattr(self, "keep_order") and self.keep_order) or keep_order:
                    nll.index_copy_(0, indices_i, -logprob_i)
                else:
                    nll[offset : offset + logprob_i.size(0)].copy_(-logprob_i)

                offset += logprob_i.size(0)

        return nll
