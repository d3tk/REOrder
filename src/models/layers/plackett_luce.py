import torch
import torch.nn as nn
from src.models.layers.layers import get_permute_indices


def sample_gumbel(shape, device, eps=1e-8):
    u = torch.empty(shape, device=device).uniform_(0, 1)
    return -torch.log(-torch.log(u + eps) + eps)


class PlackettLucePolicy(nn.Module):
    def __init__(
        self,
        num_patches=196,
        method="gumbel",
        granularity="batch",
        logit_init="row-major",
    ):
        """
        Args:
            num_patches: How many items (patches) we have
            method: "iterative" or "gumbel"
            granularity: "batch" or "image"
            logit_init: Initialization scheme for logits. When patch_dir is RL, this determines the initial ordering.
                       Can be "row-major", "column-major", "hilbert-curve", etc.
        """
        super().__init__()
        permute_indices = get_permute_indices(
            order_needed=logit_init, num_patches=num_patches
        )
        if permute_indices is not None:
            initial_logits = torch.linspace(0, -1, num_patches)
            initial_logits = initial_logits[permute_indices]  # Shuffle
        else:
            # Default to row-major ordering
            initial_logits = torch.linspace(0, -1, num_patches)

        self.logits = nn.Parameter(initial_logits)
        self.num_patches = num_patches
        self.granularity = granularity

        # Possibly store a baseline for REINFORCE
        self.register_buffer("running_baseline", torch.tensor(0.0))

        assert method in [
            "iterative",
            "gumbel",
        ], "method must be 'iterative' or 'gumbel'"
        self.method = method
        self.temperature = 1.0

    def forward(self, dummy_input: torch.Tensor, batch_size: int = 1):
        """
        Returns:
          permutation: (tensor) shape [num_patches],
                       a sample from Plackett-Luce
          log_prob: (scalar) log pi(permutation)
        """
        if self.method == "iterative":
            if batch_size != 1:
                raise NotImplementedError(
                    "Batch sampling not implemented for 'iterative'."
                )
            permutation, log_prob = self._sample_iterative()
            return permutation.unsqueeze(0), log_prob.unsqueeze(0)
        else:
            return self._sample_gumbel(batch_size=batch_size)

    def _sample_iterative(self):
        """
        Naive O(n^2) iterative approach
        """
        logits = self.logits
        num_patches = logits.shape[0]

        available = list(range(num_patches))
        permutation = []
        log_prob = 0.0

        exp_logits = torch.exp(logits)

        for _ in range(num_patches):
            scores = torch.stack([exp_logits[i] for i in available])
            scores_sum = scores.sum()
            probs = scores / scores_sum

            chosen_idx = torch.multinomial(probs, 1).item()
            chosen_patch = available[chosen_idx]

            # accumulate log prob
            log_prob += torch.log(probs[chosen_idx] + 1e-8)

            permutation.append(chosen_patch)
            available.pop(chosen_idx)

        permutation = torch.tensor(permutation, dtype=torch.long, device=logits.device)
        return permutation, log_prob

    def _sample_gumbel(self, batch_size: int = 1):
        """
        O(n log n) approach:
          1) Add Gumbel noise to each logit
          2) Sort descending
          3) Use single-pass formula for log probability
        """
        logits = self.logits.unsqueeze(0)
        eps = 1e-8

        if self.granularity == "batch":
            # Sample the same permutation for all images
            g = sample_gumbel((1, self.num_patches), device=logits.device, eps=eps)
            g = g.repeat(batch_size, 1)
        elif self.granularity == "image":
            # Sample different permutations for all images by having different noise vectors
            g = sample_gumbel(
                (batch_size, self.num_patches), device=logits.device, eps=eps
            )

        z = logits + self.temperature * g  # Control the importance of the Gumbel noise

        sorted_indices = torch.argsort(z, dim=-1, descending=True)
        logits_sorted = torch.gather(
            logits.expand(batch_size, -1), dim=1, index=sorted_indices
        )

        log_cumsums_rev = torch.logcumsumexp(
            torch.flip(logits_sorted, dims=[-1]), dim=-1
        )
        log_denominators = torch.flip(log_cumsums_rev, dims=[-1])

        log_prob_terms = logits_sorted - log_denominators
        log_prob = log_prob_terms.sum(dim=-1)

        permutation = sorted_indices.contiguous()

        return permutation, log_prob, g
