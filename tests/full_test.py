import os
import sys
import torch
import pytest
from torchinfo import summary

# add project root to path so that src/ can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.build import build_model
from src.models.layers.layers import get_permute_indices
from src.config.utils import load_and_build_config
from src.config.config_schema import Order

# Placeholder for a real image path (to be provided)
REAL_IMAGE_PATH = "/datasets/ilsvrc/current/train/n02093754/n02093754_6259.JPEG"

# Constants for full-size image testing
IMAGE_SIZE = 224
PATCH_SIZE = 16
GRID_DIM = IMAGE_SIZE // PATCH_SIZE  # 14
NUM_PATCHES = GRID_DIM * GRID_DIM  # 196

# All supported patch order types
PATCH_ORDERS = [
    "row-major",
    "column-major",
    "custom",
    "hilbert-curve",
    # "spiral-curve",
    # "peano-curve",
    "diagonal",
    "snake",
    # "random",
]

# Mapping of model names to their config files
MODEL_CONFIGS = {
    "txl": "configs/test-configs/txl.yaml",
    "longformer": "configs/test-configs/longformer.yaml",
    "mamba2": "configs/test-configs/mamba.yaml",
    "vit": "configs/test-configs/vit.yaml",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("model_name,config_path", MODEL_CONFIGS.items())
@pytest.mark.parametrize("patch_order", PATCH_ORDERS)
def test_patch_embed_order_and_plot(model_name, config_path, patch_order, tmp_path):
    """
    Build each model with the specified patch_order, run summary, then:
      - Use a dummy indexed image to verify the patch shuffle order matches get_permute_indices.
      - Load a real image (placeholder path) to extract and plot actual image patches.
    """
    print(f"testing {model_name} with {patch_order}, config: {config_path}")

    # Load and override config
    assert os.path.exists(config_path), f"Config not found: {config_path}"

    cfg = load_and_build_config(config_path)
    # print(cfg["model"]["patch_dir"])
    # map patch_order to Order enum
    cfg["model"]["patch_dir"] = Order(patch_order)

    # For custom patch ordering, we need to provide permute_indices
    if patch_order == "custom":
        cfg["model"]["custom_permute"] = list(range(NUM_PATCHES - 1, -1, -1))

    # Instantiate model and run summary
    model = build_model(cfg).to(DEVICE)

    summary(model, input_size=(1, 3, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE, depth=5)

    # Dummy image for verification: each pixel stores its patch index
    rows = torch.arange(IMAGE_SIZE).unsqueeze(1).expand(IMAGE_SIZE, IMAGE_SIZE)
    cols = torch.arange(IMAGE_SIZE).unsqueeze(0).expand(IMAGE_SIZE, IMAGE_SIZE)
    block_idx = (rows // PATCH_SIZE) * GRID_DIM + (cols // PATCH_SIZE)
    dummy = block_idx.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    dummy = dummy.expand(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    # Extract and verify observed order
    if model_name == "txl":
        patches = model.word_emb.patchify(dummy)
    else:
        patches = model.patch_embed.patchify(dummy)
    observed = patches[0, :, 0, 0, 0]  # → shape [196]
    observed = observed.detach().cpu().long().tolist()

    if patch_order == "custom":
        expected_perm = list(range(NUM_PATCHES - 1, -1, -1))
    else:
        expected_perm = get_permute_indices(patch_order, NUM_PATCHES)
        if expected_perm is None:
            expected_perm = list(range(NUM_PATCHES))
        elif hasattr(expected_perm, "tolist"):
            expected_perm = expected_perm.tolist()
        else:
            expected_perm = list(expected_perm)

    assert (
        observed == expected_perm
    ), f"{model_name}: for order '{patch_order}', expected {expected_perm[:10]}..., got {observed[:10]}..."


@pytest.mark.parametrize("model_name,config_path", MODEL_CONFIGS.items())
def test_patch_embed_custom_permutation(
    model_name, config_path, patch_size=16, image_size=224, device="cuda"
):
    print(f"testing {model_name}, config: {config_path}")

    # Setup
    grid_dim = image_size // patch_size
    num_patches = grid_dim * grid_dim

    # Load your model config (adjust path as needed)
    assert os.path.exists(config_path), f"Config not found: {config_path}"

    cfg = load_and_build_config(config_path)

    model = build_model(cfg).to(device)

    # Dummy image: each patch index is encoded in the pixel value
    rows = torch.arange(image_size).unsqueeze(1).expand(image_size, image_size)
    cols = torch.arange(image_size).unsqueeze(0).expand(image_size, image_size)
    block_idx = (rows // patch_size) * grid_dim + (cols // patch_size)
    dummy = block_idx.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    dummy = dummy.expand(1, 3, image_size, image_size).to(device)

    # 1. Get default patch order
    if model_name == "txl":
        patches_default = model.word_emb.patchify(dummy)
    else:
        patches_default = model.patch_embed.patchify(dummy)
    observed_default = patches_default[0, :, 0, 0, 0].detach().cpu().long().tolist()

    # 2. Create a custom permutation (e.g., reversed order)
    custom_perm = torch.arange(num_patches - 1, -1, -1, device=device)
    # Pass the custom permutation to the model
    if model_name == "txl":
        patches_custom = model.word_emb.patchify(
            dummy, perm=custom_perm.unsqueeze(0).expand(1, -1)
        )
    else:
        patches_custom = model.patch_embed.patchify(
            dummy, perm=custom_perm.unsqueeze(0).expand(1, -1)
        )
    observed_custom = patches_custom[0, :, 0, 0, 0].detach().cpu().long().tolist()

    # 3. Check that the custom permutation is respected
    assert observed_custom == custom_perm.cpu().tolist(), (
        f"Custom permutation not respected! "
        f"Expected {custom_perm[:10].cpu().tolist()}..., got {observed_custom[:10]}..."
    )

    # 4. Check that the default and custom orders are different (unless the default is also reversed)
    if not torch.equal(
        custom_perm, torch.arange(num_patches, device=custom_perm.device)
    ):
        assert (
            observed_default != observed_custom
        ), "Default and custom patch orders should differ!"

    print("Default patch order (first 10):", observed_default[:10])
    print("Custom patch order (first 10):", observed_custom[:10])
    print("Test passed: custom permutation is respected.")


@pytest.mark.parametrize("model_name,config_path", MODEL_CONFIGS.items())
def test_pos_emb_permutation(
    model_name, config_path, patch_size=16, image_size=224, device="cuda"
):
    print(f"Testing positional embedding permutation for {model_name}")

    # Setup
    grid_dim = image_size // patch_size
    num_patches = grid_dim * grid_dim

    # Load model config
    assert os.path.exists(config_path), f"Config not found: {config_path}"
    cfg = load_and_build_config(config_path)
    model = build_model(cfg).to(device)
    model.eval()

    # Dummy image
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    # Custom permutation (reverse order)
    custom_perm = torch.arange(num_patches - 1, -1, -1, device=device).unsqueeze(0)

    # Run a full forward pass with custom perm
    with torch.no_grad():
        # If your model supports passing perm through the forward, do so:
        output = model(dummy, perm=custom_perm)
        # If not, you may need to set the perm in the patch embedding module directly, or adapt as needed.

    print("Forward pass complete. Check above for print statements from model.")
