# Eval image tasks
SUPPORTED_IMAGE_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg")
LORA_SDXL_WORKFLOW_PATH = "validator/evaluation/comfy_workflows/lora_sdxl.json"
LORA_SDXL_WORKFLOW_PATH_DIFFUSERS = "validator/evaluation/comfy_workflows/lora_sdxl_diffusers.json"
LORA_FLUX_WORKFLOW_PATH = "validator/evaluation/comfy_workflows/lora_flux.json"
CHECKPOINTS_SAVE_PATH = "validator/evaluation/ComfyUI/models/checkpoints"
UNET_SAVE_PATH = "validator/evaluation/ComfyUI/models/unet"
DIFFUSERS_PATH = "validator/evaluation/ComfyUI/models/diffusers"
LORAS_SAVE_PATH = "validator/evaluation/ComfyUI/models/loras"
DIFFUSION_HF_DEFAULT_FOLDER = "checkpoint"
DIFFUSION_HF_DEFAULT_CKPT_NAME = "last.safetensors"
DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT = 0.25
EVAL_DEFAULTS = {"sdxl": {"steps": 20, "cfg": 8, "denoise": 0.9}, "flux": {"steps": 35, "cfg": 100, "denoise": 0.75}}

# Eval text tasks
DPO_DEFAULT_DATASET_TYPE = "chatml.intel"
VALI_CONFIG_PATH = "validator/test_axolotl.yml"
DOCKER_EVAL_HF_CACHE_DIR = "/root/.cache/huggingface"
CONTAINER_EVAL_RESULTS_PATH = "/workspace/evaluation_results.json"
CONTAINER_EVAL_RESULTS_PATH_SYNTHETIC = "/workspace/evaluation_results_synthetic.json"

# Eval DPO tasks
TRL_DPO_FIELD_PROMPT = "prompt"
TRL_DPO_FIELD_CHOSEN = "chosen"
TRL_DPO_FIELD_REJECTED = "rejected"

# GRPO evaluation
GRPO_DEFAULT_FIELD_PROMPT = "prompt"
TRL_GRPO_FIELD_PROMPT = GRPO_DEFAULT_FIELD_PROMPT

# Default, fixed Hyperparameters
BETA_DPO = 0.1
BETA_GRPO = 0.04

# GRPO evaluation
GRPO_INITIAL_BATCH_SIZE = 32
GRPO_DEFAULT_NUM_GENERATIONS = 2