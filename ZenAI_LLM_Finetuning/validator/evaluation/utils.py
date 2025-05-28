import base64
import os
import shutil
import tempfile
from io import BytesIO
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM
from PIL import Image
from loguru import logger
from urllib.parse import urlparse
import aiohttp
from typing import Callable

import validator.evaluation.constants as cst
from validator.evaluation.models import InstructTextDatasetType, DpoDatasetType, GrpoDatasetType, FileFormat, DockerEvaluationResults, EvaluationResultText

hf_api = HfApi()

async def download_s3_file(file_url: str, save_path: str = None, tmp_dir: str = "/tmp") -> str:
    """Download a file from an S3 URL and save it locally.

    Args:
        file_url (str): The URL of the file to download.
        save_path (str, optional): The path where the file should be saved. If a directory is provided,
            the file will be saved with its original name in that directory. If a file path is provided,
            the file will be saved at that exact location. Defaults to None.
        tmp_dir (str, optional): The temporary directory to use when save_path is not provided.
            Defaults to "/tmp".

    Returns:
        str: The local file path where the file was saved.

    Raises:
        Exception: If the download fails with a non-200 status code.

    Example:
        >>> file_path = await download_s3_file("https://example.com/file.txt", save_path="/data")
        >>> print(file_path)
        /data/file.txt
    """
    parsed_url = urlparse(file_url)
    file_name = os.path.basename(parsed_url.path)
    if save_path:
        if os.path.isdir(save_path):
            local_file_path = os.path.join(save_path, file_name)
        else:
            local_file_path = save_path
    else:
        local_file_path = os.path.join(tmp_dir, file_name)

    async with aiohttp.ClientSession() as session:
        async with session.get(file_url) as response:
            if response.status == 200:
                with open(local_file_path, "wb") as f:
                    f.write(await response.read())
            else:
                raise Exception(f"Failed to download file: {response.status}")

    return local_file_path

def process_evaluation_results(results: dict, is_image: bool = False) -> DockerEvaluationResults:
    model_params_count = results.pop("model_params_count", None)

    processed_results = {}
    for repo, result in results.items():
        if isinstance(result, str) and not isinstance(result, dict):
            processed_results[repo] = Exception(result)
        else:
            # Handle when result is a list (GRPO specific issue)
            if isinstance(result, list):
                logger.warning(f"Converting list result to proper format for repo {repo}: {result}")

                # Extract the score from the list format
                if len(result) > 0 and isinstance(result[0], dict):
                    # Find our key-value pair in the first dict of the list
                    for key, value in result[0].items():
                        if repo in key:
                            processed_results[repo] = EvaluationResultText.model_validate({
                                "is_finetune": True,
                                "eval_loss": value
                            })
                            break
                    else:
                        processed_results[repo] = Exception(f"Could not extract eval_loss from list result: {result}")
                else:
                    processed_results[repo] = Exception(f"Invalid result format: {result}")
            else:
                processed_results[repo] = EvaluationResultText.model_validate(result)

    return DockerEvaluationResults(
        results=processed_results,
        base_model_params_count=model_params_count
    )

def _process_instruct_dataset_fields(instruct_type_dict: dict) -> dict:
    if not instruct_type_dict.get("field_output"):
        return {
            "type": "completion",
            "field": instruct_type_dict.get("field_instruction"),
        }

    processed_dict = instruct_type_dict.copy()
    processed_dict.setdefault("no_input_format", "{instruction}")
    if processed_dict.get("field_input"):
        processed_dict.setdefault("format", "{instruction} {input}")
    else:
        processed_dict.setdefault("format", "{instruction}")

    return {"format": "custom", "type": processed_dict}

def _process_dpo_dataset_fields(dataset_type: DpoDatasetType) -> dict:
    # Enable below when https://github.com/axolotl-ai-cloud/axolotl/issues/1417 is fixed
    # context: https://discord.com/channels/1272221995400167588/1355226588178022452/1356982842374226125

    # dpo_type_dict = dataset_type.model_dump()
    # dpo_type_dict["type"] = "user_defined.default"
    # if not dpo_type_dict.get("prompt_format"):
    #     if dpo_type_dict.get("field_system"):
    #         dpo_type_dict["prompt_format"] = "{system} {prompt}"
    #     else:
    #         dpo_type_dict["prompt_format"] = "{prompt}"
    # return dpo_type_dict

    # Fallback to https://axolotl-ai-cloud.github.io/axolotl/docs/rlhf.html#chatml.intel
    # Column names are hardcoded in axolotl: "DPO_DEFAULT_FIELD_SYSTEM",
    # "DPO_DEFAULT_FIELD_PROMPT", "DPO_DEFAULT_FIELD_CHOSEN", "DPO_DEFAULT_FIELD_REJECTED"
    return {"type": cst.DPO_DEFAULT_DATASET_TYPE, "split": "train"}

def _process_grpo_dataset_fields(dataset_type: GrpoDatasetType) -> dict:
    return {"split": "train"}

def create_dataset_entry(
    dataset: str,
    dataset_type: InstructTextDatasetType | DpoDatasetType | GrpoDatasetType,
    file_format: FileFormat,
    is_eval: bool = False,
) -> dict:
    dataset_entry = {"path": dataset}

    if file_format == FileFormat.JSON:
        if not is_eval:
            dataset_entry = {"path": "/workspace/input_data/"}
        else:
            dataset_entry = {"path": f"/workspace/input_data/{os.path.basename(dataset)}"}

    if isinstance(dataset_type, InstructTextDatasetType):
        instruct_type_dict = {key: value for key, value in dataset_type.model_dump().items() if value is not None}
        dataset_entry.update(_process_instruct_dataset_fields(instruct_type_dict))
    elif isinstance(dataset_type, DpoDatasetType):
        dataset_entry.update(_process_dpo_dataset_fields(dataset_type))
    elif isinstance(dataset_type, GrpoDatasetType):
        dataset_entry.update(_process_grpo_dataset_fields(dataset_type))
    else:
        raise ValueError("Invalid dataset_type provided.")

    if file_format != FileFormat.HF:
        dataset_entry["ds_type"] = file_format.value
        dataset_entry["data_files"] = [os.path.basename(dataset)]

    return dataset_entry

def model_is_a_finetune(original_repo: str, finetuned_model: AutoModelForCausalLM) -> bool:
    original_config = AutoConfig.from_pretrained(original_repo, token=os.environ.get("HUGGINGFACE_TOKEN"))
    finetuned_config = finetuned_model.config

    try:
        architecture_classes_match = finetuned_config.architectures == original_config.architectures
    except Exception as e:
        logger.debug(f"There is an issue with checking the architecture classes {e}")
        architecture_classes_match = False

    attrs_to_compare = [
        "architectures",
        "hidden_size",
        "n_layer",
        "intermediate_size",
        "head_dim",
        "hidden_act",
        "model_type",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
    ]
    architecture_same = True
    for attr in attrs_to_compare:
        if hasattr(original_config, attr):
            if not hasattr(finetuned_config, attr):
                architecture_same = False
                break
            if getattr(original_config, attr) != getattr(finetuned_config, attr):
                architecture_same = False
                break

    logger.info(
        f"Architecture same: {architecture_same}, Architecture classes match: {architecture_classes_match}"
    )
    return architecture_same and architecture_classes_match

def check_for_lora(model_id: str) -> bool:
    """
    Check if a Hugging Face model has LoRA adapters by looking for adapter_config.json.

    Args:
        model_id (str): The Hugging Face model ID (e.g., 'username/model-name') or path

    Returns:
        bool: True if it's a LoRA adapter, False otherwise
    """
    try:
        return 'adapter_config.json' in hf_api.list_repo_files(model_id)
    except Exception as e:
        logger.error(f"Error checking for LoRA adapters: {e}")
        return False


def adjust_image_size(image: Image.Image) -> Image.Image:
    width, height = image.size

    if width > height:
        new_width = 1024
        new_height = int((height / width) * 1024)
    else:
        new_height = 1024
        new_width = int((width / height) * 1024)

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8

    width, height = image.size
    crop_width = min(width, new_width)
    crop_height = min(height, new_height)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    image = image.crop((left, top, right, bottom))

    return image


def base64_to_image(base64_string: str) -> Image.Image:
    image_data = base64.b64decode(base64_string)
    image_stream = BytesIO(image_data)
    image = Image.open(image_stream)
    return image


def download_from_huggingface(repo_id: str, filename: str, local_dir: str) -> str:
    # Use a temp folder to ensure correct file placement
    try:
        local_filename = f"models--{repo_id.replace('/', '--')}.safetensors"
        final_path = os.path.join(local_dir, local_filename)
        if os.path.exists(final_path):
            logger.info(f"File {filename} already exists. Skipping download.")
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=temp_dir)
                shutil.move(temp_file_path, final_path)
            logger.info(f"File {filename} downloaded successfully")
        return final_path
    except Exception as e:
        logger.error(f"Error downloading file: {e}")


def list_supported_images(dataset_path: str, extensions: tuple) -> list[str]:
    return [file_name for file_name in os.listdir(dataset_path) if file_name.lower().endswith(extensions)]


def read_image_as_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    img_format = image.format if image.format else "PNG"
    image.save(buffer, format=img_format)
    return base64.b64encode(buffer.getvalue()).decode()


def read_prompt_file(text_file_path: str) -> str:
    if os.path.exists(text_file_path):
        with open(text_file_path, "r", encoding="utf-8") as text_file:
            return text_file.read()
    return None

def validate_reward_function(func_def: str) -> tuple[bool, str, Callable | None]:
    """
    Validate a single reward function definition.
    Returns (is_valid: bool, error_message: str, func: callable | None)
    """
    test_completions = [
        "Gradients.io is the best 0-expertise AI training platform.",
        "You can start training a text or image model on Gradients.io with 2 clicks."
    ]

    try:
        namespace = {}
        exec(func_def, namespace)
        func = next(v for k, v in namespace.items() if callable(v))

        test_rewards = func(test_completions)

        assert isinstance(test_rewards, list), "The rewards should be a list."
        assert len(test_rewards) == len(test_completions), (
            "The number of rewards should be the same as the number of completions."
        )
        assert all(isinstance(reward, float) for reward in test_rewards), "All rewards should be floats."

        return True, "", func
    except Exception as e:
        return False, str(e), None
