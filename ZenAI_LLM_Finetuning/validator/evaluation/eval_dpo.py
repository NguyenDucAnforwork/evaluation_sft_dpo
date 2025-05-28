import argparse
import asyncio
import requests
from loguru import logger
import math
import os
import traceback
import torch
from accelerate.utils import find_executable_batch_size
from axolotl.utils.dict import DictDefault
from datasets import Dataset
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from trl import DPOConfig
from trl import DPOTrainer

import validator.evaluation.constants as cst
from validator.evaluation.models import DpoRawTask, DpoDatasetType, EvaluationResultText, FileFormat, EvaluationArgs
from validator.evaluation.utils import download_s3_file, process_evaluation_results, model_is_a_finetune
from validator.evaluation.common import (
    ProgressLoggerCallback,
    load_results_dict,
    load_tokenizer,
    load_model,
    count_model_parameters,
    load_finetuned_model,
    log_memory_stats,
    save_results_dict,
    _log_dataset_and_model_info,
    _load_and_update_evaluation_config
)

def _adapt_dpo_columns_to_trl(dataset: Dataset, dataset_type: DpoDatasetType) -> Dataset:
    """
    Transform a DPO dataset to match trl's expected column names.

    Args:
        dataset: Hugging Face dataset object
        dataset_type: DpoDatasetType with field mappings
    """
    logger.info("Adapting DPO columns to standard format")

    chosen_field = dataset_type.field_chosen
    rejected_field = dataset_type.field_rejected
    
    if chosen_field in dataset.column_names and rejected_field in dataset.column_names:
        identical_count = 0
        sample_size = min(10, len(dataset))
        sample_indices = list(range(sample_size))
        
        for idx in sample_indices:
            example = dataset[idx]
            chosen = example[chosen_field]
            rejected = example[rejected_field]
            
            if chosen == rejected:
                identical_count += 1
        
        if identical_count > 0:
            logger.warning(f"CRITICAL: Found {identical_count}/{sample_size} samples with identical chosen/rejected, causing random predictions")

            if identical_count > 0:
                example = dataset[sample_indices[0]]
                chosen = example[chosen_field]
                rejected = example[rejected_field]
                logger.warning(f"Example: Chosen/Rejected: '{chosen[:100]}...'")

    column_mapping = {
        dataset_type.field_prompt: cst.TRL_DPO_FIELD_PROMPT,
        dataset_type.field_chosen: cst.TRL_DPO_FIELD_CHOSEN,
        dataset_type.field_rejected: cst.TRL_DPO_FIELD_REJECTED
    }
    for src_col, dst_col in column_mapping.items():
        if src_col in dataset.column_names and src_col != dst_col:
            dataset = dataset.rename_column(src_col, dst_col)

    columns_to_keep = [cst.TRL_DPO_FIELD_PROMPT, cst.TRL_DPO_FIELD_CHOSEN, cst.TRL_DPO_FIELD_REJECTED]
    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
    for col in columns_to_remove:
        dataset = dataset.remove_columns(col)

    return dataset


def _collate_dpo_batch(batch: list[dict[str, list[int]]], tokenizer: AutoTokenizer) -> dict[str, torch.Tensor]:
    logger.debug(f"Collating batch of size {len(batch)}")
    try:
        prompt_ids = [torch.tensor(item["prompt_ids"]) for item in batch]
        prompt_attention_mask = [torch.tensor(item["prompt_attention_mask"]) for item in batch]
        chosen_ids = [torch.tensor(item["chosen_ids"]) for item in batch]
        chosen_attention_mask = [torch.tensor(item["chosen_attention_mask"]) for item in batch]
        rejected_ids = [torch.tensor(item["rejected_ids"]) for item in batch]
        rejected_attention_mask = [torch.tensor(item["rejected_attention_mask"]) for item in batch]

        prompt_ids = pad_sequence(prompt_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        prompt_attention_mask = pad_sequence(prompt_attention_mask, batch_first=True, padding_value=0)
        chosen_ids = pad_sequence(chosen_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        chosen_attention_mask = pad_sequence(chosen_attention_mask, batch_first=True, padding_value=0)
        rejected_ids = pad_sequence(rejected_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        rejected_attention_mask = pad_sequence(rejected_attention_mask, batch_first=True, padding_value=0)

        return {
            "prompt_ids": prompt_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "chosen_ids": chosen_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_ids": rejected_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }
    except Exception as e:
        logger.error(f"Error in collate function: {e}")
        logger.error(traceback.format_exc())
        raise


def evaluate_dpo_model(
    evaluation_config: DictDefault,
    finetuned_model: AutoModelForCausalLM,
    reference_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    evaluation_args: EvaluationArgs
) -> dict[str, float]:
    evaluation_config.tokenizer_config = tokenizer.name_or_path
    logger.info(f"Config: {evaluation_config}")

    dataset_path = evaluation_config.datasets[0]["path"]
    eval_dataset = load_dataset("json", data_files=dataset_path, split="train")
    eval_dataset = _adapt_dpo_columns_to_trl(eval_dataset, evaluation_args.dataset_type)

    _log_dataset_and_model_info(eval_dataset, finetuned_model, tokenizer)

    def custom_data_collator(features):
        logger.debug(f"Collating {len(features)} features")
        return _collate_dpo_batch(features, tokenizer)

    @find_executable_batch_size(starting_batch_size=evaluation_config.starting_batch_size)
    def evaluate_dpo_with_batch_size(batch_size):
        training_args = DPOConfig(
            output_dir=evaluation_config.output_dir,
            per_device_eval_batch_size=batch_size,
            report_to="none",
            bf16=True,
            beta=cst.BETA_DPO,
        )
        dpo_trainer = DPOTrainer(
            model=finetuned_model,
            ref_model=reference_model,
            args=training_args,
            train_dataset=Dataset.from_dict({col: [] for col in eval_dataset.column_names}),
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=[ProgressLoggerCallback(log_interval_seconds=evaluation_config.log_interval_seconds)],
        )

        results = dpo_trainer.evaluate()
        return results

    eval_results = evaluate_dpo_with_batch_size()
    logger.info(f"Final DPO evaluation results: {eval_results}")

    if abs(eval_results["eval_loss"] - 0.6931) < 0.0001:
        logger.error("CRITICAL: Loss value is approximately ln(2) ≈ 0.6931, suggesting models are making random predictions")

    evaluation_results = {
        "eval_loss": eval_results["eval_loss"],
    }
    return evaluation_results


def evaluate_finetuned_dpo_model(
    evaluation_args: EvaluationArgs,
    finetuned_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    reference_model: AutoModelForCausalLM
) -> dict[str, float]:
    evaluation_config = _load_and_update_evaluation_config(
        evaluation_args=evaluation_args,
        finetuned_model=finetuned_model,
        config_path=cst.VALI_CONFIG_PATH
    )
    return evaluate_dpo_model(
        evaluation_config, finetuned_model, reference_model, tokenizer, evaluation_args
    )

def evaluate_dpo_repo(
    evaluation_args: EvaluationArgs,
    synthetic: bool = False
):
    """Evaluate a single model repository and save results directly to file."""
    results_dict = load_results_dict(synthetic=synthetic)
    repo = evaluation_args.repo

    # Skip if duplicate
    if repo in results_dict:
        logger.info(f"Skipping {repo} as it's already evaluated")
        return

    tokenizer = load_tokenizer(evaluation_args.original_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        logger.info(f"Loading reference model: {evaluation_args.original_model}")
        reference_model = load_model(evaluation_args.original_model, is_base_model=True)
        if reference_model is None:
            raise ValueError(f"Reference model {evaluation_args.original_model} failed to load")

        if "model_params_count" not in results_dict:
            results_dict["model_params_count"] = count_model_parameters(reference_model)
        try:
            logger.info(f"Loading finetuned model as LoRA adapter: {repo}")
            finetuned_model = load_finetuned_model(reference_model, repo)
            is_finetune = True
        except Exception as lora_error:
            logger.info(f"Failed to load as LoRA adapter: {lora_error}")
            logger.info(f"Loading finetuned model as full model: {repo}")
            finetuned_model = load_model(repo, is_base_model=False)

            if finetuned_model is None:
                raise ValueError(f"Finetuned model {repo} failed to load as full model")

            try:
                is_finetune = model_is_a_finetune(evaluation_args.original_model, finetuned_model)
            except Exception as e:
                logger.warning(f"Problem with detection of finetune for {repo}: {e}")
                is_finetune = False

        log_memory_stats()
        finetuned_model.eval()
        reference_model.eval()

        results = evaluate_finetuned_dpo_model(
            evaluation_args=evaluation_args,
            finetuned_model=finetuned_model,
            tokenizer=tokenizer,
            reference_model=reference_model,
        )
        results["is_finetune"] = is_finetune
        results_dict[repo] = results
    except Exception as e:
        logger.error(f"Error evaluating {repo}: {e}", exc_info=True)
        results_dict[repo] = str(e)
    finally:
        save_results_dict(results_dict, repo, synthetic=synthetic)
        log_memory_stats()
        return results_dict

def run_evaluation_dpo(
    dataset: str,
    file_format: FileFormat,
    original_model: str,
    models: list[str],
    dataset_type: DpoDatasetType,
    synthetic: bool = False
):
    task_type = type(dataset_type).__name__
    dataset_type_str = dataset_type.model_dump_json()
    dataset_filename = os.path.basename(dataset)

    logger.info(f"Running {task_type} evaluation for models: {models}")

    try:
        for repo in models:
            evaluation_args = EvaluationArgs(
                dataset=f"/workspace/input_data/{dataset_filename}",
                original_model=original_model,
                dataset_type=dataset_type_str,
                file_format=file_format.value,
                repo=repo
            )
            eval_results = evaluate_dpo_repo(evaluation_args, synthetic=synthetic)

        return process_evaluation_results(eval_results, is_image=False)

    except Exception as e:
        logger.error(f"Failed to retrieve {task_type} evaluation results: {str(e)}", exc_info=True)
        raise Exception(f"Failed to retrieve {task_type} evaluation results: {str(e)}")

async def evaluate_submission(
    task: DpoRawTask,
    submission_repos: list[str],
    dataset_type: DpoDatasetType
):
    unique_repos = list(set(submission_repos))
    if len(unique_repos) != len(submission_repos):
        logger.warning(f"Found duplicate repos. Deduplicating {len(submission_repos)} repos to {len(unique_repos)} unique repos")

    results: dict[str, tuple[EvaluationResultText, EvaluationResultText] | Exception] = {}
    repos_to_evaluate = []
    for repo in unique_repos:
        if repo == task.model_id:
            logger.warning(f"Repository {repo} matches original model ID - marking as non-finetuned")
            results[repo] = (
                EvaluationResultText(is_finetune=False, eval_loss=0.0),
                EvaluationResultText(is_finetune=False, eval_loss=0.0),
            )
        else:
            repos_to_evaluate.append(repo)

    if not repos_to_evaluate:
        return results

    assert task.synthetic_data is not None, "Synthetic data shouldn't be none for text tasks"
    assert task.test_data is not None, "Test data shouldn't be none for text tasks"

    evaluation_params = {
        "file_format": FileFormat.JSON,
        "original_model": task.model_id,
        "models": repos_to_evaluate,
        "dataset_type": dataset_type,
    }

    logger.info("Starting test evaluation")
    test_data_filepath = await download_s3_file(task.test_data, save_path="/workspace/input_data")
    
    test_results = run_evaluation_dpo(
        dataset=test_data_filepath,
        **evaluation_params
    )

    try:
        os.remove(test_data_filepath)
    except Exception as e:
        logger.warning(f"Failed to remove test data file {test_data_filepath}: {e}")

    test_eval_results = test_results.results
    task.model_params_count = test_results.base_model_params_count

    test_losses = []
    for repo in repos_to_evaluate:
        if isinstance(test_eval_results.get(repo), Exception):
            results[repo] = test_eval_results[repo]
            continue

        test_result = test_eval_results[repo]
        if not test_result.is_finetune:
            results[repo] = (
                EvaluationResultText(is_finetune=False, eval_loss=0.0),
                EvaluationResultText(is_finetune=False, eval_loss=0.0),
            )
        else:
            test_losses.append((repo, test_result.eval_loss))

    test_losses.sort(key=lambda x: float("inf") if math.isnan(x[1]) else x[1])
    top_4_repos = [repo for repo, _ in test_losses[:4]]

    for repo, _ in test_losses[4:]:
        results[repo] = (
            EvaluationResultText(is_finetune=True, eval_loss=1000.0),
            test_eval_results[repo],
        )

    if top_4_repos:
        logger.info(f"Evaluating synthetic data for top {len(top_4_repos)} models")
        synthetic_data_filepath = await download_s3_file(task.synthetic_data, save_path="/workspace/input_data")

        synth_results = run_evaluation_dpo(
            dataset=synthetic_data_filepath,
            models=top_4_repos,
            **{k: v for k, v in evaluation_params.items() if k != "models"},
            synthetic=True
        )

        try:
            os.remove(synthetic_data_filepath)
        except Exception as e:
            logger.warning(f"Failed to remove synthetic data file {synthetic_data_filepath}: {e}")

        synth_eval_results = synth_results.results

        for repo in top_4_repos:
            if isinstance(synth_eval_results.get(repo), Exception):
                results[repo] = synth_eval_results[repo]
            else:
                results[repo] = (synth_eval_results[repo], test_eval_results[repo])

    for repo in unique_repos:
        if repo not in results:
            results[repo] = Exception("Evaluation failed to complete")

    return results

async def main(task_id: str, submission_repo: str):
    # get task info from task_id
    url = f"https://api.gradients.io/auditing/tasks/{task_id}"
    response = requests.get(url)
    response.raise_for_status()
    task_info = response.json()

    task = DpoRawTask(
        task_id=task_id,
        model_id=task_info["model_id"],
        test_data=task_info["test_data"],
        training_data=task_info["training_data"],
        field_prompt=task_info["field_prompt"],
        field_system=task_info["field_system"],
        field_chosen=task_info["field_chosen"],
        field_rejected=task_info["field_rejected"],
        prompt_format=task_info["prompt_format"],
        chosen_format=task_info["chosen_format"],
        rejected_format=task_info["rejected_format"],
        synthetic_data=task_info["synthetic_data"],
    )

    submissions = [repo for repo in submission_repo.split(",")]

    dataset_type = DpoDatasetType(
        field_prompt=task_info["field_prompt"],
        field_system=task_info["field_system"],
        field_chosen=task_info["field_chosen"],
        field_rejected=task_info["field_rejected"],
        prompt_format=task_info["prompt_format"],
        chosen_format=task_info["chosen_format"],
        rejected_format=task_info["rejected_format"],
    )

    await evaluate_submission(task, submissions, dataset_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=str, required=True)
    parser.add_argument("--submission_repo", type=str, required=True)
    args = parser.parse_args()

    asyncio.run(main(args.task_id,
                     args.submission_repo))