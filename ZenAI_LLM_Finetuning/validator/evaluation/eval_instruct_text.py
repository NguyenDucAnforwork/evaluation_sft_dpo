import argparse
import requests
import asyncio
import os
from pathlib import Path
import math
from loguru import logger
from accelerate.utils import find_executable_batch_size
from axolotl.utils.data import load_tokenized_prepared_datasets
from axolotl.utils.dict import DictDefault
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

from validator.evaluation.models import InstructTextRawTask, InstructTextDatasetType, EvaluationResultText, FileFormat, EvaluationArgs
import validator.evaluation.constants as cst
from validator.evaluation.utils import download_s3_file, process_evaluation_results, check_for_lora, model_is_a_finetune
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

def _load_evaluation_dataset(evaluation_config: DictDefault, tokenizer: AutoTokenizer) -> Dataset:
    prepared_path = Path(evaluation_config.output_dir) / "prepared"
    eval_dataset, _ = load_tokenized_prepared_datasets(tokenizer, evaluation_config, prepared_path)

    original_length = len(eval_dataset)
    eval_dataset = [sample for sample in eval_dataset if any(label != -100 for label in sample["labels"])]
    filtered_length = len(eval_dataset)

    logger.info(f"Filtered out {original_length - filtered_length} samples with empty outputs")
    eval_dataset = sorted(eval_dataset, key=lambda x: len(x["input_ids"]))
    logger.info(f"Loaded evaluation dataset with {filtered_length} samples")
    return eval_dataset


def _collate_evaluation_batch(batch: list[dict[str, list[int]]], tokenizer: AutoTokenizer) -> dict[str, torch.Tensor]:
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def evaluate_instruct_text_model(
    evaluation_config: DictDefault,
    language_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> dict[str, float]:
    evaluation_config.tokenizer_config = tokenizer.name_or_path
    logger.info(f"Config: {evaluation_config}")

    eval_dataset = _load_evaluation_dataset(evaluation_config, tokenizer)

    _log_dataset_and_model_info(eval_dataset, language_model, tokenizer)

    def custom_data_collator(features):
        return _collate_evaluation_batch(features, tokenizer)

    @find_executable_batch_size(starting_batch_size=evaluation_config.starting_batch_size)
    def evaluate_with_batch_size(batch_size):
        training_args = TrainingArguments(
            output_dir=evaluation_config.output_dir,
            per_device_eval_batch_size=batch_size,
            report_to="none",
            bf16=True
        )

        trainer = Trainer(
            model=language_model,
            args=training_args,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            data_collator=custom_data_collator,
            callbacks=[ProgressLoggerCallback(log_interval_seconds=evaluation_config.log_interval_seconds)],
        )

        eval_results = trainer.evaluate()
        return eval_results

    eval_results = evaluate_with_batch_size()
    logger.info(f"Final evaluation results: {eval_results}")
    evaluation_results = {
        "eval_loss": eval_results["eval_loss"],
    }
    return evaluation_results

def evaluate_finetuned_model(
    evaluation_args: EvaluationArgs,
    finetuned_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> dict[str, float]:
    evaluation_config = _load_and_update_evaluation_config(
        evaluation_args=evaluation_args,
        finetuned_model=finetuned_model,
        config_path=cst.VALI_CONFIG_PATH
    )
    return evaluate_instruct_text_model(evaluation_config, finetuned_model, tokenizer)

def evaluate_repo(evaluation_args: EvaluationArgs, synthetic: bool = False) -> None:
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
        if check_for_lora(repo):
            logger.info("LoRA adapter detected. Loading as with Peft")
            base_model = load_model(evaluation_args.original_model, is_base_model=True)
            if "model_params_count" not in results_dict:
                results_dict["model_params_count"] = count_model_parameters(base_model)
            finetuned_model = load_finetuned_model(base_model, repo)
            is_finetune = True
        else:
            logger.info("No LoRA adapter detected. Loading full model")
            finetuned_model = load_model(repo, is_base_model=False)
            if "model_params_count" not in results_dict:
                results_dict["model_params_count"] = count_model_parameters(finetuned_model)
            try:
                is_finetune = model_is_a_finetune(evaluation_args.original_model, finetuned_model)
            except Exception as e:
                logger.info(f"Problem with detection of finetune for {repo}: {e}")
                logger.info("Assuming False")
                is_finetune = False
        log_memory_stats()
        finetuned_model.eval()

        results = evaluate_finetuned_model(
            evaluation_args=evaluation_args,
            finetuned_model=finetuned_model,
            tokenizer=tokenizer,
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

def run_evaluation_text(
    dataset: str,
    file_format: FileFormat,
    original_model: str,
    models: list[str],
    dataset_type: InstructTextDatasetType,
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
            eval_results = evaluate_repo(evaluation_args, synthetic=synthetic)

        return process_evaluation_results(eval_results, is_image=False)

    except Exception as e:
        logger.error(f"Failed to retrieve {task_type} evaluation results: {str(e)}", exc_info=True)
        raise Exception(f"Failed to retrieve {task_type} evaluation results: {str(e)}")

async def evaluate_submission(
    task: InstructTextRawTask,
    submission_repos: list[str],
    dataset_type: InstructTextDatasetType
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
    # test_data_filepath = await download_s3_file(task.test_data, save_path="/workspace/input_data")
    # Old code, failed to download, link expired
    
    # new code, replace path with local test data
    test_data_filepath = "/workspace/input_data/test_data.json"

    test_results = run_evaluation_text(
        dataset=test_data_filepath,
        **evaluation_params
    )

    # try:
    #     os.remove(test_data_filepath)
    # except Exception as e:
    #     logger.warning(f"Failed to remove test data file {test_data_filepath}: {e}")

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
        # synthetic_data_filepath = await download_s3_file(task.synthetic_data, save_path="/workspace/input_data")
        synthetic_data_filepath = "/workspace/input_data/synthetic_data.json"

        synth_results = run_evaluation_text(
            dataset=synthetic_data_filepath,
            models=top_4_repos,
            **{k: v for k, v in evaluation_params.items() if k != "models"},
            synthetic=True
        )

        # try:
        #     os.remove(synthetic_data_filepath)
        # except Exception as e:
        #     logger.warning(f"Failed to remove synthetic data file {synthetic_data_filepath}: {e}")

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

    task = InstructTextRawTask(
        task_id=task_id,
        model_id=task_info["model_id"],
        test_data=task_info["test_data"],
        training_data=task_info["training_data"],
        field_system=task_info["field_system"],
        field_input=task_info["field_input"],
        format=task_info["format"],
        no_input_format=task_info["no_input_format"],
        system_format=task_info["system_format"],
        field_instruction=task_info["field_instruction"],
        field_output=task_info["field_output"],
        synthetic_data=task_info["synthetic_data"],
    )

    submissions = [repo for repo in submission_repo.split(",")]
    
    dataset_type = InstructTextDatasetType(
        system_format=task_info["system_format"],
        field_system=task_info["field_system"],
        field_instruction=task_info["field_instruction"],
        field_input=task_info["field_input"],
        field_output=task_info["field_output"],
        format=task_info["format"],
        no_input_format=task_info["no_input_format"]
    )

    await evaluate_submission(task, submissions, dataset_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=str, required=True)
    parser.add_argument("--submission_repo", type=str, required=True)
    args = parser.parse_args()

    asyncio.run(main(args.task_id,
                     args.submission_repo))
