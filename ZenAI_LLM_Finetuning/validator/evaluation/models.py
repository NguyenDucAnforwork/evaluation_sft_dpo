import hashlib
from pydantic import BaseModel, ConfigDict, field_validator, model_validator, Field
from enum import Enum
from uuid import UUID
from datetime import datetime
import json

class FileFormat(str, Enum):
    CSV = "csv"  # needs to be local file
    JSON = "json"  # needs to be local file
    HF = "hf"  # Hugging Face dataset
    S3 = "s3"

class TaskType(str, Enum):
    INSTRUCTTEXTTASK = "InstructTextTask"
    IMAGETASK = "ImageTask"
    DPOTASK = "DpoTask"
    GRPOTASK = "GrpoTask"

    def __hash__(self):
        return hash(str(self))

class InstructTextDatasetType(BaseModel):
    system_prompt: str | None = ""
    system_format: str | None = "{system}"
    field_system: str | None = None
    field_instruction: str | None = None
    field_input: str | None = None
    field_output: str | None = None
    format: str | None = None
    no_input_format: str | None = None
    field: str | None = None

class DpoDatasetType(BaseModel):
    field_prompt: str | None = None
    field_system: str | None = None
    field_chosen: str | None = None
    field_rejected: str | None = None
    prompt_format: str | None = "{prompt}"
    chosen_format: str | None = "{chosen}"
    rejected_format: str | None = "{rejected}"

class RewardFunction(BaseModel):
    """Model representing a reward function with its metadata"""
    reward_func: str = Field(
        ...,
        description="String with the python code of the reward function to use",
        examples=[
            "def reward_func_conciseness(completions, **kwargs):",
            "\"\"\"Reward function that favors shorter, more concise answers.\"\"\"",
            "    return [100.0/(len(completion.split()) + 10) for completion in completions]"
        ]
    )
    reward_weight: float = Field(..., ge=0)
    func_hash: str | None = None
    is_generic: bool | None = None

class GrpoDatasetType(BaseModel):
    field_prompt: str | None = None
    reward_functions: list[RewardFunction] | None = []

class RawTask(BaseModel):
    """
    Task data as stored in the base Task table.
    """

    is_organic: bool = False
    task_id: UUID | None = None
    status: str = "pending"
    model_id: str
    ds: str = "hidden"
    account_id: UUID = "00000000-0000-0000-0000-000000000000"
    times_delayed: int = 0
    hours_to_complete: int = 1
    test_data: str | None = None
    training_data: str | None = None
    assigned_miners: list[int] | None = None
    miner_scores: list[float] | None = None
    training_repo_backup: str | None = None
    result_model_name: str | None = None

    created_at: datetime = datetime.now()
    next_delay_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    termination_at: datetime | None = None
    completed_at: datetime | None = None
    n_eval_attempts: int = 0
    task_type: TaskType
    model_params_count: int = 0

    # Turn off protected namespace for model
    model_config = {"protected_namespaces": ()}


class DpoRawTask(RawTask):
    """
    DPO task data as stored in the database. It expand the RawTask with fields from the DpoTask table.
    """
    field_prompt: str
    field_system: str | None = None
    field_chosen: str
    field_rejected: str
    prompt_format: str | None = None
    chosen_format: str | None = None
    rejected_format: str | None = None
    synthetic_data: str | None = None
    file_format: FileFormat = FileFormat.HF
    task_type: TaskType = TaskType.DPOTASK

class GrpoRawTask(RawTask):
    """
    GRPO task data as stored in the database. It expand the RawTask with fields from the GrpoTask table.
    """
    field_prompt: str
    reward_functions: list[RewardFunction]
    file_format: FileFormat = FileFormat.HF
    task_type: TaskType = TaskType.GRPOTASK
    synthetic_data: str | None = None

    @model_validator(mode="after")
    def validate_reward_functions(self) -> "GrpoRawTask":
        for reward_function in self.reward_functions:
            if reward_function.func_hash is None:
                reward_function.func_hash = hashlib.sha256(reward_function.reward_func.encode()).hexdigest()
        return self

class InstructTextRawTask(RawTask):
    """
    Instruct Text task data as stored in the database. It expand the RawTask with fields from the instruct_text_tasks table.
    """

    field_system: str | None = None
    field_instruction: str
    field_input: str | None = None
    field_output: str | None = None
    format: str | None = None
    no_input_format: str | None = None
    system_format: None = None  # NOTE: Needs updating to be optional once we accept it
    synthetic_data: str | None = None
    file_format: FileFormat = FileFormat.HF
    task_type: TaskType = TaskType.INSTRUCTTEXTTASK
    
class EvaluationResultText(BaseModel):
    is_finetune: bool
    eval_loss: float

class DockerEvaluationResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    results: dict[str, EvaluationResultText | Exception]
    base_model_params_count: int = 0

class ImageModelType(str, Enum):
    FLUX = "flux"
    SDXL = "sdxl"

class Img2ImgPayload(BaseModel):
    ckpt_name: str
    lora_name: str
    steps: int
    cfg: float
    denoise: float
    comfy_template: dict
    height: int = 1024
    width: int = 1024
    model_type: str = "sdxl"
    is_safetensors: bool = True
    prompt: str | None = None
    base_image: str | None = None

class EvaluationArgs(BaseModel):
    dataset: str
    original_model: str
    dataset_type: InstructTextDatasetType | DpoDatasetType | GrpoDatasetType
    file_format: FileFormat
    repo: str

    @field_validator("file_format", mode="before")
    def parse_file_format(cls, value):
        if isinstance(value, str):
            return FileFormat(value)
        return value

    @field_validator("dataset_type", mode="before")
    def parse_dataset_type(cls, value):
        if isinstance(value, str):
            try:
                data = json.loads(value)
                if "field_instruction" in data and "field_input" in data:
                    return InstructTextDatasetType.model_validate(data)
                elif "field_chosen" in data:
                    return DpoDatasetType.model_validate(data)
                elif "reward_functions" in data:
                    return GrpoDatasetType.model_validate(data)
            except Exception as e:
                raise ValueError(f"Failed to parse dataset type: {e}")
        return value
    
AnyTextTypeRawTask = InstructTextRawTask | DpoRawTask | GrpoRawTask
AnyTypeRawTask = AnyTextTypeRawTask