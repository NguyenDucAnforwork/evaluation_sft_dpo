# sn56_evaluation

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Instruct Text tasks and DPO tasks evaluation

### Install axolotl

```bash
pip install axolotl==0.9.0
pip install -U hf_xet
```

### Run evaluation

Edit the ``TASK_ID`` and ``SUBMISSION_REPO`` in the [eval_text.sh](eval_text.sh) or [eval_dpo.sh](eval_dpo.sh)files.

Create folder to save dataset
```bash
mkdir /workspace/input_data
```

Eval instruct text:

```bash
bash eval_text.sh
```

Eval DPO:

```bash
bash eval_dpo.sh
```

Remember to remove the file ``/workspace/evaluation_results.json`` and ``/workspace/evaluation_results_synthetic.json`` before running the script again.
