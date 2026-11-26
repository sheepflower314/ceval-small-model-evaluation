import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluate_ceval import ANSWER, STRICT_PROMPT, build_prompt, key, read_jsonl
from qwen_smoke_test import generate

SCRIPT = Path(__file__).resolve()
CONFIG_PATH = SCRIPT.with_name("config.json")


def render(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def parsed_answer(raw):
    match = ANSWER.fullmatch(raw.strip())
    return match[1] if match else None


def summarize(rows, stage, calls):
    correct = sum(row[f"{stage}_correct"] for row in rows)
    valid = sum(row[f"{stage}_answer"] is not None for row in rows)
    metric_stages = ("c0", "c1") if stage == "c1" else ("c0",)
    metrics = [row[f"{metric_stage}_metrics"]
               for row in rows for metric_stage in metric_stages]
    seconds = sum(item["seconds"] for item in metrics)
    tokens = sum(item["generated_tokens"] for item in metrics)
    cuda_peaks = [item["peak_cuda_allocated_mib"] for item in metrics
                  if "peak_cuda_allocated_mib" in item]
    return {
        "count": len(rows), "correct": correct, "accuracy": correct / len(rows),
        "format_valid": valid, "format_valid_rate": valid / len(rows),
        "calls": len(rows) * calls, "total_seconds": round(seconds, 4),
        "mean_seconds_per_question": round(seconds / len(rows), 4),
        "mean_generated_tokens_per_call": tokens / (len(rows) * calls),
        "peak_cuda_allocated_mib": max(cuda_peaks, default=None),
    }


def summarize_pair(rows):
    subjects = dict.fromkeys(row["subject"] for row in rows)
    transitions = {
        "wrong_to_correct": sum(not row["c0_correct"] and row["c1_correct"] for row in rows),
        "correct_to_wrong": sum(row["c0_correct"] and not row["c1_correct"] for row in rows),
        "both_correct": sum(row["c0_correct"] and row["c1_correct"] for row in rows),
        "both_wrong": sum(not row["c0_correct"] and not row["c1_correct"] for row in rows),
        "format_became_valid": sum(row["c0_answer"] is None and row["c1_answer"] is not None
                                   for row in rows),
        "format_became_invalid": sum(row["c0_answer"] is not None and row["c1_answer"] is None
                                     for row in rows),
        "answer_changed": sum(row["c0_answer"] != row["c1_answer"] for row in rows),
    }
    return {
        "overall": {
            "C0": summarize(rows, "c0", 1),
            "C1": summarize(rows, "c1", 2),
            "transitions": transitions,
        },
        "by_subject": {
            subject: {
                "C0": summarize([row for row in rows if row["subject"] == subject], "c0", 1),
                "C1": summarize([row for row in rows if row["subject"] == subject], "c1", 2),
            }
            for subject in subjects
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Paired C0/C1 self-check evaluation.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--owner", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data = ROOT / "data" / "ceval"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    questions = read_jsonl(data / "questions.jsonl")
    answer_rows = read_jsonl(data / "answers.jsonl")
    answers = {key(row): row["answer"] for row in answer_rows}
    if (len(questions) != 200 or len(answer_rows) != 200 or len(answers) != 200
            or list(map(key, questions)) != list(map(key, manifest["items"]))
            or set(map(key, questions)) != answers.keys()
            or any(answer not in "ABCD" for answer in answers.values())):
        raise ValueError("Question, manifest and answer IDs must agree.")
    if args.smoke:
        questions = list({row["subject"]: row for row in reversed(questions)}.values())

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch environment.")
    dtype = torch.float16 if device == "cuda" else torch.float32
    torch.manual_seed(manifest["seed"])

    model_dir = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
    load_start = datetime.now()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=dtype, local_files_only=True, use_safetensors=True,
    ).to(device).eval()
    model_load_seconds = (datetime.now() - load_start).total_seconds()

    run_id = "C1" + ("-smoke" if args.smoke else "") + "-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir = ROOT / "runs" / "C_self_check" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    git = __import__("subprocess").run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
    )
    data_hashes = {
        name: hashlib.sha256((data / name).read_bytes()).hexdigest()
        for name in ("manifest.json", "questions.jsonl", "answers.jsonl")
    }
    run_config = {
        "run_id": run_id, "owner": args.owner, "mode": "smoke" if args.smoke else "full",
        "commit": git.stdout.strip() if git.returncode == 0 else None,
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (SCRIPT, CONFIG_PATH, ROOT / "evaluate_ceval.py", ROOT / "qwen_smoke_test.py")
        },
        "dataset": manifest["name"], "dataset_revision": manifest["revision"],
        "dataset_sha256": data_hashes, "seed": manifest["seed"],
        "model": model_dir.name,
        "model_revision": (model_dir / "revision.txt").read_text(encoding="utf-8").strip(),
        "torch": str(torch.__version__), "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
        "device_type": device, "dtype": str(dtype).removeprefix("torch."),
        "model_load_seconds": round(model_load_seconds, 4),
        "prompt": STRICT_PROMPT, "self_check_instruction": config["self_check_instruction"],
        "do_sample": config["do_sample"], "max_new_tokens": config["max_new_tokens"],
        "timing": "Tokenization through decoding per generate call; model loading recorded separately.",
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    rows = []
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("x", encoding="utf-8") as log:
        for index, question in enumerate(questions, 1):
            c0_prompt = build_prompt(STRICT_PROMPT, question)
            c0_input = render(tokenizer, c0_prompt)
            c0_output, c0_metrics = generate(model, tokenizer, device, c0_prompt)
            self_check_prompt = (
                f"{config['self_check_instruction']}\n\n{question['question']}\n"
                + "\n".join(f"{letter}. {question['choices'][letter]}" for letter in "ABCD")
                + f"\n\n你的初答如下\n{c0_output}"
            )
            c1_input = render(tokenizer, self_check_prompt)
            c1_output, c1_metrics = generate(model, tokenizer, device, self_check_prompt)
            c0_answer = parsed_answer(c0_output)
            c1_answer = parsed_answer(c1_output)
            reference = answers[key(question)]
            row = {
                "subject": question["subject"], "split": question["split"], "id": question["id"],
                "reference": reference,
                "c0_input": c0_input, "c0_output": c0_output, "c0_answer": c0_answer,
                "c0_correct": c0_answer == reference, "c0_parse_failed": c0_answer is None,
                "c0_length_limit": c0_metrics["generated_tokens"] == config["max_new_tokens"],
                "c0_metrics": c0_metrics,
                "self_check_prompt": self_check_prompt,
                "c1_input": c1_input, "c1_output": c1_output, "c1_answer": c1_answer,
                "c1_correct": c1_answer == reference, "c1_parse_failed": c1_answer is None,
                "c1_length_limit": c1_metrics["generated_tokens"] == config["max_new_tokens"],
                "c1_metrics": c1_metrics,
                "status": [
                    status for status, failed in (
                        ("c0_parse_failed", c0_answer is None),
                        ("c1_parse_failed", c1_answer is None),
                        ("c0_length_limit", c0_metrics["generated_tokens"] == config["max_new_tokens"]),
                        ("c1_length_limit", c1_metrics["generated_tokens"] == config["max_new_tokens"]),
                    ) if failed
                ],
            }
            rows.append(row)
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            if index % 20 == 0:
                print(f"{index}/{len(questions)}", flush=True)

    summary = summarize_pair(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(output_dir)
    print(json.dumps(summary["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
