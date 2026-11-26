import argparse
import hashlib
import json
import re
import subprocess
import unicodedata
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_smoke_test import generate

ROOT = Path(__file__).resolve().parent
PROMPT = "请回答下面的中文四选一题。只输出一行，格式为“最终答案：A”，将 A 替换为你的选项。"
STRICT_PROMPT = "请回答下面的中文四选一题。只输出一行 `最终答案：X`，X 必须是 A、B、C、D 之一，选项后不加文字或标点。"
OPTION_CHECK_PROMPT = STRICT_PROMPT + "\n先确认题干要求选择的是正确项还是错误项，再逐一核对四个选项，选出最符合题意的一项。"
THINK_CAREFULLY_PROMPT = "请仔细思考后再作答。"
SUBJECT_ROLES = {
    "computer_network": "计算机网络工程师",
    "high_school_mathematics": "高中数学教师",
    "high_school_geography": "高中地理教师",
    "chinese_language_and_literature": "中国语言文学教师",
    "accountant": "注册会计师",
}
ANSWER = re.compile(r"最终答案[:：][ \t]*([A-D])")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def key(row):
    return row["subject"], row["split"], row["id"]


def load_examples(path, questions, revision):
    rows = read_jsonl(path)
    subjects = dict.fromkeys(row["subject"] for row in questions)
    grouped = {subject: [] for subject in subjects}
    seen_ids = set()
    seen_texts = {
        "".join(unicodedata.normalize("NFKC", row["question"]).split())
        for row in questions
    }
    for row in rows:
        text = "".join(unicodedata.normalize("NFKC", row["question"]).split())
        if (row["subject"] not in grouped or row["split"] != "dev"
                or row["dataset_revision"] != revision
                or row["answer"] not in ("A", "B", "C", "D")
                or key(row) in seen_ids or text in seen_texts
                or not row["question"].strip()
                or any(not isinstance(row["choices"].get(letter), str)
                       or not row["choices"][letter].strip() for letter in "ABCD")):
            raise ValueError("Examples must be unique complete dev rows outside the evaluation set.")
        seen_ids.add(key(row))
        seen_texts.add(text)
        grouped[row["subject"]].append(row)
    if any(len(examples) != 2 for examples in grouped.values()):
        raise ValueError("Exactly two examples per evaluation subject are required.")
    return grouped


def build_prompt(instruction, row, examples=()):
    parts = [instruction]
    for index, example in enumerate(examples, 1):
        choices = "\n".join(f"{letter}. {example['choices'][letter]}" for letter in "ABCD")
        parts.append(f"示例{index}\n{example['question']}\n{choices}\n最终答案：{example['answer']}")
    choices = "\n".join(f"{letter}. {row['choices'][letter]}" for letter in "ABCD")
    parts.append(f"{row['question']}\n{choices}")
    return "\n\n".join(parts)


def summarize(rows):
    n = len(rows)
    seconds = sum(row["seconds"] for row in rows)
    return {
        "count": n, "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / n,
        "format_valid_rate": sum(row["prediction"] is not None for row in rows) / n,
        "length_limit_rate": sum(row["length_limit"] for row in rows) / n,
        "calls": n, "total_seconds": round(seconds, 4),
        "mean_seconds": round(seconds / n, 4),
        "mean_generated_tokens": sum(row["generated_tokens"] for row in rows) / n,
        "peak_cuda_allocated_mib": max(
            (row["peak_cuda_allocated_mib"] for row in rows
             if "peak_cuda_allocated_mib" in row),
            default=None,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--owner", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    instructions = {
        "baseline": PROMPT,
        "strict-prompt": STRICT_PROMPT,
        "option-check": OPTION_CHECK_PROMPT,
        "few-shot": PROMPT,
    }
    parser.add_argument("--variant", choices=instructions, default="baseline")
    parser.add_argument("--subject-role", action="store_true",
                        help="Add a subject-specific role to strict-prompt or option-check.")
    parser.add_argument("--think-carefully", action="store_true",
                        help="Append the same careful-thinking instruction to every question.")
    args = parser.parse_args()
    if args.subject_role and args.variant not in ("strict-prompt", "option-check"):
        parser.error("--subject-role requires --variant strict-prompt or option-check")
    prompt_instruction = instructions[args.variant]
    if args.think_carefully:
        prompt_instruction += "\n" + THINK_CAREFULLY_PROMPT
    data = ROOT / "data" / "ceval"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    questions = read_jsonl(data / "questions.jsonl")
    answer_rows = read_jsonl(data / "answers.jsonl")
    answers = {key(row): row["answer"] for row in answer_rows}
    if (len(questions) != 200 or len(answer_rows) != 200 or len(answers) != 200
            or list(map(key, questions)) != list(map(key, manifest["items"]))
            or {key(row) for row in questions} != answers.keys()
            or any(answer not in ("A", "B", "C", "D") for answer in answers.values())):
        raise ValueError("Question, manifest and answer IDs must agree.")
    examples_path = data / "few_shot_examples.jsonl"
    examples_by_subject = {}
    if args.variant == "few-shot":
        examples_by_subject = load_examples(examples_path, questions, manifest["revision"])
    role_prefixes = {}
    if args.subject_role:
        subjects = dict.fromkeys(row["subject"] for row in questions)
        missing = subjects.keys() - SUBJECT_ROLES.keys()
        if missing:
            raise ValueError(f"Missing role prefixes for subjects: {', '.join(sorted(missing))}")
        role_prefixes = {subject: f"你是一名{SUBJECT_ROLES[subject]}。" for subject in subjects}
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
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=dtype, local_files_only=True, use_safetensors=True,
    ).to(device).eval()
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
    )
    run_type = f"{args.variant}{'-subject-role' if args.subject_role else ''}"
    if args.think_carefully:
        run_type += "-think-carefully"
    if args.smoke:
        run_type += "-smoke"
    output = ROOT / "runs" / (run_type + "-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True)
    config = {
        "run_id": output.name, "owner": args.owner,
        "variant": args.variant,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("evaluate_ceval.py", "qwen_smoke_test.py")},
        "commit": git.stdout.strip() if git.returncode == 0 else None,
        "dataset": manifest["name"], "dataset_revision": manifest["revision"],
        "seed": manifest["seed"], "model": model_dir.name,
        "model_revision": (model_dir / "revision.txt").read_text().strip(),
        "dataset_sha256": {name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                           for name in ("manifest.json", "questions.jsonl", "answers.jsonl")},
        "torch": str(torch.__version__), "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
        "device_type": device, "dtype": str(dtype).removeprefix("torch."),
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "prompt": prompt_instruction, "answer_regex": ANSWER.pattern, "do_sample": False,
        "max_new_tokens": 64, "timing": "Tokenization through decoding; no warm-up.",
    }
    if role_prefixes:
        config["subject_role_prefixes"] = role_prefixes
    if args.variant == "few-shot":
        config["few_shot"] = {
            "file": str(examples_path.relative_to(ROOT)),
            "sha256": hashlib.sha256(examples_path.read_bytes()).hexdigest(),
            "examples_per_question": 2,
            "selection": "First two complete dev rows by original ID, excluding normalized duplicate question text.",
            "order": {subject: [dict(zip(("subject", "split", "id"), key(row)))
                                for row in examples]
                      for subject, examples in examples_by_subject.items()},
        }
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    results = []
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as log:
        for row in questions:
            subject = row["subject"]
            instruction = prompt_instruction
            if role_prefixes:
                instruction = f"{role_prefixes[subject]}\n{instruction}"
            prompt = build_prompt(instruction, row, examples_by_subject.get(subject, ()))
            messages = [{"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            raw, metrics = generate(model, tokenizer, device, prompt)
            match = ANSWER.fullmatch(raw.strip())
            prediction = match[1] if match else None
            reference = answers[key(row)]
            length_limit = metrics["generated_tokens"] == 64
            result = {
                "subject": row["subject"], "split": row["split"], "id": row["id"],
                "input": rendered, "output": raw, "prediction": prediction,
                "reference": reference, "correct": prediction == reference,
                "length_limit": length_limit,
                "status": [flag for flag, active in (
                    ("parse_failed", prediction is None), ("length_limit", length_limit),
                ) if active],
                **metrics,
            }
            log.write(json.dumps(result, ensure_ascii=False) + "\n")
            log.flush()
            results.append(result)
            if len(results) % 20 == 0:
                print(f"{len(results)}/{len(questions)}", flush=True)
    summary = {
        "overall": summarize(results),
        "by_subject": {subject: summarize([r for r in results if r["subject"] == subject])
                       for subject in dict.fromkeys(r["subject"] for r in results)},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output)
    print(json.dumps(summary["overall"]))


if __name__ == "__main__":
    main()
