import json
import logging
import random
from importlib import resources
from pathlib import Path
from typing import cast

import requests
import tqdm
from fuzzysearch import Match, find_near_matches
from pydantic import TypeAdapter, ValidationError

from . import data_models, prompts

logger = logging.getLogger(__name__)

ta_opinions = TypeAdapter(list[data_models.Opinion])
ta_sentences = TypeAdapter(list[data_models.Sentence])
ta_simple_opinions = TypeAdapter(list[data_models.SimpleOpinion])

NUM_SHOTS = 20
LLM_HOST = "0.0.0.0:5000"
MAX_ATTEMPTS_PER_SENTENCE = 5


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def mock_generate(prompt: str, tokens: int = 200) -> str:
    return "[]"


def generate(prompt: str, tokens: int = 200) -> str:
    request = {"prompt": prompt, "max_new_tokens": tokens}
    response = requests.post(f"http://{LLM_HOST}/api/v1/generate", json=request)

    if response.status_code == 200:
        response_model = data_models.TextGenerationWebUIGenerateResponse(
            **response.json()
        )
        return response_model.results[0].text

    raise ValueError(f"Unexpected response: `{response}`")


prompt_template = resources.read_text(prompts, "few_shot.txt")

monolingual_ref_path = Path("./evaluation/submission/ref/data/monolingual")
monolingual_template_prediction_path = Path(
    "./evaluation/template_submission/res/monolingual"
)
monolingual_prediction_path = Path("./evaluation/submission/res/monolingual")
monolingual_datasets = [
    "norec",
    "multibooked_ca",
    "multibooked_eu",
    "opener_en",
    "opener_es",
    "mpqa",
    "darmstadt_unis",
]


def find_substring(text: str, substr: str | None) -> list[list[str]]:
    if substr is None:
        return [[], []]
    matches = cast(
        list[Match], find_near_matches(substr, text, max_l_dist=1 + len(substr) // 6)
    )
    if len(matches) == 0:
        logger.warning(f"No close matches found for `{substr}` in `{text}`")
        return [[], []]
    match = matches[0]
    return [[match.matched], [f"{match.start}:{match.end}"]]


def sample_example_sents(train_data: list[data_models.Sentence]) -> list[str]:
    random.seed(42)
    example_sents: list[str] = []
    shots_count = 0

    for sent in random.sample(train_data, len(train_data)):
        if shots_count >= NUM_SHOTS:
            break
        if shots_count == 0 and len(sent.opinions) > 0:
            continue
        if shots_count > 0 and len(sent.opinions) == 0:
            continue
        if shots_count > NUM_SHOTS // 2 and len(sent.opinions) <= 1:
            continue
        example_sents.append(f"### Input:\n{sent.text}\n\n")
        simple_opinions = []
        for opinion in sent.opinions:
            simple_opinions.append(
                data_models.SimpleOpinion(
                    Source=opinion.Source[0][0] if len(opinion.Source[0]) > 0 else None,
                    Target=opinion.Target[0][0] if len(opinion.Target[0]) > 0 else None,
                    Polar_expression=opinion.Polar_expression[0][0],
                    Polarity=data_models.Polarity[opinion.Polarity],
                    Intensity=data_models.Intensity[opinion.Intensity]
                    if opinion.Intensity is not None
                    else None,
                )
            )

        example_sents.append("### Response:\n")
        example_sents.append(
            ta_simple_opinions.dump_json(simple_opinions, indent=2).decode("utf-8")
        )
        example_sents.append("\n\n")
        shots_count += 1
    return example_sents


class EmptyStringError(RuntimeError):
    pass


def process_sentence(
    train_data: list[data_models.Sentence],
    input_sent: data_models.Sentence,
    log_prompt: bool = False,
) -> data_models.Sentence:
    example_sents = sample_example_sents(train_data)

    example_sents_str = "".join(example_sents).strip()

    input_sent_str = f"Text: {input_sent.text}"
    prompt = prompt_template.format(
        schema=json.dumps(data_models.SimpleOpinion.model_json_schema(), indent=2),
        examples=example_sents_str,
        input=input_sent_str,
    )
    if log_prompt:
        logger.info("Prompt:\n%s", prompt)
    text = generate(prompt, tokens=512)
    # print(len(prompt) // 4, len(text) // 4, len(prompt) // 4 + len(text) // 4)
    if text == "":
        raise EmptyStringError("Generated empty text")
    try:
        simple_opinions_pred = ta_simple_opinions.validate_json(text)
    except ValidationError as e:
        raise ValueError(f"Generated invalid JSON: `{text}`") from e
        # print(simple_opinions_pred)
    opinions_pred: list[data_models.Opinion] = []
    for simple_opinion_pred in simple_opinions_pred:
        opinion_pred = data_models.Opinion(
            Source=find_substring(input_sent.text, simple_opinion_pred.Source),
            Target=find_substring(input_sent.text, simple_opinion_pred.Target),
            Polar_expression=find_substring(
                input_sent.text, simple_opinion_pred.Polar_expression
            ),
            Polarity=simple_opinion_pred.Polarity,
            Intensity=simple_opinion_pred.Intensity,
        )
        opinions_pred.append(opinion_pred)

    return data_models.Sentence(
        sent_id=input_sent.sent_id,
        text=input_sent.text,
        opinions=opinions_pred,
    )


def process_dataset(dataset_dir_name: str) -> None:
    with open(monolingual_ref_path / dataset_dir_name / "train.json") as f:
        train_data = ta_sentences.validate_json(f.read())

    with open(
        monolingual_template_prediction_path / dataset_dir_name / "predictions.json"
    ) as f:
        test_data = ta_sentences.validate_json(f.read())

    pred_sentences: list[data_models.Sentence] = []
    for i, input_sent in enumerate(tqdm.tqdm(test_data)):
        # immediately remove the gold opinions to guarantee no data leakage
        input_sent.opinions = []
        debug_log_prompt = i == 0
        attempt = 0
        while True:
            try:
                pred_sentence = process_sentence(
                    train_data, input_sent, log_prompt=debug_log_prompt
                )
                break
            except EmptyStringError:
                logger.error("Empty string error, retrying indefinitely...")
            except Exception:
                if attempt >= MAX_ATTEMPTS_PER_SENTENCE:
                    logger.exception(
                        f"Error occured on sentence: `{input_sent}`"
                        ", max attempts exceeded"
                    )
                    pred_sentence = data_models.Sentence(
                        sent_id=input_sent.sent_id,
                        text=input_sent.text,
                        opinions=[],
                    )
                    break
                logger.error(
                    f"Error occured on sentence: `{input_sent}`, retrying {attempt}..."
                )
                attempt += 1
        pred_sentences.append(pred_sentence)
        with open(
            monolingual_prediction_path / dataset_dir_name / "predictions.json", "w"
        ) as f:
            f.write(ta_sentences.dump_json(pred_sentences, indent=2).decode("utf-8"))


def main() -> None:
    setup_logger()
    # process_dataset("mpqa")
    for dataset in monolingual_datasets:
        print(f"Processing dataset: {dataset}")
        process_dataset(dataset)


if __name__ == "__main__":
    main()
