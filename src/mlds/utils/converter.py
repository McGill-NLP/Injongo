# ref: https://github.com/HumanSignal/label-studio-sdk/blob/master/src/label_studio_sdk/converter/converter.py
import io
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from operator import itemgetter
from typing import Any, Dict, List

import ijson
from nltk.tokenize.treebank import TreebankWordTokenizer
from tqdm import tqdm


TreebankWordTokenizer.PUNCTUATION = [
    (re.compile(r"([:,])([^\d])"), r" \1 \2"),
    (re.compile(r"([:,])$"), r" \1 "),
    (re.compile(r"\.\.\."), r" ... "),
    (re.compile(r"[;@#$/%&]"), r" \g<0> "),
    (
        re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'),
        r"\1 \2\3 ",
    ),  # Handles the final period.
    (re.compile(r"[?!]"), r" \g<0> "),
    (re.compile(r"([^'])' "), r"\1 ' "),
]


def tokenize(text):
    return [
        text[start:end] for start, end in TreebankWordTokenizer().span_tokenize(text)
    ]


def span_tokenize(text):
    return [(start, end) for start, end in TreebankWordTokenizer().span_tokenize(text)]


def create_tokens_and_tags(text, spans, format):
    if format == "BIO":
        # tokens_and_idx = tokenize(text) # This function doesn't work properly if text contains multiple whitespaces...
        token_index_tuples = [
            token for token in TreebankWordTokenizer().span_tokenize(text)
        ]
        tokens_and_idx = [(text[start:end], start) for start, end in token_index_tuples]
        if spans and all(
            [
                span.get("start") is not None and span.get("end") is not None
                for span in spans
            ]
        ):
            spans = list(sorted(spans, key=itemgetter("start")))
            span = spans.pop(0)
            span_start = span["start"]
            span_end = span["end"] - 1
            prefix = "B-"
            tokens, tags = [], []
            for token, token_start in tokens_and_idx:
                tokens.append(token)
                token_end = (
                    token_start + len(token) - 1
                )  # "- 1" - This substraction is wrong. token already uses the index E.g. "Hello" is 0-4
                token_start_ind = token_start  # It seems like the token start is too early.. for whichever reason

                # if for some reason end of span is missed.. pop the new span (Which is quite probable due to this method)
                # Attention it seems like span['end'] is the index of first char afterwards. In case the whitespace is part of the
                # labell we need to subtract one. Otherwise next token won't trigger the span update.. only the token after next..
                if token_start_ind > span_end:
                    while spans:
                        span = spans.pop(0)
                        span_start = span["start"]
                        span_end = span["end"] - 1
                        prefix = "B-"
                        if token_start <= span_end:
                            break
                # Add tag "O" for spans that:
                # - are empty
                # - span start has passed over token_end
                # - do not have any label (None or empty list)
                if not span or token_end < span_start or not span.get("labels"):
                    tags.append("O")
                elif span_start <= token_end and span_end >= token_start_ind:
                    tags.append(prefix + span["labels"][0])
                    prefix = "I-"
                else:
                    tags.append("O")
        else:
            tokens = [token for token, _ in tokens_and_idx]
            tags = ["O"] * len(tokens)

    elif format == "IO":
        # NO prefix for the tags
        tokens_and_idx = [
            token for token in TreebankWordTokenizer().span_tokenize(text)
        ]
        tokens_and_idx = [(text[start:end], start) for start, end in tokens_and_idx]

        if spans and all(
            [
                span.get("start") is not None and span.get("end") is not None
                for span in spans
            ]
        ):
            spans = list(sorted(spans, key=itemgetter("start")))
            span = spans.pop(0)
            span_start = span["start"]
            span_end = span["end"] - 1
            tokens, tags = [], []
            for token, token_start in tokens_and_idx:
                tokens.append(token)
                token_end = token_start + len(token) - 1

                if token_start > span_end:
                    while spans:
                        span = spans.pop(0)
                        span_start = span["start"]
                        span_end = span["end"] - 1
                        if token_start <= span_end:
                            break
                if not span or token_end < span_start or not span.get("labels"):
                    tags.append("O")
                elif span_start <= token_end and span_end >= token_start:
                    tags.append(span["labels"][0])
                else:
                    tags.append("O")
        else:
            tokens = [token for token, _ in tokens_and_idx]
            tags = ["O"] * len(tokens)
    else:
        raise ValueError("Unknown format: {}".format(format))

    return tokens, tags


def get_json_root_type(filename):
    char = "x"
    with open(filename, "r", encoding="utf-8") as f:
        # Read the file character by character
        while char != "":
            char = f.read(1)

            # Skip any whitespace
            if char.isspace():
                continue

            # If the first non-whitespace character is '{', it's a dict
            if char == "{":
                return "dict"

            # If the first non-whitespace character is '[', it's an array
            if char == "[":
                return "list"

            # If neither, the JSON file is invalid
            return "invalid"

    # If the file is empty, return "empty"
    return "empty"


def get_data(task, outputs, annotation):
    return {
        "id": task["id"],
        "input": task["data"],
        "output": outputs or {},
        "completed_by": annotation.get("completed_by", {}),
        "annotation_id": annotation.get("id"),
        "created_at": annotation.get("created_at"),
        "updated_at": annotation.get("updated_at"),
        "lead_time": annotation.get("lead_time"),
        "history": annotation.get("history"),
        "was_cancelled": annotation.get("was_cancelled"),
    }


def annotation_result_from_task(task):
    has_annotations = "completions" in task or "annotations" in task
    if not has_annotations:
        # logger.warning(
        #     'Each task dict item should contain "annotations" or "completions" [deprecated], '
        #     "where value is list of dicts"
        # )
        return None

    # get last not skipped completion and make result from it
    annotations = task["annotations"] if "annotations" in task else task["completions"]

    # return task with empty annotations
    if not annotations:
        data = get_data(task, {}, {})
        yield data

    # skip cancelled annotations
    cancelled = lambda x: not (x.get("skipped", False) or x.get("was_cancelled", False))
    annotations = list(filter(cancelled, annotations))
    if not annotations:
        return None

    # sort by creation time
    annotations = sorted(
        annotations, key=lambda x: x.get("created_at", 0), reverse=True
    )

    for annotation in annotations:
        result = annotation["result"]
        outputs = defaultdict(list)

        # get results only as output
        for r in result:
            # if "from_name" in r and (
            #     tag_name := self._maybe_matching_tag_from_schema(r["from_name"])
            # ):
            v = deepcopy(r["value"])
            v["type"] = "labels"  # self._schema[tag_name]["type"]
            # if "original_width" in r:
            #     v["original_width"] = r["original_width"]
            # if "original_height" in r:
            #     v["original_height"] = r["original_height"]
            outputs[r["from_name"]].append(v)

        data = get_data(task, outputs, annotation)
        if "agreement" in task:
            data["agreement"] = task["agreement"]
        yield data


def iter_from_json_file(json_file):
    """Extract annotation results from json file

    param json_file: path to task list or dict with annotations
    """

    if isinstance(json_file, str):
        data_type = get_json_root_type(json_file)
        # one task
        if data_type == "dict":
            with open(json_file, "r") as json_file:
                data = json.load(json_file)
            for item in annotation_result_from_task(data):
                yield item

        # many tasks
        elif data_type == "list":
            with io.open(json_file, "rb") as f:
                # logger.debug(f"ijson backend in use: {ijson.backend}")
                data = ijson.items(
                    f, "item", use_float=True
                )  # 'item' means to read array of dicts
                for task in data:
                    for item in annotation_result_from_task(task):
                        if item is not None:
                            yield item
    else:
        data_type = "dict" if isinstance(json_file, dict) else "list"

        # one task
        if data_type == "dict":
            for item in annotation_result_from_task(json_file):
                yield item

        # many tasks
        elif data_type == "list":
            for task in json_file:
                for item in annotation_result_from_task(task):
                    if item is not None:
                        yield item


def convert_to_conll2003(input_data, output_file, format="IO"):
    data_key = "text"
    with io.open(output_file, "w", encoding="utf8") as fout:
        fout.write("-DOCSTART- -X- O\n")
        item_iterator = iter_from_json_file
        for item in tqdm(item_iterator(input_data)):
            # for item in input_data:
            # item = item["annotations"][0]["result"]
            filtered_output = list(
                filter(
                    lambda x: x[0]["type"].lower() == "labels",
                    item["output"].values(),
                )
            )
            tokens, tags = create_tokens_and_tags(
                text=item["input"][data_key],
                spans=next(iter(filtered_output), None),
                format=format,
            )
            for token, tag in zip(tokens, tags):
                fout.write("{token} -X- _ {tag}\n".format(token=token, tag=tag))
            fout.write("\n")


def convert_to_jsonl(input_data: List[Dict[str, Any]], output_file: str, lang: str):
    with open(output_file, "w") as f:
        for item in input_data:
            text = item["data"]["text"]
            spans = []
            for annotation in item["annotations"]:
                for result in annotation["result"]:
                    value = result["value"]
                    spans.append(
                        {
                            "start_byte": value["start"],
                            "limit_byte": value["end"],
                            "label": value["labels"][0],
                        }
                    )
            f.write(
                json.dumps(
                    {
                        "example_id": str(item["inner_id"]),
                        "language": lang,
                        "text": text,
                        "spans": spans,
                        "target": " $$ ".join(
                            [
                                f"{s['label']}: {text[s['start_byte']:s['limit_byte']]}"
                                for s in spans
                            ]
                        ),
                    }
                )
                + "\n"
            )


def simple_majority_vote(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Perform simple majority vote on the json format annotations"""

    all_result = []

    for item in data:
        # Initialize the new item with the original data
        new_item = {
            "id": item["id"],
            "inner_id": item["inner_id"],
            "data": item["data"],
            "total_annotations": item["total_annotations"],
        }
        # Collect all annotations
        all_annotations = []
        for annotation in item["annotations"]:
            for result in annotation["result"]:
                all_annotations.append(
                    {
                        "text": result["value"]["text"],
                        "start": result["value"]["start"],
                        "end": result["value"]["end"],
                        "labels": tuple(
                            result["value"]["labels"]
                        ),  # Convert list to tuple for hashing
                    }
                )

        # Count occurrences of each annotation
        annotation_counts = Counter(tuple(a.items()) for a in all_annotations)

        # Select annotations with majority vote
        majority_threshold = len(item["annotations"]) / 2
        majority_annotations = [
            dict(a)
            for a, count in annotation_counts.items()
            if count > majority_threshold
        ]

        # Convert labels back to list
        for annotation in majority_annotations:
            annotation["labels"] = list(annotation["labels"])

        # Create the new annotation result
        new_annotation = {
            "id": f"majority_vote_{item['id']}",
            "completed_by": {"id": "MAJORITY_VOTE"},
            "result": [
                {
                    "id": f"majority_{i}",
                    "type": "labels",
                    "value": annotation,
                    "origin": "manual",
                    "to_name": "text",
                    "from_name": "label",
                }
                for i, annotation in enumerate(majority_annotations)
            ],
        }

        # Calculate agreement
        total_annotations = sum(annotation_counts.values())
        agreed_annotations = sum(
            count for count in annotation_counts.values() if count > majority_threshold
        )
        agreement = (
            (agreed_annotations / total_annotations) * 100
            if total_annotations > 0
            else 100
        )

        # Add the new annotation and agreement to the item
        new_item["annotations"] = [new_annotation]
        new_item["agreement"] = agreement
        # new_item['changed'] = item["changed"]
        all_result.append(new_item)

    return all_result
