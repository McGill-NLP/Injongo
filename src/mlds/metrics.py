import collections
import re

import pandas as pd


def evaluate(df: pd.DataFrame, gt_col: str, pred_col: str) -> dict:
    """
    Evaluate the performance of the model
    1. accuracy of the intent
    2. f1 score of the intent
    3. accuracy of the slots
    4. f1 score of the slots
    """

    # target: [IN: selected_intention [SL:selected_slot_type slot entity from text] 
    # [SL:selected_slot_type slot entity from text]  ...]]
    # Convert logical form strings to dictionaries
    return sequence_accuracy(
        df[gt_col].str.lower().tolist(), df[pred_col].str.lower().tolist()
    )

    df["target"] = df[gt_col].apply(logical_form_to_dict)
    df["predicted"] = df[pred_col].apply(logical_form_to_dict)


def logical_form_to_dict(logical_form):
    """
    Convert a logical form string to a dictionary.
    [IN: selected_intention [SL:selected_slot_type slot entity from text] [SL:selected_slot_type slot entity from text]  ...]]
    """
    logical_form = logical_form.strip()
    intention = re.search(r"\[IN:(.*?) ", logical_form).group(1)
    slots = {}
    for slot in re.findall(r"\[SL:(.*?) (.*?)\]", logical_form):
        if slot[0] not in slots:
            slots[slot[0]] = []
        slots[slot[0]].append(slot[1])
    return {"intention": intention, "slots": slots}


def f1_score(y_true, y_pred):
    """
    Calculate the F1 score of the intent or slots
    """
    if isinstance(y_true, str):
        y_true = [y_true]
    if isinstance(y_pred, str):
        y_pred = [y_pred]
    tp = len(set(y_true) & set(y_pred))
    fp = len(set(y_pred) - set(y_true))
    fn = len(set(y_true) - set(y_pred))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


import numpy as np


def sequence_accuracy(targets: list[str], predictions: list[str]) -> float:
    """Computes per-sequence accuracy.

    This function is copied from t5.evaluation.metrics
    For each example, returns 1.0 if the target sequence EXACTLY matches the
    predicted sequence. Else, 0.0.

    Args:
      targets: list of strings
      predictions: list of strings

    Returns:
      float. Average sequence-level accuracy.

    Source: https://github.com/google-research/xtreme-up/blob/main/evaluation/metrics.py#L59-L77
    """
    assert len(targets) == len(predictions)
    seq_acc = 100 * np.mean([p == t for p, t in zip(predictions, targets, strict=True)])
    return seq_acc



def span_f1_seqio(targets, predictions):
  """Computes Span based F1 score.

  This function is copied from
  https://github.com/google-research/multilingual-t5/blob/master/multilingual_t5/evaluation/metrics.py

  Args:
    targets: list of strings or list of list of strings if multiple references
      are present.
    predictions: list of strings

  Returns:
    span f1 across all targets and predictions (Based on CoNLL script)
  """
  true_positives = collections.defaultdict(int)
  false_positives = collections.defaultdict(int)
  false_negatives = collections.defaultdict(int)

  def tags_to_spans(tag_sequence, delimiter=" $$ "):
    """Extract spans from IOB1 or BIO tags."""
    tag_sequence_split = [x.strip() for x in tag_sequence.split(delimiter)]
    tags_entities = []
    for tag_entity in tag_sequence_split:
      tag_entity_split = tag_entity.split(":")
      if len(tag_entity_split) != 2:
        continue
      tag = tag_entity_split[0].strip()
      entity = tag_entity_split[1].strip()
      tags_entities.append((tag, entity))
    return tags_entities

  def compute_f1_metrics(true_positives, false_positives, false_negatives):
    precision = float(true_positives) / float(
        true_positives + false_positives + 1e-13
    )
    recall = float(true_positives) / float(
        true_positives + false_negatives + 1e-13
    )
    f1_measure = 2.0 * ((precision * recall) / (precision + recall + 1e-13))
    return precision, recall, f1_measure

  for target, pred in zip(targets, predictions, strict=True):
    gold_spans = tags_to_spans(target)
    predicted_spans = tags_to_spans(pred)

    for span in predicted_spans:
      if span in gold_spans:
        true_positives[span[0]] += 1
        gold_spans.remove(span)
      else:
        false_positives[span[0]] += 1
    # These spans weren't predicted.
    for span in gold_spans:
      false_negatives[span[0]] += 1

  _, _, f1_measure = compute_f1_metrics(
      sum(true_positives.values()),
      sum(false_positives.values()),
      sum(false_negatives.values()),
  )

  return {"span_f1": f1_measure}


def span_f1(targets: list[str], predictions: list[str]) -> float:
  """Computes span F1 score based on mT5/ByT5 output format."""
  return 100 * span_f1_seqio(targets, predictions)["span_f1"]


if __name__ == "__main__":
    df = pd.read_csv("data/output/hau.csv")
    res = sequence_accuracy(df["logical_form"].tolist(), df["logical_form"].tolist())
    print(res)
