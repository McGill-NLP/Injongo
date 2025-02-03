import collections

import numpy as np
import statsmodels.stats.inter_rater as irr

from mlds.utils.converter import span_tokenize, tokenize, simple_majority_vote
from mlds.data_loader import SLOTS, SLOTS_MERGED, LANGUAGES


def analyze_data(data):
    entity_counts = []
    token_counts = []
    all_tokens = set()
    all_entities = set()
    annotations_list = []

    for entry in data:
        text = entry["data"]["text"]
        annotations = entry["annotations"]
        # spans = span_tokenize(text)
        tokens = tokenize(text)
        token_counts.append(len(tokens))
        all_tokens.update(tokens)

        # Process entities
        entity_count = 0
        # token_labels = ['O'] * len(tokens)  # 'O' for no entity
        for annotation in annotations:
            for result in annotation["result"]:
                # start = result['value']['start']
                # end = result['value']['end']
                label = result["value"]["labels"][0]
                entity_count += 1
                all_entities.add(label)

        entity_counts.append(entity_count)

        # Add this annotation to the list for Fleiss Kappa later
        annotations_list.append((text, annotations))

    # Calculate average number of entities and tokens
    avg_entities_per_text = sum(entity_counts) / len(entity_counts)
    avg_tokens_per_text = sum(token_counts) / len(token_counts)

    unique_entities = len(all_entities)
    unique_tokens = len(all_tokens)

    return {
        "total_entities": sum(entity_counts),
        "avg_entities_per_text": avg_entities_per_text,
        "total_tokens": sum(token_counts),
        "avg_tokens_per_text": avg_tokens_per_text,
        "unique_entities": unique_entities,
        "unique_tokens": unique_tokens,
    }


# Fleiss' Kappa calculation
def calculate_fleiss_kappa(annotations_list):
    # assert len(annotations_list) > 120, "Not support English language"
    annotators = set()
    for item in annotations_list:
        annotations = item["annotations"]
        for annotation in annotations:
            annotator_id = annotation["completed_by"]["id"]
            annotators.add(annotator_id)

    # Prepare the agreement matrix for Fleiss Kappa
    agreement_matrix = collections.defaultdict(list)

    # Process each annotation and its corresponding labels
    for item in annotations_list:
        text = item["data"]["text"]
        annotations = item["annotations"]
        spans = span_tokenize(text)
        text_length = len(spans)
        annotations = sorted(annotations, key=lambda x: x["updated_at"])

        for annotator_id in annotators:
            token_label_agreement = [-1 for _ in range(text_length)]
            for annotation in annotations:
                if annotator_id == annotation["completed_by"]["id"]:
                    for result in annotation["result"]:
                        label = result["value"]["labels"][0]
                        start = result["value"]["start"]
                        end = result["value"]["end"]
                        token_start_idx = len(span_tokenize(text[:start]))
                        token_end_idx = len(span_tokenize(text[:end]))
                        for i in range(token_start_idx, token_end_idx):
                            token_label_agreement[i] = SLOTS.index(label)

            agreement_matrix[annotator_id].extend(token_label_agreement)

    most_three_annotaters = sorted(
        agreement_matrix.keys(), key=lambda x: agreement_matrix[x].count(-1)
    )[:3]

    agreement_matrix = [agreement_matrix[i] for i in most_three_annotaters]
    agreement_matrix = np.array(agreement_matrix).T
    agreement_matrix = irr.aggregate_raters(agreement_matrix)[0]
    kappa_score = irr.fleiss_kappa(agreement_matrix)

    return kappa_score


# import collections
from typing import Dict, List, Tuple

# import matplotlib
# print(matplotlib.get_cachedir())
import matplotlib.pyplot as plt
import plotly.express as px

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def version_compare(
        unreviewed_data: List[Dict], 
        reviewed_data: List[Dict], 
        majority:bool = False, 
        prefix: str = "",
        merged: bool = False
    ) -> Dict:
    def process_annotations(
        data: List[Dict],
        majority: bool = False,
    ) -> Dict[str, Dict[str, List[Tuple[str, int, int]]]]:
        processed = collections.defaultdict(lambda: collections.defaultdict(list))
        if majority:
            data = simple_majority_vote(data)
        for entry in data:
            text = entry["data"]["text"]
            for annotation in entry["annotations"]:
                annotator_id = annotation["completed_by"]["id"]
                for result in annotation["result"]:
                    label = result["value"]["labels"][0]
                    start, end = result["value"]["start"], result["value"]["end"]
                    entity_text = text[start:end]
                    processed[text][annotator_id].append(
                        (entity_text, start, end, label)
                    )
        return processed

    prefix_clean = prefix.replace("\t", "").replace("[", "").replace("]", "").replace(": ", "")

    unreviewed = process_annotations(unreviewed_data, majority)
    reviewed = process_annotations(reviewed_data, majority)

    changes = collections.defaultdict(lambda: collections.defaultdict(int))
    examples = collections.defaultdict(lambda: collections.defaultdict(list))
    # long_tail_entities = collections.Counter()
    majority_vote_influence = collections.Counter()

    for text in unreviewed:
        if text not in reviewed:
            continue

        unreviewed_annotations = [
            item for sublist in unreviewed[text].values() for item in sublist
        ]
        reviewed_annotations = [
            item for sublist in reviewed[text].values() for item in sublist
        ]

        for entity, start, end, old_label in unreviewed_annotations:
            matching_reviewed = [
                item
                for item in reviewed_annotations
                if item[1] == start and item[2] == end
            ]
            if matching_reviewed:
                new_label = matching_reviewed[0][3]
                if old_label != new_label:
                    changes[old_label][new_label] += 1
                    examples[old_label][new_label].append(entity)
                    majority_vote_influence[new_label] += 1
            else:
                changes[old_label]["Removed"] += 1
                examples[old_label]["Removed"].append(entity)

        for entity, start, end, new_label in reviewed_annotations:
            if not any(
                item[1] == start and item[2] == end for item in unreviewed_annotations
            ):
                changes["Added"][new_label] += 1
                examples["Added"][new_label].append(entity)
                # long_tail_entities[new_label] += 1

    # # Generate insights
    # most_corrected = max(
    #     changes, key=lambda x: sum(changes[x].values()) if x != "Added" else 0
    # )
    # long_tail = min(long_tail_entities, key=long_tail_entities.get)

    # Prepare data for plotting
    entity_changes = {entity: sum(changes[entity].values()) for entity in SLOTS if sum(changes[entity].values()) > 0}

    def counter_entity(data: List[Dict], majority = False):
        if majority:
            data = simple_majority_vote(data)

        labels = []
        for entry in data:
            for annotation in entry["annotations"]:
                # annotator_id = annotation["completed_by"]["id"]
                for result in annotation["result"]:
                    labels.append(result["value"]["labels"][0])
        c = collections.Counter(labels)
        return dict(sorted(c.items(), key=lambda item: item[1], reverse=True))

    unreviewed_ed = counter_entity(unreviewed_data, majority) # entity_distribution
    reviewed_ed = counter_entity(reviewed_data, majority) # entity_distribution
    fig_entity_dist = go.Figure(
        data=[
            go.Bar(x=list(unreviewed_ed.keys()), y=list(unreviewed_ed.values()), name="Unreviewed"),
            go.Bar(x=list(reviewed_ed.keys()), y=list(reviewed_ed.values()), name="Reviewed"),
        ],
        layout=dict(
            width=1500,
            height=500
        )
    )
    fig_entity_dist.update_layout(
        title=dict(
            text=f"{prefix}Entity Type Distribution",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Entity Type",
        yaxis_title="Number of Apperance",
        xaxis_tickangle=-45,
    )
    fig_entity_dist.write_image(f"figures/data/{prefix_clean}_entity_distribution.pdf", scale=5)
    fig_entity_dist.show()

    # Bar chart for entity changes
    fig1 = go.Figure(
        data=[go.Bar(x=list(entity_changes.keys()), y=list(entity_changes.values()))]
    )
    fig1.update_layout(
        title=dict(
            text=f"{prefix}Changes per Entity Type",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Entity Type",
        yaxis_title="Number of Changes",
        xaxis_tickangle=-45,
        width=1500,
        height=500
    )
    fig1.write_image(f"figures/data/{prefix_clean}_entity_changes.pdf", scale=5)
    fig1.show()

    # Pie chart for entity changes
    fig2 = go.Figure(
        data=[
            go.Pie(
                labels=list(entity_changes.keys()), values=list(entity_changes.values())
            )
        ]
    )
    fig2.update_layout(
        title=dict(
            text=f"{prefix}Changes per Entity Type{' with Majority' if majority else ''}",
            x=0.5,
            xanchor='center'
        ),
        width=1500,
        height=500
    )
    fig2.write_image(f"figures/data/{prefix_clean}_entity_changes{'_majority' if majority else ''}.pdf", scale=5)
    fig2.show()

    # Bar chart for sorted entity changes pairs
    pairs = {
        f"{type_a} => {type_b}": v
        for type_a, type_b_dict in changes.items()
        for type_b, v in type_b_dict.items()
    }
    pairs = dict(sorted(pairs.items(), key=lambda item: item[1], reverse=True))

    fig3 = go.Figure(data=[go.Bar(x=list(pairs.keys()), y=list(pairs.values()))])
    fig3.update_layout(
        title=dict(
            text=f"{prefix}Distribution of Changes{' with Majority' if majority else ''}",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Entity Type",
        yaxis_title="Number of Changes",
        xaxis_tickangle=-45,
        xaxis_tickfont=dict(size=8),
        width=1500,
        height=500
    )
    fig3.write_image(f"figures/data/{prefix_clean}_entity_changes_pairs{'_majority' if majority else ''}.pdf", scale=5)
    fig3.show()

    # Sankey diagram FOR entity changes
    source = []
    target = []
    value = []
    if merged:
        labels = (
            ["Removed_FIN", "Added_FIN"]
            + [f"{s}_FIN" for s in SLOTS_MERGED]
            + ["Removed_FOUT", "Added_FOUT"]
            + [f"{s}_FOUT" for s in SLOTS_MERGED]
        )
    else:
        labels = (
            ["Removed_FIN", "Added_FIN"]
            + [f"{s}_FIN" for s in SLOTS]
            + ["Removed_FOUT", "Added_FOUT"]
            + [f"{s}_FOUT" for s in SLOTS]
        )
    # apply colors
    node_color_scale = px.colors.qualitative.Plotly
    link_color_scale = px.colors.qualitative.Plotly

    node_colors = [
        node_color_scale[i * (len(node_color_scale) - 1) // len(labels)]
        for i in range(len(labels)//2)
    ] * 2
    node_color_mapper = {
        label: color for label, color in zip(labels, node_colors)
    }

    link_colors = []
    for type_a, type_b_dict in changes.items():
        for type_b, v in type_b_dict.items():
            # if v < 2:
            #     continue
            source.append(type_a + "_FIN")
            target.append(type_b + "_FOUT")
            value.append(v)
            link_colors.append(node_color_mapper[type_a + "_FIN"])

    source_indices = [labels.index(s) for s in source]
    target_indices = [labels.index(t) for t in target]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=node_colors,
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=value,
                    color=link_colors,
                ),
            )
        ]
    )
    fig.update_layout(
        title=dict(
            text=f"{prefix}Sankey Diagram of Entity Type Changes" + (" with Majority" if majority else ""),
            x=0.5,
            xanchor='center'
        ), 
        font_size=8,
        width=1500,
        height=500
    )
    fig.write_image(f"figures/data/{prefix_clean}_entity_changes_sankey{'_majority' if majority else ''}.pdf", scale=5)
    fig.show()

    return {
        # "most_corrected_entity": most_corrected,
        # "long_tail_entity": long_tail,
        "changes": dict(changes),
        "examples": dict(examples),
        "majority_vote_influence": dict(majority_vote_influence),
    }


if __name__ == "__main__":
    from mlds import data_loader

    entity_manager = data_loader.EntityDataManager(data_folder="data/json")

    cols_used = [
        "total_entities",
        "avg_entities_per_text",
        "total_tokens",
        "avg_tokens_per_text",
        "unique_entities",
        "unique_tokens",
    ]
    cols = [
        "Language",
        "Total Entities",
        "Avg Entities per Text",
        "Total Tokens",
        "Avg Tokens per Text",
        "Unique Entities",
        "Unique Tokens",
        "Reviewed",
        "Unreviewed Fleiss Kappa",
        "Reviewed Fleiss Kappa",
        "delta",
    ]
    print("\t".join(cols))

    for lan in LANGUAGES:

        language = entity_manager.load_data(lan, reviewed=False)
        analysis = analyze_data(language)
        unreviewed_kappa = calculate_fleiss_kappa(language)

        language = entity_manager.load_data(lan, reviewed=True)
        review_kappa = calculate_fleiss_kappa(language)

        print(
            "\t".join(
                map(
                    str,
                    [lan]
                    + [round(analysis[col], 3) for col in cols_used]
                    + [
                        "True",
                        round(unreviewed_kappa, 4),
                        round(review_kappa, 4),
                        round(review_kappa - unreviewed_kappa, 4),
                    ],
                )
            )
        )


    # English data
    lan = "eng"
    language = entity_manager.load_english_data(reviewed=False)
    language = [item for sublist in language.values() for item in sublist]
    analysis = analyze_data(language)
    kappa = calculate_fleiss_kappa(language)

    print(
        "\t".join(
            map(
                str,
                [lan]
                + [round(analysis[col], 3) for col in cols_used]
                + [
                    "True",
                    round(kappa, 4),
                    round(kappa, 4),
                    round(0, 4),
                ],
            )
        )
    )

    # Merged Entity Data
    SLOTS = SLOTS_MERGED
    print("\n\nMerged Entity Data Analysis\n")

    for lan in LANGUAGES:

        language = entity_manager.load_data(lan, reviewed=False, merged=True)
        analysis = analyze_data(language)
        unreviewed_kappa = calculate_fleiss_kappa(language)

        language = entity_manager.load_data(lan, reviewed=True, merged=True)
        review_kappa = calculate_fleiss_kappa(language)

        print(
            "\t".join(
                map(
                    str,
                    [lan]
                    + [round(analysis[col], 3) for col in cols_used]
                    + [
                        "True",
                        round(unreviewed_kappa, 4),
                        round(review_kappa, 4),
                        round(review_kappa - unreviewed_kappa, 4),
                    ],
                )
            )
        )

    lan = "eng"
    language = entity_manager.load_english_data(merged=True)
    language = [item for sublist in language.values() for item in sublist]
    analysis = analyze_data(language)
    kappa = calculate_fleiss_kappa(language)

    print(
        "\t".join(
            map(
                str,
                [lan]
                + [round(analysis[col], 3) for col in cols_used]
                + [
                    "True",
                    round(kappa, 4),
                    round(kappa, 4),
                    round(0, 4),
                ],
            )
        )
    )
