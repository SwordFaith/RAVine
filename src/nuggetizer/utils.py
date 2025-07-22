''' Utility functions for nuggetization '''

from dotenv import load_dotenv
from tqdm import tqdm
import os
import json
import difflib


def get_openai_api_url_key() -> str | None:
    load_dotenv(dotenv_path=".env")
    openai_api_url = os.getenv('OPEN_API_URL')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    return openai_api_url, openai_api_key


def compute_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def match_nuggets(
    old_nuggets: list[str],
    new_nuggets: list[str],
    similarity_threshold: float = 0.5,
) -> tuple[dict[int, int], set[int], set[int]]:
    """Greedy‑match nuggets between *old* and *new* lists based on highest similarity.

    Returns
    -------
    matches
        Mapping of `old_index -> new_index` for all matched pairs.
    unmatched_old
        Indices of *old_nuggets* that were not matched (⇒ deleted).
    unmatched_new
        Indices of *new_nuggets* that were not matched (⇒ added).
    """
    unmatched_old = set(range(len(old_nuggets)))
    unmatched_new = set(range(len(new_nuggets)))
    matches: dict[int, int] = {}

    # Simple greedy strategy: for each old nugget, pick best available new one
    for old_idx, old_nugget in enumerate(old_nuggets):
        best_score, best_new_idx = 0.0, None
        for new_idx in unmatched_new:
            score = compute_similarity(old_nugget, new_nuggets[new_idx])
            if score > best_score:
                best_score, best_new_idx = score, new_idx
        if best_new_idx is not None and best_score >= similarity_threshold:
            matches[old_idx] = best_new_idx
            unmatched_old.remove(old_idx)
            unmatched_new.remove(best_new_idx)

    return matches, unmatched_old, unmatched_new


def update_nuggets_docids(
    old_nuggets: list[str],
    old_docids: list[set[str]],
    new_nuggets: list[str],
    context_docid: str | None,
    similarity_threshold: float = 0.5,
):
    """Update *docids* mapping after a nuggetization iteration.

    Parameters
    ----------
    old_nuggets / new_nuggets
        Lists of nugget strings *before* and *after* the iteration.
    old_docids
        List of **sets** containing the source *docids* for each *old_nugget* (same length as
        `old_nuggets`).
    context_docid
        Docid of the *context* fed into this iteration - becomes a new source whenever a nugget
        is **modified** or **added**.
    similarity_threshold
        Minimum `SequenceMatcher` ratio to treat two nuggets as the same entity (changed/unchanged).

    Returns
    -------
    updated_docids
        List[Set[str]] aligned with *new_nuggets* where each set contains the up-to-date sources.
    diff_summary
        Dict summarising the change categories:
        {
            "unchanged": List[Tuple[old, new]],
            "modified":  List[Tuple[old, new, similarity]],
            "added":     List[new_nugget],
            "deleted":   List[old_nugget]
        }
    """
    if len(old_nuggets) != len(old_docids):
        raise ValueError("old_nuggets and old_docids must have the same length")

    matches, unmatched_old, unmatched_new = match_nuggets(
        old_nuggets, new_nuggets, similarity_threshold
    )

    updated_docids: list[set[str]] = [set() for _ in new_nuggets] # A list of docids for each new nugget
    diff_summary = {"unchanged": [], "modified": [], "added": [], "deleted": []}

    # Handle matched nuggets
    for old_idx, new_idx in matches.items():
        sim_score = compute_similarity(old_nuggets[old_idx], new_nuggets[new_idx])
        if sim_score == 1.0:
            # Unchanged
            updated_docids[new_idx] = set(old_docids[old_idx]) # The nugget that has not changed indicates the current context is irrelevant to it, and the related docid list is not updated
            diff_summary["unchanged"].append((old_nuggets[old_idx], new_nuggets[new_idx]))
        else:
            # Modified: union old sources + context_docid
            # Meets the threshold conditions
            updated_docids[new_idx] = set(old_docids[old_idx])
            if context_docid is not None:
                updated_docids[new_idx].add(context_docid)
            diff_summary["modified"].append((old_nuggets[old_idx], new_nuggets[new_idx], sim_score))

    # Handle additions
    for new_idx in unmatched_new:
        # If a new nugget is not matched, a new docid list is created for it
        updated_docids[new_idx] = {context_docid} if context_docid else set()
        diff_summary["added"].append(new_nuggets[new_idx])

    # Handle deletions (only recorded in summary)
    for old_idx in unmatched_old:
        # If the old nugget is not matched, the docid list will no longer be set for it, that is, it will be discarded
        diff_summary["deleted"].append(old_nuggets[old_idx])

    return updated_docids, diff_summary # Ensures that the length of `updated_docids` is consistent with the length of the new nugget list.


def process_nugget_trace(
    trace: list[dict],
    similarity_threshold: float = 0.5,
) -> tuple[list[str], list[set[str]]]:
    """Process a full nuggetization trace to compute final nuggets and their docids.

    Parameters
    ----------
    trace : List of dicts with fields:
        - 'docid': docid of the context used for this step
        - 'nuggets': nugget list after this step

    Returns
    -------
    final_nuggets : Final list of nuggets (after all iterations)
    final_docids : List of source docids per nugget
    """
    nuggets: list[str] = []
    docids: list[set[str]] = []

    for step in trace:
        context_docid = step["segid"]
        new_nuggets = step["nuggets"]
        docids, _ = update_nuggets_docids(
            nuggets, docids, new_nuggets, context_docid, similarity_threshold
        )
        nuggets = new_nuggets

    return nuggets, docids


def nuggetization_with_trace(model_name: str, nugget_output_file_path: str, final_nugget_output_file_path: str):
    with open(nugget_output_file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    
    output_list = []
    for instance in tqdm(data):
        # WARNING: maybe there's a gap between `the created nuggets` and `the scored nuggets` due to model capability limitation
        final_nuggets, final_docids = process_nugget_trace(instance['trace'], similarity_threshold=0.6)

        importance_map = {item['text']: item['importance'] for item in instance['nuggets_list']}
        segids_map = {final_nuggets[i]: final_docids[i] for i in range(len(final_nuggets))}

        final_nuggets_list = []
        merged_map = set(importance_map.keys()).union(segids_map.keys())
        for text in merged_map:
            final_nuggets_list.append({
                'text': text,
                'importance': importance_map[text],
                'segids': list(segids_map[text]),
                'docids': list(set(segid.split('#')[0] for segid in segids_map[text]))
            })
        
        output_list.append({
            'qid': instance['qid'],
            'query': instance['query'],
            'nuggets_list': final_nuggets_list,
        })
    
    print(f'Saving the results of rag24.test at {final_nugget_output_file_path}')
    with open(final_nugget_output_file_path, 'w') as file:
        for output_data in output_list:
            file.write(json.dumps(output_data) + '\n')

