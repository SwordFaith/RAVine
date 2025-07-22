''' Prompts for nuggetization (referenced from https://github.com/castorini/nuggetizer) '''

from src.data import Query, Segment, Nugget, ScoredNugget
import warnings


@DeprecationWarning
def iterative_create_nugget_prompt(query: Query,
                                   segments: list[Segment],
                                   idx: int,
                                   nuggets: list[str],
                                   creator_max_nuggets: int=30):
    warnings.warn("This prompt template has issues and has been deprecated.",
                  category=DeprecationWarning,
                  stacklevel=2)
    prompt = (
        "Update the list of atomic nuggets of information (1-12 words), if needed, so they best provide the information required for the query. " # BUG: 1-12 words may not be a good instruction, as it may lead to too short nuggets.
        "Leverage only the initial list of nuggets (if exists) and the provided context (this is an iterative process). "
        "Return only the final list of all nuggets in a markdown-style python code-block, and place nuggets in pythonic list format (even if no updates). "  # BUG: seems misleading, considering change to markdown format
        "Make sure there is no redundant information. "
        "Ensure the updated nugget list has at most {creator_max_nuggets} nuggets (can be less), keeping only the most vital ones. "
        "Order them in decreasing order of importance. Prefer nuggets that provide more interesting information.\n\n"
        "Search Query: {query}\n"
        "Context:\n"
        "{context}\n"
        "Search Query: {query}\n"
        "Initial Nugget List: {nuggets}\n"
        "Initial Nugget List Length: {num_nuggets}\n"
        "Only update the list of atomic nuggets (if needed, else return as is). "
        "Do not explain. "
        "Always answer in short nuggets (not questions). " # BUG: maybe this confuse the llm
        "List in the form [\"a\", \"b\", ...] and a and b are strings with no mention of \".\n\n"
        "Updated Nugget List:"
    )
    messages = [
        {
            'role': 'model', # gemini: model; gpt-4.1: developer
            'content': 'You are NuggetizeLLM, an intelligent assistant that can update a list of atomic nuggets to best provide all the information required for the query.',
        },
        {
            'role': 'user',
            'content': prompt.format(query=query.text, context=segments[idx], nuggets=nuggets, num_nuggets=len(nuggets), creator_max_nuggets=creator_max_nuggets)
        }
    ]
    return messages


def create_nugget_prompt(query: Query,
                         segments: list[Segment],
                         creator_max_nuggets: int):
    # doc-as-batch
    prompt = (
        "Given a query and a list of possibly relevant context, generate a list of atomic nuggets of information from the context, so they best provide the information required for the query. "
        "Each generated nugget should be a complete and unique statement of a fact from the context (a sentence of about 10 words). "
        "A nugget should include a clear subject, verb, and object, and should avoid using pronouns such a \"it\". "
        "A nugget is not simply a salient statement within the context, but also one that helps answer the query. "
        "Return only the list of nuggets in a markdown-style python code-block, and place nuggets in pythonic list format. "
        "Ensure the nuggets list has at most {creator_max_nuggets} nuggets (can be less or empty). "
        "Return only the most vital nuggets.\n\n"
        "Search Query: {query}\n\n"
        "Context:\n"
        "{context}\n\n"
        "Search Query: {query}\n\n"
        "List in the form [\"a\", \"b\", ...] and a and b are strings with no mention of \". "
        "If no complete statement that is valuable to the query can be found in the context, do not generate low-quality nuggets, and return [] directly. "
        "Do not explain and make sure there is no redundant information.\n\n"
        "Nugget List:"
    )
    context = '\n'.join([segment.text for segment in segments])
    messages = [
        {
            'role': 'model', # gemini: model; gpt-4.1: developer
            'content': 'You are NuggetizeLLM, an intelligent assistant that can generate a list of atomic nuggets to best provide the information required for the query.',
        },
        {
            'role': 'user',
            'content': prompt.format(query=query.text, context=context, creator_max_nuggets=creator_max_nuggets)
        }
    ]
    return messages


def merge_nugget_prompt(query: Query, nuggets: list[str]):
    # merge nuggets in each cluster
    prompt = (
        "Given a query, please merge its list of atomic nuggets (if necessary) by combining similar nuggets, and return a new list of atomic nuggets. "
        "A nugget refers to a semantically complete and unique statement of a fact (a sentence of around 10 words) that helps answer the query.\n\n"
        "Query: {query}\n\n"
        "Nuggets List:\n"
        "{nuggets_list}\n\n"
        "Your output should be: one nugget per line, and for each nugget, indicate which original nuggets were merged (by listing their indices). "
        "Example: nugget_text [1, 2, ...]\n"
        "If there are no similar nuggets in the list, indicating that no merging is needed, simply return: [NO NEED]. "
        "Make sure there is no redundant information."
    )
    nuggets_list = '\n'.join([f'{i}: {nugget}' for i, nugget in enumerate(nuggets)])
    messages = [
        {
            'role': 'model', # gemini: model; gpt-4.1: developer
            'content': 'You are NuggetizeLLM, an intelligent assistant that can merge similar nuggets.',
        },
        {
            'role': 'user',
            'content': prompt.format(query=query.text, nuggets_list=nuggets_list)
        }
    ]
    return messages


def score_nugget_prompt(query: Query, nuggets: list[Nugget]):
    prompt = (
        "Based on the query, label each of the {num_nuggets} nuggets either a vital or okay based on the following criteria. "
        "Vital nuggets represent concepts that must be present in a \"good\" answer; on the other hand, okay nuggets contribute worthwhile information about the target but are not essential. "
        "Return the list of labels in a Pythonic list format (type: List[str]). "
        "The list should be in the same order as the input nuggets. "
        "Make sure to provide a label for each nugget.\n\n"
        "Search Query: {query}\n"
        "Nugget List: {nugget_list}\n\n"
        "Only return the list of labels (List[str]). "
        "Do not explain.\n\n"
        "Labels:"
    )
    messages = [
        {
            'role': 'model', # gemini: model; gpt-4.1: developer
            'content': 'You are NuggetizeScoreLLM, an intelligent assistant that can label a list of atomic nuggets based on their importance for a given search query.',
        },
        {
            'role': 'user',
            'content': prompt.format(num_nuggets=len(nuggets), query=query.text, nugget_list=[nugget.text for nugget in nuggets])
        }
    ]
    return messages


def assign_nugget_prompt(query: Query, block_text: str, nuggets: list[ScoredNugget], assigner_mode: int = 3):
    if assigner_mode == 3:
        instruction = (
            f"Based on the query and passage, label each of the {len(nuggets)} nuggets either as support, partial_support, or not_support using the following criteria."
            "A nugget that is fully captured in the passage should be labeled as support. "
            "A nugget that is partially captured in the passage should be labeled as partial_support. "
            "If the nugget is not captured at all, label it as not_support."
            "Return the list of labels in a Pythonic list format (type: List[str]). "
            "The list should be in the same order as the input nuggets. "
            "Make sure to provide a label for each nugget. "
        )
    else:
        instruction = (
            f"Based on the query and passage, label each of the {len(nuggets)} nuggets either as support or not_support using the following criteria. "
            "A nugget that is fully captured in the passage should be labeled as support; otherwise, label them as not_support. "
            "Return the list of labels in a Pythonic list format (type: List[str]). "
            "The list should be in the same order as the input nuggets. "
            "Make sure to provide a label for each nugget. "
        )
    prompt = (
        "{instruction}\n\n"
        "Search Query: {query}\n"
        "Passage:\n"
        "{context}\n"
        "Nugget List: {nugget_texts}\n\n"
        "Only return the list of labels (List[str]). "
        "Do not explain.\n\n"
        "Labels:"
    )
    messages = [
        {
            'role': 'model', # gemini: model; gpt-4.1: developer
            'content': 'You are NuggetizeAssignerLLM, an intelligent assistant that can label a list of atomic nuggets based on if they are captured by a given passage.',
        },
        {
            'role': 'user',
            'content': prompt.format(instruction=instruction, query=query.text, context=block_text, nugget_texts=[nugget.text for nugget in nuggets])
        }
    ]
    return messages

