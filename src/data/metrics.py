''' Metric class definition '''

from dataclasses import dataclass, field


@dataclass
class RunMetric:
    is_failed: bool = False
    task_completeness: float = 0.
    citation_recall: float = 0.
    citation_precision: float = 0.
    internal_knowledge_completeness: float = 0. # scoring based on model internal knowledge
    
    latency: float = 0.
    cost: float = 0.
    new_tokens_count: float = 0.
    iteration_turns: float = 0.

    search_params_error: float = 0.
    search_extra_params_error: float = 0.
    fetch_params_error: float = 0.
    fetch_extra_params_error: float = 0.
    fetch_url_error: float = 0.
    tool_choice_error: float = 0.
    tool_name_error: float = 0.

    search_recall: float = 0.
    search_precision: float = 0.
    fetch_precision: float = 0.
    raw_search_recall: float = 0.
    raw_search_precision: float = 0.
    raw_fetch_precision: float = 0.
    search_gain_list: list = field(default_factory=list)
    raw_search_gain_list: list = field(default_factory=list)
    avg_search_gain: float = 0.
    avg_raw_search_gain: float = 0.

    num_search_call: float = 0.
    num_fetch_call: float = 0.
    num_tool_call: float = 0.
    num_correct_search_call: float = 0.
    num_correct_fetch_call: float = 0.

    report: str = None
    blocks: list = field(default_factory=list) # [(block_text. citation_list)]
    block_nuggets_assignment: list = field(default_factory=list) # [[support_label], [support_label], ...]
    global_nuggets_assignment: list = field(default_factory=list) # [support_label]



@dataclass
class ModelMetric:
    num_run: float = 0.
    num_valid_run: float = 0.
    num_failed_runs: float = 0.
    
    ### report quality
    avg_task_completeness: float = 0.
    final_task_completeness: float = 0.
    task_finish_rate: float = 0.
    avg_citation_recall: float = 0.
    avg_citation_precision: float = 0.
    avg_internal_knowledge_completeness: float = 0.

    ### efficiency
    avg_latency: float = 0.
    avg_cost: float = 0.
    avg_new_tokens_count: float = 0.
    avg_iteration_turns: float = 0.

    ### tool performance
    avg_tool_choice_error: float = 0.
    avg_num_search_call: float = 0.
    avg_num_fetch_call: float = 0.
    micro_avg_tool_name_error: float = 0.
    macro_avg_tool_name_error: float = 0.

    # search/fetch call precision
    macro_avg_search_precision: float = 0.
    micro_avg_fetch_precision: float = 0.
    macro_avg_fetch_precision: float = 0.
    macro_avg_raw_search_precision: float = 0.
    micro_avg_raw_fetch_precision: float = 0.
    macro_avg_raw_fetch_precision: float = 0.

    # search call recall
    macro_avg_search_recall: float = 0.
    macro_avg_raw_search_recall: float = 0.

    # search gain
    avg_avg_search_gain: float = 0.
    avg_avg_raw_search_gain: float = 0.

    total_num_search_call: float = 0.
    total_num_fetch_call: float = 0.
    total_num_tool_call: float = 0.
    
    # call correctness
    micro_avg_num_correct_search_call: float = 0.
    macro_avg_num_correct_search_call: float = 0.
    micro_avg_num_correct_fetch_call: float = 0.
    macro_avg_num_correct_fetch_call: float = 0.

    # search params error (wrong params)
    micro_avg_search_params_error: float = 0.
    macro_avg_search_params_error: float = 0.

    # search params error (extra params)
    micro_avg_search_extra_params_error: float = 0.
    macro_avg_search_extra_params_error: float = 0.

    # fetch params error (wrong params)
    micro_avg_fetch_params_error: float = 0.
    macro_avg_fetch_params_error: float = 0.

    # fetch params error (extrac params)
    micro_avg_fetch_extra_params_error: float = 0.
    macro_avg_fetch_extra_params_error: float = 0.
    
    # fetch params error (url error)
    micro_avg_fetch_url_error: float = 0.
    macro_avg_fetch_url_error: float = 0.
