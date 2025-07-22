''' Function definition and tool calling function '''

from .client import SearchClient


def search(client: SearchClient, query: str, num_results: int|str=10):
    if isinstance(num_results, str):
        num_results = int(num_results)
    
    results = client.search(query, num_results) # in json format
    prompt = ''
    for i, result in enumerate(results):
        prompt += f'Result #{i + 1}\n'
        prompt += f'Web title: {result["title"]}\n'
        prompt += f'Web URL: {result["url"]}\n'
        prompt += f'Headings: {result["headings"]}\n\n'
    return prompt, results


def fetch(client: SearchClient, url: str):
    prompt = f'Fetched content from {url}:\n'
    try:
        result = client.fetch(url)
        if result:
            prompt += result
        else:
            prompt = f'[Fetch Error]: Wrong URL - {url}'
    except Exception as e:
        prompt = f'[Fetch Error]: {e} - URL: {url}'
    
    return prompt, None


def fetch_v2(client: SearchClient, url: str, max_length: int|str=None):
    if isinstance(max_length, str):
        max_length = int(max_length)
    
    result = client.fetch(url)
    if result:
        prompt = f'Fetched content from {url}:\n'
        if max_length:
            prompt += result[: max_length]
        else:
            prompt += result
    else:
        prompt = f'[Fetch Error]: Wrong URL - {url}'
    return prompt, None


def fetch_v3(client: SearchClient, url: str, start: int|str=None, end: int|str=None):
    if isinstance(start, str):
        start = int(start)
    if isinstance(end, str):
        end = int(end)
    
    result = client.fetch(url)
    if result:
        prompt = f'Fetched content from {url}:\n'
        if start and end:
            prompt += result[start: end]
        elif start:
            prompt += result[start:]
        elif end:
            prompt += result[:end]
        else:
            prompt += result
    else:
        prompt = f'[Fetch Error]: Wrong URL - {url}'
    return prompt, None


search_tool_v2 = {
    'type': 'function',
    'function': {
        'name': 'web_search',
        'description': 'Retrieve a list of documents from the web corpus based on query relevance.',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': ''
                },
                'num_results': {
                    'type': 'number',
                    'description': 'Number of top results to return.',
                },
            },
            'required': [
                'query',
                'num_results',
            ],
            'additionalProperties': False,
        }
    }
}

fetch_tool_v2 = {
    'type': 'function',
    'function': {
        'name': 'web_fetch',
        'description': 'Fetch the content of a web page based on its URL.',
        'parameters': {
            'type': 'object',
            'properties': {
                'url': {
                    'type': 'string',
                    'description': 'The full URL of the web page to fetch content from.',
                },
            },
            'required': [
                'url',
            ],
            'additionalProperties': False,
        }
    }
}




# TODO: future improvement
fetch_tool_v3 = {
    'type': 'function',
    'name': 'web_fetch',
    'description': 'Fetch the content of a web page based on its URL.',
    'parameters': {
        'type': 'object',
        'properties': {
            'url': {
                'type': 'string',
                'description': 'The full URL of the web page to fetch content from.',
            },
            'max_length': {
                'type': 'number',
                'description': 'Maximum number of characters to return from the web page content.',
            },
            'format': {
                'type': 'string',
                'enum': ['text', 'html'],
                'description': 'The format of the returned content: plain text or raw HTML.',
            },
        },
        'required': [
            'url',
        ],
        'additionalProperties': False,
    }
}


search_tool_v0 = {
    'type': 'function',
    'name': 'web_search',
    'description': 'Retrieve a list of documents from the web corpus based on query relevance.',
    'parameters': {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': ''
            },
            'options': {
                'type': 'object',
                'properties': {
                    'num_results': {
                        'type': 'number',
                        'description': 'Number of top results to return.',
                    },
                },
                'required': [
                    'num_results',
                ],
                'additionalProperties': False, # not allow to contain nndeclared fields to detect agent tool call errors
            }
        },
        'required': [
            'query',
            'options',
        ],
        'additionalProperties': False,
    }
}


# `search_tool_v1` referenced from https://platform.openai.com/docs/guides/function-calling?api-mode=responses&example=search-knowledge-base
search_tool_v1 = {
    'type': 'function',
    'name': 'web_search',
    'description': 'Retrieve a list of documents from the web corpus based on query relevance.',
    'parameters': {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': ''
            },
            'num_results': {
                'type': 'number',
                'description': 'Number of top results to return.',
            },
        },
        'required': [
            'query',
            'num_results',
        ],
        'additionalProperties': False,
    }
}


fetch_tool_v0 = {
    'type': 'function',
    'name': 'web_fetch',
    'description': 'Fetch the content of a web page based on its URL.',
    'parameters': {
        'type': 'object',
        'properties': {
            'url': {
                'type': 'string',
                'description': 'The full URL of the web page to fetch content from.',
            },
            'max_length': {
                'type': 'number',
                'description': 'Maximum number of characters to return from the web page content.',
            },
        },
        'required': [
            'url',
        ],
        'additionalProperties': False,
    }
}

fetch_tool_v1 = {
    'type': 'function',
    'function': {
        'name': 'web_fetch',
        'description': 'Fetch the content of a web page based on its URL.',
        'parameters': {
            'type': 'object',
            'properties': {
                'url': {
                    'type': 'string',
                    'description': 'The full URL of the web page to fetch content from.',
                },
                'max_length': {
                    'type': 'number',
                    'description': 'Maximum number of characters to return from the web page content.',
                },
            },
            'required': [
                'url',
            ],
            'additionalProperties': False,
        }
    }
}

