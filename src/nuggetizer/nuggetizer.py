''' Implementation of nuggetizer (referenced from https://github.com/castorini/nuggetizer) '''

from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from .embedder import EmbeddingModel
from .handler import LLMHandler
from .prompts import (
    create_nugget_prompt,
    merge_nugget_prompt,
    score_nugget_prompt,
    assign_nugget_prompt,
)
from src.data import (
    Query,
    Segment,
    Nugget,
    ScoredNugget,
    AssignedScoredNugget
)
from dataclasses import asdict
from typing import Any
from tqdm import tqdm
import re
import ast
import umap
import json
import hdbscan
import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', category=FutureWarning)


class Nuggetizer:
    
    def __init__(self,
                 model: str=None,
                 creator_model: str=None,
                 scorer_model: str=None,
                 assigner_model: str=None,
                 embedding_model: str=None,
                 scorer_window_size: int=10,
                 assigner_window_size: int=10,
                 log_level: int=0, # 0: NO_LOG, 1: DEBUG/INFO, 2: WARNING/ERROR
                 **llm_kwargs):
        if model is not None:
            creator_model = model
            scorer_model = model
            assigner_model = model
        
        self.creator_llm = LLMHandler(creator_model, **llm_kwargs) if creator_model else None
        self.scorer_llm = LLMHandler(scorer_model, **llm_kwargs) if scorer_model else None
        self.assigner_llm = LLMHandler(assigner_model, **llm_kwargs) if assigner_model else None

        self.embedding_model = EmbeddingModel(embedding_model) if embedding_model else None

        self.scorer_window_size = scorer_window_size
        self.assigner_window_size = assigner_window_size

        self.log_level = log_level

    
    def create(self, query: Query, segments: list[Segment]) -> dict[str, list]:
        nuggets: dict[str, list] = {} # docid : list of nuggets
        docid2segments = defaultdict(list)
        for segment in segments:
            docid = segment.docid if segment.docid else segment.segid.split('#')[0]
            docid2segments[docid].append(segment)
        
        for idx, (docid, segments) in tqdm(enumerate(docid2segments.items()), desc=f'Creating nuggets', total=len(docid2segments)):
            # According to the segmentation strategy, since there are duplicates between two adjacent segments, 
            # `creator_max_nuggets` is set to at most half the number of segments.
            messages = create_nugget_prompt(query, segments, int(len(segments) / 2) if len(segments) > 1 else 1)
            
            temperature = 0.0
            trial_count = 50
            while trial_count > 0:
                raw_response = None
                try:
                    raw_response = self.creator_llm.run(messages, temperature=temperature)
                    response = raw_response.replace("```python", "").replace("```", "").strip()
                    nugget_texts = ast.literal_eval(response) # convert str to list of str safely
                    nuggets[docid] = nugget_texts
                    break
                except Exception as e:
                    print(f'Error: {str(e)} ---- Topic: {query.text}\nSegment: {[segment.text for segment in segments]}\nRaw Response: {raw_response}')
                    temperature = 0.2
                    trial_count -= 1
                    if trial_count == 0:
                        print(f"Failed to parse response after 50 attempts.")
                        nuggets[docid] = []
        
        return nuggets

    
    def split_large_clusters(self, embeddings, labels, max_size=3):
        new_clusters = {}
        next_cluster_id = 0

        for cluster_id in set(labels):
            if cluster_id == -1:
                continue  # skipping outliers

            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_embs = embeddings[cluster_indices]

            if len(cluster_indices) <= max_size:
                new_clusters[next_cluster_id] = cluster_indices
                next_cluster_id += 1
            else:
                n_subclusters = int(np.ceil(len(cluster_indices) / max_size))
                subclusterer = AgglomerativeClustering(
                    n_clusters=n_subclusters,
                    metric='cosine',
                    linkage='average'
                )
                sub_labels = subclusterer.fit_predict(cluster_embs)
                for sub_id in range(n_subclusters):
                    sub_indices = cluster_indices[sub_labels == sub_id]
                    new_clusters[next_cluster_id] = sub_indices
                    next_cluster_id += 1

        # initialized to -1 (all outliers by default)
        final_labels = np.full(len(embeddings), -1)

        for cluster_id, indices in new_clusters.items():
            for idx in indices:
                final_labels[idx] = cluster_id

        return final_labels


    def merge(self, query: Query, nuggets: dict[str, list], plot: bool=False, log: bool=False) -> tuple[list[Nugget], Any]:
        id2docids = defaultdict(set) # id --> docids mapping
        text_list = []
        idx = 0
        for docid, nugget_list in nuggets.items():
            for nugget in nugget_list:
                if nugget in text_list:
                    # lexical deduplication: combine with the existing id
                    existed_idx = next((i for i, text in enumerate(text_list) if text == nugget))
                    id2docids[existed_idx].add(docid)
                else:
                    text_list.append(nugget)
                    id2docids[idx].add(docid)
                    idx += 1
        
        embeddings = self.embedding_model.encode(text_list)
        # at least two nuggets will be similar
        # if none of the most similar ones are similar, then it is an outlier.
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean') # L2
        clusterer.fit(embeddings)
        labels = clusterer.labels_

        if log:
            print(f'Total nuggets after deduplication: {len(set(text_list))}')
            print(f'Total labeled nuggets: {len(labels)}')
            print(f'Total clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}')
            print(f'Total noisy nuggets found: {np.sum(labels == -1)}')
            print(f'Query: {query.text}')
            print(f'Clustered nuggets:')
            for cluster_label in set(labels):
                for i, label in enumerate(labels):
                    if label == cluster_label:
                        print(f'Cluster {label} - {id2docids[i]}: {text_list[i]}')

        results: list[Nugget] = []
        for cluster_label in tqdm(set(labels), desc='Merging nuggets'):

            if cluster_label == -1:
                # outliers do not need to be clustered anymore and are directly added to the results
                for idx, label in enumerate(labels):
                    if label == cluster_label:
                        results.append(Nugget(text=text_list[idx], docids=[docid for docid in id2docids[idx]]))
                continue

            cluster_nuggets = []
            for idx, label in enumerate(labels):
                if label == cluster_label:
                    cluster_nuggets.append({'idx': idx, 'text': text_list[idx]})
            
            if len(cluster_nuggets) < 2:
                # no need to merge
                for item in cluster_nuggets:
                    results.append(Nugget(text=item['text'], docids=[docid for docid in id2docids[item['idx']]]))
                continue
            
            messages = merge_nugget_prompt(query=query, nuggets=[item['text'] for item in cluster_nuggets])
            temperature = 0.0
            trial_count = 50
            while trial_count > 0:
                raw_response = None
                try:
                    raw_response = self.creator_llm.run(messages, temperature=temperature)
                    if '[NO NEED]' in raw_response:
                        for item in cluster_nuggets:
                            results.append(Nugget(text=item['text'], docids=[docid for docid in id2docids[item['idx']]]))
                    else:
                        lines = raw_response.strip().split('\n')
                        for line in lines:
                            matches = re.match(r'^(.*)\s+\[([\d,\s]+)\][\.\s]*$', line)
                            nugget_text = matches.group(1).strip()
                            indices = [int(i.strip()) for i in matches.group(2).split(',')] # local indices
                            results.append(Nugget(text=nugget_text, docids=[docid for index in indices for docid in id2docids[cluster_nuggets[index]['idx']]]))
                    break
                except Exception as e:
                    print(f'Error: {str(e)} Raw Response: {raw_response}')
                    temperature = 0.2
                    trial_count -= 1
                    if trial_count == 0:
                        print(f"Failed to parse response after 50 attempts.")
                        for item in cluster_nuggets:
                            results.append(Nugget(text=item['text'], docids=[docid for docid in id2docids[item['idx']]]))
        
        if log:
            for result in results:
                print(f"{result.text} - {result.docids}")
        
        if plot:
            # Visualize the clustering results by UMAP
            noise_mask = labels == -1
            cluster_mask = ~noise_mask
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
            embedding_umap = reducer.fit_transform(embeddings)
            plt.figure(figsize=(10, 8))
            plt.scatter(embedding_umap[cluster_mask, 0], embedding_umap[cluster_mask, 1], c=labels[cluster_mask], cmap='Spectral', s=5, label='Clustered')
            plt.scatter(embedding_umap[noise_mask, 0], embedding_umap[noise_mask, 1], color='k', s=5, label='Outliers')
            plt.legend()
            plt.show()

        return results, labels
    
    def merge_by_precomputed(self, query: Query, nuggets: dict[str, list], log: bool=False):

        def compute_cosine_distance(embeddings):
            embeddings = np.array(embeddings, dtype=np.float64)
            similarity_matrix = np.dot(embeddings, embeddings.T)
            distance_matrix = 1.0 - similarity_matrix
            return distance_matrix

        text_list = [nugget for docid, nugget_list in nuggets.items() for nugget in nugget_list]
        text_list = list(dict.fromkeys(text_list)) # deduplicate while preserving order
        embeddings = self.embedding_model.encode(text_list)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='precomputed')
        clusterer.fit(compute_cosine_distance(embeddings))
        labels = clusterer.labels_

        if log:
            print(f'Total labeled nuggets: {len(labels)}')
            print(f'Total clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}')
            print(f'Query: {query.text}')
            print(f'Clustered nuggets:')
            for cluster_label in set(labels):
                for i, label in enumerate(labels):
                    if label == cluster_label:
                        print(f'Cluster {label}: {text_list[i]}')
    

    def score(self, query: Query, nuggets: list[Nugget]) -> list[ScoredNugget]:
        scored_nuggets: list[ScoredNugget] = []
        start = 0
        while start < len(nuggets):
            end = min(start + self.scorer_window_size, len(nuggets))
            window_nuggets = nuggets[start: end]
            
            messages = score_nugget_prompt(query, window_nuggets)
            trial_count = 50
            temperature = 0.0
            while trial_count > 0:
                raw_response = None
                try:
                    raw_response = self.scorer_llm.run(messages, temperature=temperature)
                    response = raw_response.replace("```python", "").replace("```json", "").replace("```", "").strip() # json format may occur
                    importance_labels = ast.literal_eval(response) 
                    for nugget, importance in zip(window_nuggets, importance_labels):
                        scored_nuggets.append(
                            ScoredNugget(text=nugget.text, docids=nugget.docids, importance=importance.lower())
                        )
                    break
                except Exception as e:
                    print(f'Failed to parse response: {str(e)} ----- Topic: {query.text}\nRaw Response: {raw_response}')
                    temperature = 0.2
                    trial_count -= 1
                    if trial_count == 0:
                        print(f'Failed to parse response after 50 attempts')
                        scored_nuggets.extend([
                            ScoredNugget(text=nugget.text, docids=nugget.docids, importance='failed')
                            for nugget in window_nuggets
                        ])
            start += self.scorer_window_size
        
        # First sort by importance then position and then take :self.scorer_max_nuggets
        scored_nuggets = sorted(scored_nuggets, key=lambda x: (0 if x.importance == 'vital' else 1, scored_nuggets.index(x)))

        return scored_nuggets


    def assign(self, query: Query, block_text: str, nuggets: list[ScoredNugget], assigner_mode: int = 3) -> list[AssignedScoredNugget]:
        assigned_nuggets: list[AssignedScoredNugget] = []
        start = 0
        while start < len(nuggets):
            end = min(start + self.assigner_window_size, len(nuggets))
            window_nuggets = nuggets[start: end]

            messages = assign_nugget_prompt(query, block_text, window_nuggets, assigner_mode)
            trial_count = 50
            temperature = 0.0
            while trial_count > 0:
                raw_response = None
                try:
                    raw_response = self.assigner_llm.run(messages, temperature=temperature)
                    response = raw_response.replace("```python", "").replace("```json", "").replace("```", "").strip() # json format may occur
                    assignments = ast.literal_eval(response)
                    for nugget, assignment in zip(window_nuggets, assignments):
                        assigned_nuggets.append(
                            AssignedScoredNugget(
                                text=nugget.text,
                                docids=nugget.docids,
                                importance=nugget.importance,
                                assignment=assignment.lower()
                            )
                        )
                    break
                except Exception as e:
                    print(f'Failed to parse response: {str(e)} ----- Topic: {query.text}\nRaw Response: {raw_response}')
                    trial_count -= 1
                    temperature = 0.2
                    if trial_count == 0:
                        print(f'Failed to parse response after 50 attempts')
                        assigned_nuggets.extend([
                            AssignedScoredNugget(text=nugget.text,
                                                 docids=nugget.docids,
                                                 importance=nugget.importance, 
                                                 assignment="failed")
                            for nugget in window_nuggets])
            start += self.assigner_window_size
        
        return assigned_nuggets
    

    def nuggetization(self, input_file: str, output_file: str):
        # create -> merge -> score

        with open(input_file, 'r') as file:
            data = [json.loads(line) for line in file]
        
        nuggets_list = []
        for instance in tqdm(data, desc='Nuggetization Process'):
            meta_data = {} # record meta data

            query = Query(qid=instance['qid'], text=instance['query'])
            segments = [Segment(segid=seg['docid'], text=seg['segment'], docid=seg['docid'].split('#')[0]) for seg in instance['seg_list']]
            meta_data['num_qrels_segments'] = len(segments)
            meta_data['num_qrels_documents'] = len(set(segment.docid for segment in segments))
            
            nuggets = self.create(query, segments)
            meta_data['num_created_nuggets'] = len([text for docid, texts in nuggets.items() for text in texts])
            if meta_data['num_created_nuggets'] == 0: # if no valid nuggets, then remove this topic
                continue
            meta_data['avg_nuggets_per_seg'] = meta_data['num_created_nuggets'] / meta_data['num_qrels_segments']

            nuggets, labels = self.merge(query, nuggets)
            meta_data['num_merged_nuggets'] = len(nuggets)
            meta_data['num_cluster'] = len(set(labels)) - (1 if -1 in labels else 0)
            meta_data['num_outliers'] = int(np.sum(labels == -1))

            nuggets = self.score(query, nuggets)
            meta_data['num_scored_nuggets'] = len(nuggets)
            meta_data['num_vital_nuggets'] = sum([1 for nugget in nuggets if nugget.importance == 'vital'])
            meta_data['num_okay_nuggets'] = sum([1 for nugget in nuggets if nugget.importance == 'okay'])

            nuggets_list.append({
                'qid': instance['qid'],
                'query': instance['query'],
                'nuggets': [asdict(nugget) for nugget in nuggets],
                'meta': meta_data
            })
        
        print(f'Saving the nuggetization results of rag24.test at {output_file}')
        with open(output_file, 'w') as file:
            for item in nuggets_list:
                file.write(json.dumps(item) + '\n')
