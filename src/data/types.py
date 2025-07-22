''' Data class definition (referenced from https://github.com/castorini/nuggetizer) '''

from dataclasses import dataclass

@dataclass
class Query:
    qid: str
    text: str

@dataclass
class Segment:
    segid: str
    text: str
    docid: str | None

@dataclass
class Nugget:
    ''' Attributable Nugget '''
    text: str
    docids: list[str]

@dataclass
class ScoredNugget(Nugget):
    importance: str  # choices: "vital", "okay", "failed"

@dataclass
class AssignedScoredNugget(ScoredNugget):
    assignment: str # choices: "support", "not_support", "partial_support"
