from langgraph.graph import StateGraph, START, END
from .schemas import RecState
from typing import Any

class GraphBuilder:
    def __init__(self, agent_provider: Any):
        self.agent_provider = agent_provider

    def build(self) -> StateGraph:
        graph = StateGraph(RecState)

        # Nodes
        graph.add_node('user_understanding_agent', self.agent_provider.user_understanding_agent)
        graph.add_node('initial_retrieval', self.agent_provider.initial_retrieval)
        graph.add_node('nli_agent', self.agent_provider.nli_agent)
        graph.add_node('context_summary_agent', self.agent_provider.context_summary_agent)
        graph.add_node('item_ranker_agent', self.agent_provider.item_ranker_agent)
        
        # Luồng tuần tự (Sequential Flow) giống bản cũ
        graph.add_edge(START, 'user_understanding_agent')
        graph.add_edge('user_understanding_agent', 'initial_retrieval')
        graph.add_edge('initial_retrieval', 'nli_agent')
        
        graph.add_conditional_edges(
            'nli_agent',
            self.agent_provider.should_proceed_to_summary,
            {
                "continue": "context_summary_agent",
                END: END
            }
        )
        graph.add_edge('context_summary_agent', 'item_ranker_agent')
        graph.add_edge('item_ranker_agent', END)

        return graph.compile()