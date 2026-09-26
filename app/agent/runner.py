"""Small CLI wrapper: python -m app.agent.runner "your question".

Useful for quick manual testing without spinning up the full HTTP stack.
"""
import sys

from langchain_core.messages import HumanMessage

from app.agent.graph import build_agent_graph


def main():
    if len(sys.argv) < 2:
        print("usage: python -m app.agent.runner 'your question'")
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    graph = build_agent_graph()
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)], "iteration_count": 0},
        config={"configurable": {"thread_id": "cli"}},
    )
    print("\n=== ANSWER ===\n")
    print(result["messages"][-1].content)
    print(f"\n(iterations: {result['iteration_count']})")


if __name__ == "__main__":
    main()