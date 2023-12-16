from model.agent import Agent

agent = Agent()


def model_respond_to_query(query: str) -> str:
    return agent.respond_to_query(query)
