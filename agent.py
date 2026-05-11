import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()
db = SQLDatabase.from_uri("sqlite:///nfl.db")
llm = ChatOpenAI(model="gpt-4o", temperature=0)


class AgentState(TypedDict):
    question: str #The question asked by the user
    sql: str
    result: str
    answer: str
    error: str

def generate_sql(state: AgentState):
    schema = db.get_table_info()
    prompt = f"""Here is the schema of a database containing NFL data. Schema: {schema}
    
    - Player names in play_by_play are abbreviated (e.g. "K.Murray") — never filter by name directly
    - Always join play_by_play to weekly_rosters on the relevant ID column (passer_player_id, receiver_player_id, or rusher_player_id) matched to player_id in weekly_rosters to get full player names

    Write a SQLite query to answer: {state["question"]}. Return ONLY the raw SQL needed, no markdown, no explanations."""
    response = llm.invoke(prompt)
    return {"sql": response.content.strip()}

def execute_sql(state: AgentState):
    try:
        result = db.run(state["sql"])
        return {"result": result, "error": ""}
    except Exception as E:
        return {"result": "", "error": str(E)}

def generate_fixed_sql(state: AgentState):
    schema = db.get_table_info()
    prompt = f"""This SQL query failed. Please fix it.
    
    Schema: {schema}
    Question: {state["question"]}
    Failed SQL: {state["sql"]}
    Error: {state["error"]}
    
    Return ONLY the raw SQL needed, no markdown, no explanations."""
    response = llm.invoke(prompt)
    return {"sql": response.content.strip()}

def should_fix_or_answer(state: AgentState):
    if state.get("error") != "":
        return "fix_sql"
    return "done"

research_graph = StateGraph(AgentState)
research_graph.add_node("generate_sql", generate_sql)
research_graph.add_node("execute_sql", execute_sql)
research_graph.add_node("generate_fixed_sql", generate_fixed_sql)

research_graph.set_entry_point("general_sql")
research_graph.add_edge("generate_sql", "execute_sql")
research_graph.add_conditional_edges("execute_sql", should_fix_or_answer, {"fix_sql": "fix_sql", "done": END})
research_graph.add_edge("fix_sql", "execute_sql")

research_agent = research_graph.compile()

def write_response(state: AgentState):
    prompt = f"""Please write a response to this question: {state["question"]} using the information provided: {state["result"]}. Only include the response."""
    response = llm.invoke(prompt)
    return {"answer": response.content.strip()}

writing_graph = StateGraph(AgentState)
writing_graph.add_node("write_response", write_response)

writing_graph.set_entry_point("write_response")
writing_graph.add_edge("write_response", END)

writing_agent = writing_graph.compile()

supervisor_agent = StateGraph(AgentState)
supervisor_agent.add_node("research_agent", research_agent)
supervisor_agent.add_node("writing_agent", writing_agent)

supervisor_agent.set_entry_point("research_agent")
supervisor_agent.add_edge("research_agent", "writing_agent")
supervisor_agent.add_edge("writing_agent", END)

app = supervisor_agent.compile()