from langgraph.graph import START, StateGraph, END
from typing import TypedDict, Optional, List,Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from LLMs.gemini_llm import client as Gemini
from pydantic import BaseModel
from copy import deepcopy
from google.genai.types import GenerateContentConfig
from prompts.main import SYSTEM_PROMPT


class PlannerOutput(BaseModel):
    destination: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    budget: Optional[int] = None

    clarificationQuestion: Optional[str] = None

    nextAction: list[str]

class SharedState(TypedDict):
    # User Inputs
    destination: Optional[str]
    startDate: Optional[str]
    endDate: Optional[str]
    budget: Optional[int]

    # Agent Outputs
    weather: Optional[str]
    season: Optional[str]
    routes: list[str]

    # Conversation
    userQuery: str
    clarificationQuestion: Optional[str]

    # Planner
    nextAction: list[str]
    
    
def PlannerAgent(state: SharedState) -> SharedState:

    prompt = f"""
Current Shared State

{state}

Latest User Message

{state["userQuery"]}
"""

    response = Gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=PlannerOutput,
        ),
    )

    planner = response.parsed

    # -------- Merge updates --------

    updated_state = deepcopy(state)

    if planner.destination is not None:
        updated_state["destination"] = planner.destination

    if planner.startDate is not None:
        updated_state["startDate"] = planner.startDate

    if planner.endDate is not None:
        updated_state["endDate"] = planner.endDate

    if planner.budget is not None:
        updated_state["budget"] = planner.budget

    updated_state["clarificationQuestion"] = planner.clarificationQuestion
    updated_state["nextAction"] = planner.nextAction

    return updated_state
    
    
state = {
    "destination": None,
    "startDate": None,
    "endDate": None,
    "budget": None,

    "weather": None,
    "season": None,
    "routes": [],

    "userQuery": "Plan a summer trip for Goa.",

    "clarificationQuestion": None,
    "nextAction": []
}


response = PlannerAgent(state=state)

print(response)