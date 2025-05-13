from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.mcp import MCPTools
from agno.team.team import Team
import asyncio
import os
from dotenv import load_dotenv
import nest_asyncio

# Initialize environment and configurations
load_dotenv()
nest_asyncio.apply()
MCP_URL = "https://mcpai.gleeze.com/sse"

app = FastAPI(title="Healthcare Finder API", description="API for processing healthcare-related queries")

class QueryRequest(BaseModel):
    text: str

async def create_healthcare_team():
    async with MCPTools(transport="sse", url=MCP_URL) as mcp_tools:
        healthcare_agent = Agent(
            name="Healthcare Assistant",
            role="You help users find doctors, hospitals, and medicines using zipcode and names, or answer general questions using your knowledge.",
            model=Groq(id="llama-3.1-8b-instant", api_key=os.getenv('GROQ_API_KEY')),
            tools=[mcp_tools],
            show_tool_calls=True,
            markdown=True,
            instructions=[
                "If the user provides only a 5-digit zipcode (e.g., '33601' or 'My zipcode is 33601') without mentioning doctors, hospitals, or medicines, **exclusively** call the `get_county_info` tool with the zipcode and return only the county name. Do not call any other tools or provide additional information.",
                "If the user explicitly requests doctors, hospitals, or medicines (e.g., 'find doctors in 33601' or 'hospitals near 33601'), call the relevant tools (e.g., `get_doctors_by_zipcode`, `get_hospitals`, `get_medicines`).",
                "If the query is unrelated to healthcare (e.g., no mention of doctors, hospitals, medicines, or zipcodes, such as 'Can u tell me about cricket' or 'Who is Elon Musk?'), do not call any tools, create subgroups, or initiate team coordination. Instead, provide a concise, direct answer using your knowledge.",
                "Keep responses minimal and focused. Return only the requested information without team coordination, tool call logs, follow-up prompts, subgroup creation, or speculative content.",
                "For general questions about topics like sports, public figures, or events, use the most recent and accurate information available to you (up to May 13, 2025) and provide a brief, relevant answer.",
                "Do not ask the user for clarification or generate additional questions unless the query explicitly requests it."
            ]
        )

        # General Knowledge Assistant for non-healthcare queries
        general_knowledge_agent = Agent(
            name="General Knowledge Assistant",
            role="You provide concise answers to general questions on topics like science, history, culture, sports, and public figures.",
            model=Groq(id="llama-3.1-8b-instant", api_key=os.getenv('GROQ_API_KEY')),
            tools=[],  # No tools to avoid unnecessary calls
            show_tool_calls=False,
            markdown=True,
            instructions=[
                "Handle queries unrelated to healthcare (e.g., no mention of doctors, hospitals, medicines, or zipcodes, such as 'Who is Elon Musk?', 'Can u tell me about cricket').",
                "Do not call any tools, create subgroups, or initiate team coordination. Provide a concise, direct answer using your knowledge.",
                "Keep responses minimal and focused. Return only the requested information without follow-up prompts, subgroup creation, or speculative content.",
                "Use the most recent and accurate information available (up to May 13, 2025) for topics like public figures, events, or sports.",
                "Do not ask the user for clarification or generate additional questions unless the query explicitly requests it.",
                "For example, for 'Can u tell me about cricket', return a brief overview like: 'Cricket is a bat-and-ball sport played between two teams of eleven players, popular in countries like India, England, and Australia, with formats like Test matches, ODIs, and T20.'"
            ]
        )

        healthcare_team = Team(
            name="Healthcare Finder Team",
            mode="auto",
            model=Groq(id="llama-3.1-8b-instant", api_key=os.getenv('GROQ_API_KEY')),
            members=[healthcare_agent, general_knowledge_agent],
            description="Helps users search doctors, hospitals, and medicines, or answers general questions on various topics.",
            show_members_responses=False
        )
        return healthcare_team

@app.on_event("startup")
async def startup_event():
    global healthcare_team
    healthcare_team = await create_healthcare_team()

@app.post("/process", summary="Process healthcare query", response_description="Returns the response from the healthcare team")
async def process_query(request: QueryRequest):
    try:
        # Validate input text
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Process the query using the healthcare team, mimicking playground behavior
        response = healthcare_team.run(request.text)
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)