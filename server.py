import os
from dotenv import load_dotenv
from fastmcp import FastMCP

from langchain_openai import ChatOpenAI
from smart_query.data_retriever.data_commons_retriever import DataCommonsRetriever
from smart_query.data_retriever.energy_atlas_retriever import EnergyAtlasRetriever
from smart_query.data_retriever.ndpes_retriever import NDPESRetriever
from smart_query.data_retriever.wen_okn_retriever import WENOKNRetriever
from smart_query.data_system.data_system import LLMDataSystem
from smart_query.data_repo.dataframe_annotation import DataFrameAnnotation
from smart_query.utils.logger import get_logger

# Load environment variables                                                                                                                                           
load_dotenv()

mcp = FastMCP("WENOKN MCP Server")

@mcp.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"
