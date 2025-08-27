import os
import time

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from smart_query.data_retriever.data_commons_retriever import DataCommonsRetriever
from smart_query.data_retriever.energy_atlas_retriever import EnergyAtlasRetriever
from smart_query.data_retriever.ndpes_retriever import NDPESRetriever
from smart_query.data_retriever.wen_okn_retriever import WENOKNRetriever
from smart_query.data_system.data_system import LLMDataSystem
from smart_query.utils.logger import get_logger 

logger = get_logger(__name__)

load_dotenv()

OpenAI_KEY = os.getenv("OPENAI_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=5000, api_key=OpenAI_KEY)

logger.debug("Initializing the system ...")

ds = LLMDataSystem(llm)
ds.add_dataframe_retriever(WENOKNRetriever("WEN-OKN Database", llm, join_query_compatible=True))
ds.add_dataframe_retriever(DataCommonsRetriever("Data Commons", llm))
ds.add_dataframe_retriever(EnergyAtlasRetriever("Energy Atlas", llm))
ds.add_text_retriever(NDPESRetriever("NDPES", llm))

logger.debug("Initialization is done")

requests = [
    "Show Ross County in Ohio State.",
    "Show all counties in Kentucky State.",
    "Find all counties the Scioto River flows through.",
    "Find all counties both the Ohio River and the Muskingum River flow through.",        
    "Find all neighboring counties of Guernsey County.",        
    "Find all adjacent states to the state of Ohio.",
    "Show the Ohio River",
    "Find all rivers that flow through Ross County.",
    "What rivers flow through Dane County in Wisconsin?",
    "Find all dams on the Ohio River.",
    "Find all dams in Kentucky State.",
    "Find all counties downstream of Ross County on the Scioto River.",
    "Find all dams located upstream of the power station dpjc6wtthc32 along the Muskingum river",
    "Show all stream gauges on Muskingum River",
    "Show all stream gages in Ross county in Ohio",
    "What stream gages are on the Yahara River in Dana County, WI?",
    "Show the populations for all counties in Ohio State.",
    "Find the median individual income for Ross County and Scioto County.",
    "Find populations for all adjacent states to the state of Ohio.",
    "Find the number of people employed in all counties the Scioto River flows through.",
    "Find all solar power plants in California",
    "Where are the coal-fired power plants in Kentucky",
    "Show natural gas power plants in Florida.",
    "Load all wind power plants with total megawatt capacity greater than 100 in California.",
    "Find all coal mines along the Ohio River",
    "Find the basin Lower Ohio-Salt",
    "Find all watersheds in the Kanawha basin",
    "Find all basins through which the Scioto River flows",
    "Find all watersheds feed into Muskingum River",
    "Find all watersheds in Ross County in Ohio State",
    "Find the watershed with the name Headwaters Black Fork Mohican River",
    "Find all counties downstream of the coal mine with the name 'Bowser Mine' along Ohio River.",
    "Show social vulnerability index of all counties downstream of coal mine with the name 'Bowser Mine' along Ohio River",
    "Find all rivers that flow through the Roanoke basin.",
    "Find all stream gages in the watershed with the name Meigs Creek",
    "Find all stream gages in the watersheds feed into Scioto River",
    "Find all rivers that flow through the watershed with the name Headwaters Auglaize River",
    "How many rivers flow through each county in Ohio?",
    "How many dams are in each county of Ohio?",
    "How many coal mines are in each basin in Ohio?",
    "What is the longest river in each county of Ohio state? ", 
]

for index, request in enumerate(requests):
    start = time.time()

    logger.debug('/'*90)
    logger.debug(f'Request: {request}')
    logger.debug('/'*90)
    
    ds.data_repo.remove_annotations_older_than(2 * 60)
    dfa = ds.process_request(request)
    if index > 1:
        break
    
    end = time.time()
    logger.debug(f"Processing Time: {end-start}")
