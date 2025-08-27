
import sys
import json
import time
import geopandas as gpd
from datetime import datetime
from typing import List

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate

from smart_query.data_repo.data_repository import DataRepository
from smart_query.data_repo.dataframe_annotation import DataFrameAnnotation
from smart_query.data_retriever.base_retriever import TextRetriever, DataFrameRetriever
from smart_query.utils.df_utils import find_and_remove_consecutive
from smart_query.utils.df_utils import df_to_gdf, create_title_from_request
from smart_query.utils.logger import get_logger 

logger = get_logger(__name__)

class DataLoadingResponse:

    def __init__(self, request: str, dfa: DataFrameAnnotation):
        self.request = request
        self.answer = dfa
        self.respond_at = datetime.now()

    def __init__(self, request: str, text: str):
        self.request = request
        self.answer = text
        self.respond_at = datetime.now()

    def __init__(self, request: str, error: ValueError):
        self.request = request
        self.answer = error
        self.respond_at = datetime.now()


class LLMDataSystem:

    def __init__(self, llm):
        self.llm = llm
        self.data_repo = DataRepository(llm)
        self.dataframe_retrievers = []
        self.text_retrievers = []

    def add_dataframe_retriever(self, dataframe_retriever):
        self.dataframe_retrievers.append(dataframe_retriever)

    def add_text_retriever(self, text_retriever):
        self.text_retrievers.append(text_retriever)

    def get_retriever(self, name: str):
        for dataframe_retriever in self.dataframe_retrievers:
            if dataframe_retriever.name == name:
                return dataframe_retriever
        for text_retriever in self.text_retrievers:
            if text_retriever.name == name:
                return text_retriever

    def atomize(self, request: str):
        self.llm = self.llm
        return [request]

    def route(self, atomic_request: str):
        retriever_description = ""
        for index, df_retriever in enumerate(self.dataframe_retrievers):
            retriever_description = f"""{retriever_description}
{df_retriever.get_description()} 
Use "{df_retriever.get_name()}" if the requested data can be loaded by this retriever.\n"""

        for index, text_retriever in enumerate(self.text_retrievers):
            retriever_description = f"""{retriever_description}
{text_retriever.get_description()} 
Use "{text_retriever.get_name()}" if the answer to the users' request/question can be loaded by this retriever.\n"""

        prompt = PromptTemplate(
            template="""You are an expert at routing a request to different data retrievers. 
The following are the descriptions of all data retrievers and their data sources:
{retriever_descriptions}
If the request is beyond the scope of the above data retrievers, use "Other". For example, use "Other" 
if the request is "Can you give some sample questions?"\n
Return a valid JSON string with the following keys:
    data_source: The appropriate data source for the user's request. must be a Retriever name or "Other"
    explanation: A brief explanation of why this data source was chosen.
    used_data_sources: A list of data sources used as joined conditions.
Don't put any quotes at the top of the returned JSON string.

Request to route: {request}""",
            input_variables=["retriever_descriptions", "request"],
        )

        # prompt_instance = prompt.format(retriever_descriptions=retriever_description, request=atomic_request)
        # print('-' * 70)
        # print(prompt_instance)
        # print('-' * 70)

        request_router = prompt | self.llm | JsonOutputParser()
        json_result = request_router.invoke({"request": atomic_request,
                                             "retriever_descriptions": retriever_description})

        print(json.dumps(json_result, indent=4))
        # return self.get_retriever(json_result['data_source'])
        return json_result

    #----------------------------------------------------
    # make a query plan
    #----------------------------------------------------
    def get_query_plan(self, atomic_request: str):
        retriever_description = ""
        for index, df_retriever in enumerate(self.dataframe_retrievers):
            retriever_description = f"""{retriever_description}
{df_retriever.get_description()} 
Use "{df_retriever.get_name()}" if the requested data can be loaded by this retriever.\n"""

        for index, text_retriever in enumerate(self.text_retrievers):
            retriever_description = f"""{retriever_description}
{text_retriever.get_description()} 
Use "{text_retriever.get_name()}" if the answer to the users' request/question can be loaded by this retriever.\n"""

        prompt = PromptTemplate(
            template="""You are an expert at query planning by using different data retrievers. 
The following are the descriptions of all data retrievers and their data sources:
{retriever_descriptions}

Your task is to transform the user request into multiple requests, ensuring that each request uses only
one particular data source and any data obtained from prior requests. 
    
Please strictly follow the examples to generate the request sequence. The following conditions must be met:

    1. Ensure that each request uses only one data source.
    2. Very important: Two consecutive requests should not use the same data source.
    3. For two consecutive requests, the latter request is always an expansion of the previous request.
    4. Refer to data from earlier requests using the same expressions or identifiers whenever possible.
    5. Don't include references to the previous requests, for example, "the previous request" in a request. 
    6. Don't include any pronouns like "it" or "they".
    7. The last request in the sequence should be semnatically same as the user request.
 
[ Example 1 ] 
Find the population for all counties downstream of the coal power station with the name 'Pike Rock' along Muskingum River.

This request can be transformed into a request list as follows:

1. Find the coal power station with the name 'Pike Rock'.
    Data Source: Energy Atlas
The Result will be stored in the Data Repository.

2. Find all counties downstream of "the coal power station with the name 'Pike Rock'" along Muskingum River.
    Data Source: WEN-OKN Database
Use the data from the Data Repository for the coal power station with the name 'Pike Rock' to find downstream counties.
    
3. Find the population for "all counties downstream of the coal power station with the name 'Pike Rock' along Muskingum River"
    Data Source: Data Commons

You can return the following JSON list without any explanation:
[
   {{ 
      "request": "Find the coal power station with the name 'Pike Rock'",
      "data_source": "Energy Atlas"
   }},
   {{
      "request": "Find all counties downstream of \"the coal power station with the name 'Pike Rock'\" along Muskingum River",
      "data_source": "WEN-OKN Database"
   }},
    {{
      "request": "Find the population for \"all counties downstream of the coal power station with the name 'Pike Rock' along Muskingum River\""",
      "data_source": "Data Commons"
   }}
]
     
Note that each request in the list is independent, so each request can be completely independent of the others. 
Also no two consecutive requests have the same "data_source". The latter request is always a simple enrichment of
the previous request.

[ Example 2 ]
Load flood event counts for all counties Scioto river flows through.

The following is an incorrect returned JSON list:
[
    {{
        "request": "Find all rivers with the name 'Scioto River'",
        "data_source": "WEN-OKN Database"
    }},
    {{
        "request": "Find all counties 'Scioto River' flows through",
        "data_source": "WEN-OKN Database"
    }},
    {{
        "request": "Find flood event counts for all counties 'Scioto River' flows through",
        "data_source": "Data Commons"
    }}
]
Because the first and second requests have the same "data_source" as "WEN-OKN Database".

You can simply return :
[
    {{
        "request": "Find all counties 'Scioto River' flows through",
        "data_source": "WEN-OKN Database"
    }},
    {{
        "request": "Find flood event counts for all counties 'Scioto River' flows through",
        "data_source": "Data Commons"
    }}
]

[ Example 3 ]
Find Ross Ross County in Ohio State.

You can return 
[
   {{ 
       "request": "Find Ross Ross County in Ohio State",
       "data_source": "WEN-OKN Database" 
   }}
]
Because the WEN-OKN is a GraphDB containing counties and states with the Sparal query interface


[ Example 4 ]
Find all counties both the Ohio River and the Muskingum River flow through.

You can return 
[
   {{ 
       "request": "Find all counties both the Ohio River and the Muskingum River flow through",
       "data_source": "WEN-OKN Database" 
   }}
]
Because the WEN-OKN is a GraphDB containing counties and rivers with the Sparal query interface.

[ Example 5 ]
Find all dams located upstream of the power station dpjc6wtthc32 along the Muskingum river.

You can return 
[
   {{ 
       "request": "Find all dams located upstream of the power station dpjc6wtthc32 along the Muskingum river",
       "data_source": "WEN-OKN Database" 
   }}
]
Because the WEN-OKN is a GraphDB containing dams and rivers and power stations with the Sparal query interface.

[ Example 6 ]
Find all stream gages on the Yahara River, which are not in Madison, WI

You can return 
[
   {{ 
       "request": "Find all stream gages on the Yahara River, which are not in Madison, WI",
       "data_source": "WEN-OKN Database" 
   }}
]
Because the WEN-OKN is a GraphDB containing stream gages and rivers and counties and states with the Sparal query interface.

[ Example 7 ]
Show the populations for all counties in Ohio State

You can return                                                                                                                                                          
[                                                                                                                                                                       
   {{                                                                                                                                                                   
       "request": "Show the populations for all counties in Ohio State",   
       "data_source": "Data Commons"                                                                                                                                
   }}                                                                                                                                                                   
]                                                                                                                                                                       
Because the Data Commons API knows how to get the variables for counties or states.

[ Example 8 ]
Find all coal mines along the Ohio River.

You can return
[                                                                                                                                                                       
   {{                                                                                                                                                                   
       "request": "Find the Ohio River",   
       "data_source": "WEN-OKN Database"      
   }},
   {{                                                                                                                                                                   
       "request": "Find all coal mines along the Ohio River",   
       "data_source": "Energy Atlas"                                                                                                                                
   }},                                                                                                                                                                   
]                                                                                                                                                                       
Because the Energy Atlas doesn't know the Ohio River so it must be loaded first.

[ Example 9 ]
Find all basins through which the Scioto River flows.

You can return
[                                                                                                                                                                       
   {{                                                                                                                                                                   
       "request": "Find the Scioto River",   
       "data_source": "WEN-OKN Database"    
   }},
   {{                                                                                                                                                                   
       "request": "Find all basins through which the Scioto River flows",   
       "data_source": "Energy Atlas"                                                                                                                                
   }},                                                                                                                                                                   
]                                                                                                                                                                       
Because the Energy Atlas doesn't know the Scioto River so it must be loaded first.

[ Example 10 ]
Find all coal mines in each basin in Ohio.

You can return
[
    {{
        "request": "Find Ohio State",
        "data_source": "WEN-OKN Database"
    }},
    {{
        "request": "Find all basins in Ohio",
        "data_source": "Energy Atlas"
    }},
    {{
        "request": "Find all coal mines in each basin in Ohio",
        "data_source": "Energy Atlas"
    }}
]
Because the Energy Atlas doesn't know the geometry of the Ohio State so it must be loaded first. 


[ Example 11 ]
What stream gages are on the Yahara River in Dana County, WI?

You can return
[
    {{
        "request": "What stream gages are on the Yahara River in Dana County, Wisconsin",
        "data_source": "WEN-OKN Database"
    }}
]
Because WI is the abbreviation of Wisconsin.

[ Example 12 ]
Find all wind power plants with total megawatt capacity greater than 100 in California.

You can return 
[
   {{
       "request": "Find all wind power plants with total megawatt capacity greater than 100 in California",
       "data_source": "Energy Atlas" 
   }}
]
Please note that the condition "total megawatt capacity greater than 100" must be presevered.

[ Example 13 ]
Retrieve all power stations at risk of flooding in Ohio at 2 PM on July 1, 2025 

You can return
[
   {{
       "request": "Retrieve all power stations at risk of flooding in Ohio at 2 PM on July 1, 2025",
       "data_source": "Energy Atlas"  
   }}
]


Each retriever is smart enough to handle all complex requests in its way. 

Please return with a valid JSON string only without any preamble or explanation.  Don't put any quotes at the top of the returned JSON list.

Here is the user's request. 
    {request}

""",
            input_variables=["retriever_descriptions", "request"],
        )

        # prompt_instance = prompt.format(retriever_descriptions=retriever_description, request=atomic_request)
        # print('-' * 70)
        # print(prompt_instance)

        request_router = prompt | self.llm | JsonOutputParser()
        json_result = request_router.invoke({"request": atomic_request,
                                             "retriever_descriptions": retriever_description})
        # print('-' * 70)
        # print(json.dumps(json_result, indent=4))
        return json_result

    #-------------------------------------------------------
    # check if a given query is an aggregation or not
    #-------------------------------------------------------    
    def check_aggregation_query(self, request: str):
        prompt = PromptTemplate(                                                                                                                                       
            template="""
You are a query expert and need to decide whether the user request is an aggregation query.
An aggregation request may involve 5 core components:
    1) Grouping Objects: Entities to partition data by (e.g., counties, basins).
    2) Summarizing Objects: Entities to aggregate (e.g., rivers, dams).
    3) Association Conditions: Relationships between grouping and summarizing objects (e.g., spatial containment, spatail intersection).
    4) Aggregation Function: Operations like COUNT, SUM, MAX, AVG, or ARGMAX (for object-centric results).
    5) Pre-/Post-Conditions: Filters applied before/after aggregation (e.g., counties in Ohio State, result thresholds).
Please note that an aggregation request must use an aggregation function. It is not an aggregation request if no aggregation function is used.
For example, "find all counties Scioto River flows through" is not an aggregation request because it doesn't use any aggregation function.

For example, the following requests are aggregation queries:
    Find the number of rivers flow through each county in Ohio. (using COUNT)
    Find the number of dams in each county in Ohio. (using COUNT)
    Find the total number of coal mines in each basin. (using COUNT) 
    Find the total power generation capacity of gas power plants in each county in Ohio. (using SUM)
    Find the longest river in county of Ohio. (using MAX)
    Find the county with the the highest number of hospitals in Ohio . (using ARGMAX)
    Find all counties with more than 5 hospitals in Ohio . (using COUNT)
    Find all states where the total coal mine output exceeds 1 million tons. (using SUM)
    Find the river in Ohio that has the highest number of dams.  (using MAX)
    Find the watershed that has the highest total coal mine. (using SUM and MAX)

The following requests are not aggregation queries:
    Find the number of people employed in all counties the Scioto River flows through.  (the number of people employed is a variable in Data Commons)

Here is the user's request:
{request}
 
If the request is not an aggregation request, then return the following JSON string:
{{
    "is_aggregation_query": false
}}

If the request is an aggregation request, then return a JSON string in the following format:
{{
    "is_aggregation_query": true
}}

Don't put any quotes at the top of the returned JSON string.  

""",                     
            input_variables=["retriever_description", "request"],                                      
        )                                   

        # prompt_instance = prompt.format(request=request) 	
        # logger.debug('-' * 70)
        # logger.debug(prompt_instance)

        request_router = prompt | self.llm | JsonOutputParser()  
        result = request_router.invoke({"request": request}) 
        return result


    #-------------------------------------------------------
    # create an aggregation query plan
    #-------------------------------------------------------    
    def get_aggregation_plan(self, request: str):

        retriever_description = ""
        for index, df_retriever in enumerate(self.dataframe_retrievers):
            retriever_description = f"""{retriever_description}
{df_retriever.get_description()} 
Use "{df_retriever.get_name()}" if the requested data can be loaded by this retriever.\n"""

        for index, text_retriever in enumerate(self.text_retrievers):
            retriever_description = f"""{retriever_description}
{text_retriever.get_description()} 
Use "{text_retriever.get_name()}" if the answer to the users' request/question can be loaded by this retriever.\n"""

        prompt = PromptTemplate(
            template="""You are an expert at query planning by using different data retrievers.
 
The following are the descriptions of all data retrievers and their data sources:
{retriever_description}

You are also an expert in query analysis. Extract key components from the given user request, which describes an aggregation query.

Extraction Rules
    - Grouping Object: The entity used for grouping (e.g., county, state).
        * If not explicitly stated, infer the most reasonable entity from the query.
        * If multiple grouping entities exist, choose the most specific one.
    - Summarizing Object: The entity being aggregated (e.g., river, hospital).
        * If not explicitly stated, infer the entity that is being counted, summed, or aggregated.
        * Never use "aggregation" as a placeholder—always extract a meaningful entity.
    - Association Conditions: The relationship between the grouping and summarizing objects.
        * If missing, infer a reasonable relationship (e.g., "river flows through county").
    - Aggregation Function: The mathematical/statistical operation applied (e.g., COUNT, SUM, ARGMAX).
        * Always return in uppercase.
        * If missing, infer the most logical function based on the query.
    - Preconditions: Filters applied before aggregation (e.g., "county is in Ohio").
        * If none exist, return null.
    - Postconditions: Filters applied after aggregation (e.g., "COUNT > 5").
        * If none exist, return null.

Also please create a query plan which first load grouping objects by using preconditions and then load 
summarizing objects with proper bounding box and finally solve the request.

[ Example 1 ]
User Request: "For each county in Ohio, find the number of rivers flowing through the county."

This request can be defined as the following query:
    SELECT county, COUNT(river) AS river_count   
    FROM county, river
    WHERE county in 'Ohio'  
      AND river INTERSECTS county  
    GROUP BY county
    
The object used in "GROUP BY" is the grouping object. The object used to apply the aggregation function COUNT is the summarizing object.

Extraction Output:
{{
  "grouping_object": "county",
  "summarizing_object": "river",
  "association_conditions": "river flows through county",
  "aggregation_function": "COUNT",
  "preconditions": "county in Ohio state",
  "postconditions": null,
  "query_plan": [
      {{ "request": "Find all counties in Ohio state",  "data_source": "WEN-OKN Database"}}
      {{ "request": "Find all rivers", "data_source": "WEN-OKN Database"}}
      {{ "request": "Find the number of rivers flowing through each county in Ohio state",  "data_source": "System"}}
  ]
}}

[ Example 2]
User Request: How many coal mines are in each basin in Ohio?

This request can be defined as the following query:
     SELECT basin, count(coal_mine) as coal_mine_count
       FROM basin, coal_mine, state
      WHERE coal_mine in basin
        AND basin INTERSECT state
        AND state.name = 'Ohio' 

Extraction Output:
{{
  "grouping_object": "basin",
  "summarizing_object": "coal mine",
  "association_conditions": "coal mine in basin",
  "aggregation_function": "COUNT",
  "preconditions": "basin intersects Ohio state",
  "postconditions": null,
  "query_plan": [
      {{ "request": "Find Ohio state",  "data_source": "WEN-OKN Database"}}
      {{ "request": "Find basins in Ohio state", "data_source": "Energy Atlas"}}
      {{ "request": "How many coal mines are in each basin in Ohio",  "data_source": "System"}}
  ]
}}


Strict Guidelines for Extraction
    - Do not return generic placeholders like "aggregation".
    - Ensure that "grouping_object" and "summarizing_object" are never null.
    - If the user request is ambiguous, infer the most logical structure.
    - Only return a JSON object. No explanations, no additional text.
    - For association conditions, construct a meaningful relationship between the grouping and summarizing objects.
    - The field "data_source" must be the name of a data retrivere or "System".

User Request:
{request}

Don't put any quotes at the top of the returned JSON string.  

""",                     
            input_variables=["retriever_description", "request"],                                      
        )                                   

        # prompt_instance = prompt.format(request=request, retriever_description=retriever_description) 	
        # logger.debug('-' * 70)
        # logger.debug(prompt_instance)

        request_router = prompt | self.llm | JsonOutputParser()  
        result = request_router.invoke({"request": request, 
                                        "retriever_description": retriever_description}) 
        return result


    #-------------------------------------------------------
    # generate the code for the final aggregation step
    #-------------------------------------------------------    
    def get_code_for_aggregation_request(self, grouping_adf, summarizing_adf, aggregation_request): 
        prompt = PromptTemplate(                                                                                                                                       
            template="""

Given:
- `grouping_gdf` (GeoDataFrame): 
  {grouping_adf}
  
- `summarizing_object_gdf` 
  {summarizing_adf}

Generate Python code to:
1. Perform an inner spatial join between `grouping_gdf` and `summarizing_object_gdf` using an appropriate spatial predicate as `joined_gdf`.
2. Group the joined data only by the identity columns from `grouping_gdf` as `grouped`.
3. Apply an appropriate aggregation function to count or summarize the features per group.
4. Based on the request, merge the aggregation result to `grouping_gdf`.  
   for example, for each county in Ohio, to find the total number of dams it has, there may be a county without any dams. 
5. Ensure the final result (`df`) contains only the grouping object identities and aggregation result column.
6. Rename the most important identity column from `grouping_gdf` of `df` to 'Name'. 
   Please prioritize the use of columns with the word `Name` as the most important identity column.
7. Do not include additional columns from `grouping_gdf` or `summarizing_object_gdf` in the final output.
8. Do not include any "import" statements in the code.
9. Please note that `grouping_gdf` and `summarizing_object_gdf` may have the same column names, 
   for example, 'OBJECTID'. Don't include the column 'OBJECTID' in the group-by.

[Example]
Suppose `grouping_gdf` contains all counties in Ohio State and `summarizing_object_gdf` contains all rivers 
intersecting with the bounding box of all counties in Ohio State. 

To resolve the request "find the longest river in each county in Ohio",  we need to return three columns:
'countyName', 'riverName' and 'river_length'. The following code can be used:

if "Name" in grouping_gdf.columns.to_list() and "Name" in summarizing_object_gdf.columns.to_list():
    grouping_gdf = grouping_gdf.rename(columns={{'Name': 'countyName'}})
    summarizing_object_gdf = summarizing_object_gdf.rename(columns={{'Name': 'riverName'}})   

# First, ensure input data is in WGS84 (EPSG:4326)
if grouping_gdf.crs is None:
    grouping_gdf.set_crs(epsg=4326, inplace=True)
if summarizing_object_gdf.crs is None:
    summarizing_object_gdf.set_crs(epsg=4326, inplace=True)

# Reproject counties and rivers to a USA-wide projected CRS (units: meters) because we want to show the river length in meters
grouping_gdf = grouping_gdf.to_crs("EPSG:5070")  # Counties
summarizing_object_gdf = summarizing_object_gdf.to_crs("EPSG:5070")  # Rivers

# Spatial join: Match rivers to counties they intersect
joined = gpd.sjoin(
    grouping_gdf[['countyName', 'geometry']],  # Left: counties
    summarizing_object_gdf[['riverName', 'geometry']],  # Right: rivers
    how='inner',
    predicate='intersects'
)

# After the spatial join, the geometries are named differently
# The original geometry from the left dataset is named 'geometry'
# The geometry from the right dataset is named 'geometry_'
# Calculate length of river segment within each county
joined['riverLength'] = joined.apply(
    lambda row: summarizing_object_gdf.loc[row.index_right, 'geometry'].intersection(row.geometry).length,
    axis=1
)

# Find the longest river per county
df = (
    joined.groupby(['countyName', 'riverName'])['riverLength']
    .sum()
    .reset_index()
    .sort_values('river_length', ascending=False)
    .drop_duplicates('countyName', keep='first')  # Keep longest river per county
    [['countyName', 'riverName', 'riverLength']]  # Note that it is important to keep the column riverName to tell which river is the longest in each county
    .reset_index(drop=True)
)

# rename the most important column
df = df.rename(columns={{'countyName': 'Name'}})

**Return ONLY valid Python code implementing this workflow. Do not include explanations or comments.**  

**aggregation_request:** {aggregation_request}


""",                     
            input_variables=["grouping_adf", "summarizing_adf", "aggregation_request"],                                      
        )                                   

        # prompt_instance = prompt.format(grouping_adf=str(grouping_adf), summarizing_adf=str(summarizing_adf), aggregation_request=aggregation_request) 	
        # logger.debug('-' * 70)
        # logger.debug(prompt_instance)

        request_router = prompt | self.llm | StrOutputParser()  
        result = request_router.invoke({"grouping_adf": str(grouping_adf), "summarizing_adf": str(summarizing_adf), "aggregation_request": aggregation_request}) 
        return result


    #-------------------------------------------------------
    # handle aggregation queries
    #-------------------------------------------------------    
    def execute_aggregation_plan(self, aggregation_plan):

        grouping_object_request = aggregation_plan["query_plan"][0]["request"]
        
        logger.debug('-'*70)
        logger.debug(f'processing grouping_object_request: {grouping_object_request}')        

        grouping_adf = self.process_request(grouping_object_request)
        logger.debug(f'grouping_objects: \n{grouping_adf}')        	        

        summarizing_object_request = aggregation_plan["query_plan"][1]["request"]

        logger.debug('-'*70)
        logger.debug(f'processing summarizing_object_request: {summarizing_object_request}')                

        grouping_bbox = grouping_adf.df.total_bounds
        logger.debug(f'grouping_bbox: {grouping_bbox}')

        # create an optimzied summarizing_object_request
        describe_bbox = lambda bbox: f"from ({bbox[0]:.4f}, {bbox[1]:.4f}) to ({bbox[2]:.4f}, {bbox[3]:.4f})"
        bbox_desc = describe_bbox(grouping_bbox)        
        optimized_summarizing_object_request = f"{summarizing_object_request} intersects with the bounding box {bbox_desc}"
        logger.debug(f'optimized_summarizing_object_request: {optimized_summarizing_object_request}')

        summarizing_adf = self.process_request(optimized_summarizing_object_request)
        summarizing_adf.title = f"{summarizing_object_request} intersects with the bounding box of {grouping_object_request}" 
        logger.debug(f'summarizing_objects: \n{summarizing_adf}') 

        # aggregation step
        aggregation_request = aggregation_plan["query_plan"][2]["request"] 

        logger.debug('-'*70)
        logger.debug(f'processing aggregation request: {aggregation_request}')

        code = self.get_code_for_aggregation_request(grouping_adf, summarizing_adf, aggregation_request)
        code = code.strip()
        if code.startswith("```python"):
            start_index = code.find("```python") + len("```python")
            end_index = code.find("```", start_index)
            code = code[start_index:end_index].strip()
        elif code.startswith("```"):
            start_index = code.find("```") + len("```")
            end_index = code.find("```", start_index)
            code = code[start_index:end_index].strip()
        logger.debug(f'Translate into Python code:\n\n{code}\n')
   
        global_vars = {
            "grouping_gdf": grouping_adf.df,
            "summarizing_object_gdf": summarizing_adf.df,
            "gpd": gpd,
        }
        exec(code, global_vars)
        df = global_vars.get("df")
        title = create_title_from_request(self.llm, aggregation_request) 
        return DataFrameAnnotation(df, title)        

    #-------------------------------------------------------
    # process queries
    #-------------------------------------------------------    
    def process_request(self, request: str):

        if self.check_aggregation_query(request)['is_aggregation_query']:
            logger.debug("the current request is an aggregration request")
            
            aggregation_plan = self.get_aggregation_plan(request)
            logger.debug(f"aggregation plan: \n{json.dumps(aggregation_plan, indent=4)}")

            dfa = self.execute_aggregation_plan(aggregation_plan)
            dfa.creator = 'User'
            self.data_repo.add_dataframe_annotation(dfa)

            logger.debug(f"The result of aggregation request: \n\n{dfa}\n")
            return dfa

        plan = self.get_query_plan(request)
        logger.debug(f'create query plan: \n{json.dumps(plan, indent=4)}')  
       
        dfa = None
        for index, request_item in enumerate(plan):
            logger.debug('-'*70)
            logger.debug(f"request {index+1}:  {request_item['data_source']} : {request_item['request']}")
            retriever = self.get_retriever(request_item['data_source'])
            if retriever:
                if isinstance(retriever, DataFrameRetriever):
                    if self.data_repo.contain_dataframe_annotation(request_item['request']):
                        index = self.data_repo.get_dataframe_annotation(request)
                        dfa = self.data_repo.dataframe_annotations[int(index)]
                    else:
                        dfa = retriever.get_dataframe_annotation(self.data_repo, request_item['request'])
                        if dfa.df.empty:
                            raise Exception(f"Failed to fetch data for the query: {request_item['request']}")
                        if index < len(plan) - 1:
                            dfa.creator = 'System'
                        self.data_repo.add_dataframe_annotation(dfa)
        logger.debug("=" * 70)
        logger.debug("current data repo")
        logger.debug(self.data_repo)
        return dfa

    def get_text_for_off_topic_request(self, request: str):
        template = PromptTemplate(
            template="""You are an expert of following data retrievers and their data sources:
{retriever_descriptions}                 
Based on the provided context, use easy understanding language to answer the question.
Please politely reject any requests for searching any websites. Your answer should be plain text
without any explanation.

You can categorize the data based on their data sources.

Don't mention that you can only load one entity type at a time from each database. We'll use query 
planning to load different entity types multiple times to solve complex problems.

Use the given example requests separated by each data source if the user is asking for example requests.

Question:{question}?

Answer: """,
            input_variables=["question", "retriever_descriptions"],
        )
        rag_chain = template | self.llm | StrOutputParser()

        retriever_desc = ""
        for index, df_retriever in enumerate(self.dataframe_retrievers):
            retriever_desc = f"""{retriever_desc}
{df_retriever.get_description()}
Users can request to find or load the data listed from {df_retriever.get_name()}. Here are some of
example requests: 
{df_retriever.get_examples()}
"""
        for index, text_retriever in enumerate(self.text_retrievers):
            retriever_desc = f"""{retriever_desc}
{text_retriever.get_description()}
Users can get answers for any questions from {text_retriever.get_name()}. Here are some of
example questions: 
{text_retriever.get_examples()}
        """

        prompt_instance = template.format(retriever_descriptions=retriever_desc, question=request)
        print('-' * 70)
        print(prompt_instance)
        print('-' * 70)

        result = rag_chain.invoke({"question": request, "retriever_descriptions": retriever_desc})
        return result

    def get_request_plan(self, request: str):
        retriever_description = ""
        for index, df_retriever in enumerate(self.dataframe_retrievers):
            retriever_description = f"""{retriever_description}
{df_retriever.get_description()} 
Use "{df_retriever.get_name()}" if the requested data can be loaded by this retriever.\n"""

        for index, text_retriever in enumerate(self.text_retrievers):
            retriever_description = f"""{retriever_description}
{text_retriever.get_description()} 
Use "{text_retriever.get_name()}" if the answer to the users' request/question can be loaded by this retriever.\n"""

        prompt = PromptTemplate(
            template="""You are an expert at query planning by using different data retrievers. 
The following are the descriptions of all data retrievers and their data sources:
{retriever_descriptions}
Each data retriever is designed to fetch one dataframe at a time, corresponding to a 
specific entity or attribute described above. This means that a data retriever processes
a single request for one dataframe at a time.

Request: {request}
    
Please think step by step and transform this request into a list of requests, ensuring
that each request uses only one particular data retriever and the data fetched by prior
requests in the list.  To use the data fetched by the previous request, the previous 
requests can simply be replicated in a later request.
    
Please make sure the following conditions are satisfied for each request in the list:
    1. Each request must be independent and only return one dataframe.
    2. Don't include any reference to a previous request, e.g. "the first request" in a request.
    3. Don't include any pronouns like "it" or "they" in a request.   
    4. Using the same expressions to refer to data fetched by earlier requests.
    5. For two consecutive requests, the latter request should be a semantic expansion of the
       previous request.

  Return a JSON list with objects containing the following fields:
    1. request: the text for the request  
    2. data_source: the retriever name that should be used.
    3. origin: Use “User” if the request contains data that the user has asked to find it in the original request. 
       Use “System” otherwise.

If the request is beyond the scope of the all data retrievers, route the original request to the
data_source "Other". 

[ Example 1 ] 
Find the population for all counties downstream of the coal power station with the name 'Pike Rock' 
along Muskingum River.

This request can be transformed into a request list as follows:

1. request: Find the coal power station with the name 'Pike Rock'.
   data_source: Energy Atlas
   origin: System

2. request: Find all counties downstream of "the coal power station with the name 'Pike Rock'" along Muskingum River.
   data_source:  WEN-OKN Database
   origin: System

3. request: Find the population for "all counties downstream of the coal power station with the name 'Pike Rock' along Muskingum River"
   data_source: Data Commons
   origin: User

Note that each request in the list is independent, i.e., each request can be completely independent of the others. 
The latter request is always a simple enrichment of the previous request.

[ Example 2 ]
Load flood event counts for all counties Scioto river flows through.

Return:
[
    {{
        "request": "Find all counties 'Scioto River' flows through",
        "data_source": "WEN-OKN Database",
        "origin": "System"
    }},
    {{
        "request": "Find flood event counts for all counties 'Scioto River' flows through",
        "data_source": "Data Commons",
        "origin": "User"
    }}
]

[ Example 3 ]
Find Ohio River and all counties it flows through.

This request asks to return Ohio River in a dataframe, and all counties through which Ohio River flows.
Return:
[
    {{
        "request": "Find Ohio River",
        "data_source": "WEN-OKN Database",
        "origin": "User"
    }},
    {{
        "request": "Find all counties Ohio River flows through",
        "data_source": "WEN-OKN Database",
        "origin": "User"
    }}
]

[ Example 4 ]
Find all dams in Ross county and all power stations in San Diego.

Return:
[
    {{
        "request": "Find all dams in Ross county",
        "data_source": "WEN-OKN Database",
        "origin": "User"
    }},
    {{
        "request": "Find all power stations in San Diego",
        "data_source": "WEN-OKN Database",
        "origin": "User"
    }}
]

[ Example 5 ]
Find all coal mines along the Ohio River.

Return:
[
    {{
        "request": "Find Ohio River",
        "data_source": "WEN-OKN Database",
        "origin": "System"
    }},
    {{
        "request": "Find all coal mines along the Ohio River",
        "data_source": "Energy Atlas",
        "origin": "User"
    }}
]


[ Example 6 ]
Find all watersheds in Ohio State. 

You can return the following:
[
    {{
        "request": "Find Ohio State",
        "data_source": "WEN-OKN Database",
        "origin": "System"
    }},
    {{
        "request": "Find all watersheds in Ohio State",
        "data_source": "Energy Atlas",
        "origin": "User"
    }}
]

[ Example 7 ]
Find all rivers that flow through the Roanoke basin.

You can return the following:
[
    {{
        "request": "Find the Roanoke basin",
        "data_source": "Energy Atlas",
        "origin": "System"
    }},
    {{
        "request": "Find all rivers that flow through the Roanoke basin",
        "data_source": "WEN-OKN Database",
        "origin": "User"
    }}
]


[ Example 8 ]
Find all basins in Ohio State. 

You can return the following:
[
    {{
        "request": "Find Ohio State",
        "data_source": "WEN-OKN Database",
        "origin": "System"
    }},
    {{
        "request": "Find all basins in Ohio State",
        "data_source": "Energy Atlas",
        "origin": "User"
    }}
]



[ Example 9 ]
Find the populations of the tracts of all power stations in Ohio that are at risk of flooding at 2 PM on July 1, 2025.

You can return
[
  {{
    "request": "Find all power stations at risk of flooding in Ohio at 2 PM on July 1, 2025",
    "data_source": "Energy Atlas",
    "origin": "System"
  }},
  {{
    "request": "Find the tracts of all power stations at risk in flooding in Ohio at 2 PM on July 1, 2025",
    "data_source": "Energy Atlas",
    "origin": "System"
  }},
  {{
    "request": "Find the populations of the tracts of all power stations at risk of flooding in Ohio at 2 PM on July 1, 2025",
    "data_source": "Data Commons",
    "origin": "User"
  }}
]
Please note that Energy Atlas can find all power stations in risk of flooding in some states or counties in an hour. 




Don't put any quotes at the top of the returned JSON list.

Answer: """,
            input_variables=["retriever_descriptions", "request"],
        )

        prompt_instance = prompt.format(retriever_descriptions=retriever_description, request=request)
        print('-' * 70)
        print(prompt_instance)

        request_plan = prompt | self.llm | JsonOutputParser()

        start = time.time()
        json_result = request_plan.invoke({"request": request,
                                           "retriever_descriptions": retriever_description})
        end = time.time()
        print("==============> CHECK: ", end - start)

        print('-' * 70)
        print(json.dumps(json_result, indent=4))

        print('-' * 70)
        for index, df_retriever in enumerate(self.dataframe_retrievers):
            if df_retriever.get_inner_join():
                print("Optimize", df_retriever.get_name())
                # new_request_plan = remove_consecutive_system_objects(json_result, df_retriever.get_name())
                new_request_plan = find_and_remove_consecutive(json_result, df_retriever.get_name())
                print(json.dumps(new_request_plan, indent=4))

                json_result = new_request_plan

        # reviewed_plan = self.review_request_plan(request, json_result)
        # print('-' * 70)
        # print("Reviewed Result")
        # print(json.dumps(reviewed_plan, indent=4))

        return json_result

    def review_request_plan(self, request: str, plan: str):
        prompt = PromptTemplate(
            template="""
[Original Request]
{request}

[Request Plan for Handling the Request]
{plan}

For each request in the request plan, check whether the data returned by the request is explicitly
required in the original request by the term "find" or "fetch" or "load".  

Return a JSON list with objects containing the following fields:
   1. "original_request"
   2. "request"
   3. "original_required"
   
Please return the JSON result only without any explanation.
            """,
            input_variables=["request", "plan"],
        )

        plan = json.dumps(plan, indent=4)
        plan = plan.replace("{", "{{")
        plan = plan.replace("}", "}}")

        print('-' * 70)
        print(plan)

        prompt_instance = prompt.format(request=request,
                                        plan=json.dumps(plan))
        print('-' * 70)
        print(prompt_instance)

        request_plan = prompt | self.llm | JsonOutputParser()
        json_result = request_plan.invoke({"request": request,
                                           "plan": json.dumps(plan)})
        return json_result
