import requests
import sparql_dataframe
import traceback
import sys                                                                                                                                                        
from datetime import datetime
from shapely import wkt
from shapely.geometry import box
import geopandas as gpd

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser                                                                                            
from langchain_core.prompts import PromptTemplate    

from smart_query.data_repo.data_repository import DataRepository
from smart_query.data_repo.dataframe_annotation import DataFrameAnnotation
from smart_query.data_retriever.base_retriever import DataFrameRetriever
from smart_query.utils.df_utils import df_to_gdf, create_title_from_request
from smart_query.utils.string_utils import strip_sparql_decoartion
from smart_query.data_retriever.text_to_sparql import get_candidate_concepts
from smart_query.utils.logger import get_logger 

logger = get_logger(__name__)
                                                                                                                                                                  
class WENOKNRetriever(DataFrameRetriever):

    def get_description(self):
        return f"""Dataframe Retriever: "{self.name}"
This retriever can load a GeoDataFrame for one of the following entity types at a time with join conditions:
    1. Buildings in Ohio: IDs and geometries.
    2. Power Stations in Ohio: IDs and geometries.
    3. Underground Storage Tanks in Ohio: IDs and geometries.
    4. Counties in the USA: Names and geometries.
    5. States in the USA: Names and geometries.
    6. Earthquakes in the USA: Magnitudes, times, and geometries.
    7. Rivers in the USA: Names and geometries.
    8. Dams: Names and geometries.
    9. Drought Zones in the USA (2020, 2021, 2022): Geometries.
    10. Hospitals: Names, addresses, types, phones, and geometries.
    11. Neighbor.
    12. Upstream and Downstream.
    13. Stream Gages: Information of gages locations and names in USA.

Note that the above entities only have the attributes listed above. Any attributes that are not 
listed are not available in this data source. For example, the population attribute is not listed 
above, so there is no population-related data in this data source.

For example, this retriever can find a USA state or country or river by its name.

"""

    # ------------------------------------------------------------
    # get_examples
    # ------------------------------------------------------------
    def get_examples(self):
        return """Find Ohio River.
Find all neighbor counties of Ross county. 
Find all downstream counties of Morgan county on Muskingum river.
Find 10 power stations in Morgan county.
Find all dams located upstream of power station dpjc6wtthc32 along the Muskingum river. """

    # ------------------------------------------------------------
    # check if use other sources
    # ------------------------------------------------------------
    def use_other_sources(self, request):
                                                                                                                                                                       
        prompt = PromptTemplate(                                                                                                                                       
            template="""
The following are the descriptions of the this data retriever and its data:                                                                                    
{retriever_description}                                                                                                                                               

Please return a JSON string with the following a boolean field "result" and a string field "reason". 
Find the nouns in the user's request and determine if they are one of the entity types listed by 
this retriever, or if they are instances of one of the listed entity types. Returns the "result" field as false                                     
and the "reason" field as "" if all the nouns are, otherwise return the "result" field as true and the the "reason" 
field as the nouns that are not the listed entity types or instances of these types.

Request: {request}

[ Example 1 ]
Find the Scioto River.

The nouns in this request contain only "Scioto River" which is instace of the entity type of "River", so return 
{{
   "result": false,
   "reason": ""
}}


[ Example 2 ]
Find all adjacent states to the state of Ohio.
Find Florida State.

The nouns in both requests contain "state",  "Ohio" and "Florida State" , which all of them are the entity type or instance of "State", so return  
{{
   "result": false,
   "reason": ""
}}


[ Example 3 ]
Find all stream gages in the watersheds that feed into the Scioto River.

The nouns of this request contain "stream gages", "watersheds" and "Scioto River", where "stream gages" is the same as the
entity type "Stream Gages", "Scioto River" is an instance of the entity type "River", but "watersheds" is not one of the listed
entity types or their instances. So return
 {{
   "result": true,
   "reason": "watershed"
}}


[ Example 4 ]
Find all counties in Kentucky State

The nouns of this request contain "counties" and "Kentucky State", where "counties" are the same as the entity type "County" and 
"Kentucky State" is an instance of the entity type "State". So return 
{{
   "result": false,
   "reason": ""
}}


[ Example 5 ]
Find all rivers that flow through Dane County in Wisconsin.

The nouns of this request contain "rivers", "Dane County" and "Wisconsin", where "rivers" are the same as the entity type "River" and 
"Wisconsin" is an instance of the entity type "State" and "Dane County" is an instance of the entity type "County". So return 
{{
   "result": false,
   "reason": ""
}}

[ Example 6 ]
Find all dams in Kentucky State.

The nouns of this request contain "dams" and "Kentucky State", where "counties" are the same as the entity type "Dam" and 
"Kentucky State" is an instance of the entity type "State". So return 
{{
   "result": false,
   "reason": ""
}}

[ Example 7]
What stream gages are on the Yahara River in Dana County, WI?

The nouns of both request contain "stream gages", "Yahara River", "Dana County" and "WI", where "stream gages" is the same as the
entity type "Stream Gages", "Yahara River" is an instance of the entity type "River", "Dana County" is an instance of "County" and "WI" is an instance of "State". So return
 {{
   "result": false,
   "reason": ""
}}

Please note "Dana County, WI" is an instance of "County".

[ Example 8 ]
Find all dams located upstream of the power station dpjc6wtthc32 along the Muskingum river.

The nouns of the request contains "dam", "power station dpjc6wtthc32" and "Muskingum river", where "dam" is the same as the entity
type "Dam", "power station dpjc6wtthc32" is an instance of "Power Station" and "Muskingum river" is an instance of "River". So return

{{                                                                                                                                                                             
   "result": false,                                                                                                                                                            
   "reason": ""                                                                                                                                                                
}}    


Note that the entity type "power station" is different to the entity type "power plant".  

Note that "Ross County" is an instance of the entity type "County". 

Note that "Ohio River" is an instance of the entity type "River".

 
""",                     
            input_variables=["retriever_description", "request"],                                      
        )                                   

        # prompt_instance = prompt.format(request=request, retriever_description=self.get_description()) 	
        # logger.debug('-' * 70)
        # logger.debug(prompt_instance)

        request_router = prompt | self.llm | JsonOutputParser()  
        result = request_router.invoke({"request": request, 
                                        "retriever_description": self.get_description()}) 
        logger.debug(f"Use other sources: {result}")
        return result

    # ------------------------------------------------------------
    # process the query
    # ------------------------------------------------------------
    def get_dataframe_annotation(self, data_repo: DataRepository, atomic_request: str):
        
        logger.debug(f"Atomic Request: {atomic_request}")
       
        # check if the atomic_request uses entities out of the WENOKN database
        if self.use_other_sources(atomic_request)['result']:
            logger.debug(f"Need to use other sources")
            return self.get_dataframe_annotation_with_additional_sources(data_repo, atomic_request)

        max_tries = 5
        tried = 0
        while tried < max_tries:
            logger.debug(f"Try {tried + 1}")
            try:
                sparql_query = get_candidate_concepts(atomic_request)
                sparql_query = sparql_query.replace('\\n', '\n').replace('\\"', '"').replace('\\t', ' ')
                sparql_query = strip_sparql_decoartion(sparql_query)
                logger.debug(f"Translate into SPARQL:\n {sparql_query}")  

                endpoint = "http://132.249.238.155/repositories/wenokn_ohio_all"
                df = sparql_dataframe.get(endpoint, sparql_query)
                gdf = df_to_gdf(df)

                title = create_title_from_request(self.llm, atomic_request)
                return DataFrameAnnotation(gdf, title)
            except Exception as e:
                logger.debug(f"Error: {str(e)}") 
                traceback.print_exc()
                tried += 1
        return None

    # ------------------------------------------------------------
    # process the query with additional data sources
    # ------------------------------------------------------------
    def get_dataframe_annotation_with_additional_sources(self, data_repo: DataRepository, atomic_request: str):

        logger.debug(f"processing with other sources: {atomic_request}")                                                                                                                                                                                       
        prompt = PromptTemplate(                                                                                                                                       
            template="""
The following are the descriptions of the this data retriever and its data:                                                                                    
{retriever_description}                                                                                                                                               

The following is the available data in the data repository:
{data_repo}

The following is the user request:
{request}

This request asks for entities from this data retriever that satisfies certain conditions, some of 
which are described using entities out of the scope of this data retriever, but they may be available
in the data repository. Your task is to provide Python code which converts this request into another 
request in natural language using the data in this data retriever only as a Python variable 
converted_request. Please return the Python code only without any explanation. Don't include any 
print statement. Don't add ``` around the code.

[ Example 1 ]
Find all counties downstream of the coal mine with the name "Century Mine" along Ohio River.

First we find the latitude and longitude of the coal mine with the name "Century Mine". Then convert the original 
request into a request with the data in this retriever only. The following is returned Python code:

    gdf =  .... geodataframe for the coal mine with the name "Century Mine" in the data repository
    latitude = gdf.iloc[0]['Latitude']
    longitude = gdf.iloc[0]['Longitude']
    converted_request = f"Find all counties downstream of the coal mine with the location({{latitude}}, {{longitude}}) along Ohio River."

[ Example 2 ]
Find all stream gages within the watershed with the name Headwaters Black Fork Mohican River.

Find out if the data repository has a geodataframe for the watershed with the name Headwaters Black Fork Mohican River.

If the data repository doesn't contain a geodataframe for the watershed with the name Headwaters Black Fork Mohican River, 
then return the following code:
    raise Exception("The data for the watershed with the name Headwaters Black Fork Mohican River is missing. Please load it first.")

If the data repository contains a geodataframe for the watershed with the name Headwaters Black Fork Mohican River, then return 
the valid Python code in the following format:
    gdf1 = <replace by the variable of the geodataframe for the watershed with the name Headwaters Black Fork Mohican River if you found one>
    # Get stream gages in the bounding box of the watershed
    minx, miny, maxx, maxy = gdf1.total_bounds
    gdf1_bbox = box(minx, miny, maxx, maxy)
    gdf1_bbox_wkt = gdf1_bbox.wkt 
    gdf2 = get_gdf_from_data_request(f"Find all stream gages within {{gdf1_bbox_wkt}}).")
    gdf = gpd.sjoin(gdf2, gdf1, how="inner", predicate="intersects")
    gdf = gdf[gdf2.columns].drop_duplicates()
    converted_request = None

[ Example 3 ]
Find all rivers that flow through the Roanoke basin.

Find out if the data repository has a geodataframe for the Roanoke basin.

If the data repository doesn't contain a geodataframe for the Roanoke basin, then return the following code:
    raise Exception("The data for the Roanoke basin is missing. Please load it first.")

If the data repository has a geodataframe containing the Roanoke basin, then return the valid Python code in the following format:
    gdf1 = <replace by the variable of the geodataframe for the Roanoke basin if you found one>
    # Get rivers in the bounding box of the Roanoke basin
    minx, miny, maxx, maxy = gdf1.total_bounds
    gdf1_bbox = box(minx, miny, maxx, maxy)
    gdf1_bbox_wkt = gdf1_bbox.wkt 
    gdf2 = get_gdf_from_data_request(f"Find all rivers that flow through {{gdf1_bbox_wkt}}).")
    gdf = gpd.sjoin(gdf2, gdf1, how="inner", predicate="intersects")
    gdf = gdf[gdf2.columns].drop_duplicates()
    converted_request = None

[ example 4 ]
Find all stream gages in the watersheds feed into Scioto River.

Find out if the data repository has a geodataframe for the watersheds feed into Scioto River.

If the data repository doesn't contain a geodataframe for the watersheds feed into Scioto River, 
then return the following code:
    raise Exception("The data for the watersheds feed into Scioto River is missing. Please load it first.")

If the data repository contains a geodataframe for the watersheds feed into Scioto River, then return 
the valid Python code in the following format:
    gdf1 = <replace by the variable of the geodataframe for the watersheds feed into Scioto River if you found one>
    # Get stream gages in the bounding box of the watershed
    minx, miny, maxx, maxy = gdf1.total_bounds
    gdf1_bbox = box(minx, miny, maxx, maxy)
    gdf1_bbox_wkt = gdf1_bbox.wkt 
    gdf2 = get_gdf_from_data_request(f"Find all stream gages within {{gdf1_bbox_wkt}}).")
    gdf = gpd.sjoin(gdf2, gdf1, how="inner", predicate="intersects")
    gdf = gdf[gdf2.columns].drop_duplicates()
    converted_request = None
        
""",                     
            input_variables=["retriever_description", "data_repo", "request"],     
        )                                   

        data_repo_description = f""
        if len(data_repo.dataframe_annotations) > 0:
            for index, adf in enumerate(data_repo.dataframe_annotations):
                data_repo_description = f"{data_repo_description}\n{str(adf)}\n" \
                                        f"How to Access: data_repo.dataframe_annotations[{index}].df\n"
        else:
            data_repo_description = "The Data Repository is empty."

        # prompt_instance = prompt.format(data_repo=data_repo_description,
        #                                 request=atomic_request,
        #                                 retriever_description=self.get_description())
        # logger.debug('-' * 70)
        # logger.debug(prompt_instance)

        request_router = prompt | self.llm | StrOutputParser()  
        code = request_router.invoke({"request": atomic_request, 
                                       "data_repo": data_repo_description,
                                       "retriever_description": self.get_description()}) 

        code = code.strip()
        if code.startswith("```python"):
            start_index = code.find("```python") + len("```python")
            end_index = code.find("```", start_index)
            code = code[start_index:end_index].strip()
        elif code.startswith("```"):
            start_index = code.find("```") + len("```")
            end_index = code.find("```", start_index)
            code = code[start_index:end_index].strip()

        logger.debug(f"Translate the request into Python code: \n\n{code}\n")

        global_vars = {
            'data_repo': data_repo,
            'box': box,
            'gpd': gpd,
            'get_gdf_from_data_request': get_gdf_from_data_request,
        }
        exec(code, global_vars)

        refined_request = global_vars.get("converted_request")
        if refined_request:
            logger.debug(f"Refined Request: {refined_request}")

            max_tries = 5
            tried = 0
            while tried < max_tries:
                logger.debug(f"Try: {tried +1}")
                try:
                    sparql_query = get_candidate_concepts(refined_request)
                    sparql_query = sparql_query.replace('\\n', '\n').replace('\\"', '"').replace('\\t', ' ')
                    sparql_query = strip_sparql_decoartion(sparql_query)
                    logger.debug(f"Translate into SPARQL: \n\n{sparql_query}\n")  

                    endpoint = "http://132.249.238.155/repositories/wenokn_ohio_all"
                    df = sparql_dataframe.get(endpoint, sparql_query)
                    gdf = df_to_gdf(df)

                    title = create_title_from_request(self.llm, atomic_request)
                    return DataFrameAnnotation(gdf, title)
                except Exception as e:
                    logger.debug(f"Error: {str(e)}") 
                    traceback.print_exc()
                    tried += 1
            return None
        
        if "gdf" in global_vars:
            gdf = global_vars['gdf']
            title = create_title_from_request(self.llm, atomic_request) 
            logger.debug(f"Result: {title} : {gdf.shape}")   
            return DataFrameAnnotation(gdf, title)   
           

def get_gdf_from_data_request(request):
    sparql_query = get_candidate_concepts(request)                                                                                                 
    sparql_query = sparql_query.replace('\\n', '\n').replace('\\"', '"').replace('\\t', ' ')                                                               
    sparql_query = strip_sparql_decoartion(sparql_query) 
    logger.debug(f"Translate the request into SPARQL: {request} \n {sparql_query}\n")   
                                                                                                                                                                       
    endpoint = "http://132.249.238.155/repositories/wenokn_ohio_all"                                                                                       
    df = sparql_dataframe.get(endpoint, sparql_query)                                                                                                      
    gdf = df_to_gdf(df)                         
    return gdf
