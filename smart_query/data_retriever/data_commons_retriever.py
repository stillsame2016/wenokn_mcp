import json
import requests
import datacommons_pandas as dc

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from smart_query.data_repo.data_repository import DataRepository
from smart_query.data_repo.dataframe_annotation import DataFrameAnnotation
from smart_query.data_retriever.base_retriever import DataFrameRetriever
from smart_query.data_retriever.data_commons_help import ndp_search
from smart_query.utils.logger import get_logger 

logger = get_logger(__name__)


def get_variables_for_dcid(dcid_list, variable_name_list):
    _df = dc.build_multivariate_dataframe(dcid_list, variable_name_list)
    _df.insert(0, 'Name', _df.index.map(dc.get_property_values(_df.index, 'name')))
    _df['Name'] = _df['Name'].str[0]
    return _df


def get_time_series_dataframe_for_dcid(dcid_list, variable_name):
    _df = dc.build_time_series_dataframe(dcid_list, variable_name)
    _df.insert(0, 'Name', _df.index.map(dc.get_property_values(_df.index, 'name')))
    _df['Name'] = _df['Name'].str[0]

    columns = _df.columns.to_list().remove('Name')
    _df = _df.melt(
        ['Name'],
        columns,
        'Date',
        variable_name,
    )
    _df = _df.dropna()
    _df = _df.drop_duplicates(keep='first')
    _df.variable_name = variable_name
    return _df


def get_dcid_from_county_name(county_name):
    simple_query = f"""
                    SELECT ?geoId
                    WHERE {{
                      ?county typeOf County .
                      ?county name '{county_name}' .
                      ?county dcid ?geoId .
                    }}
                    LIMIT 1
                 """
    try:
        # Execute the simple query
        dcid_dict = dc.query(simple_query)
        dcid = [item['?geoId'] for item in dcid_dict]
        return dcid[0]
    except Exception as ex:
        return None


def get_dcid_from_state_name(state_name):
    simple_query = f"""
                    SELECT ?geoId
                    WHERE {{
                      ?state typeOf State .
                      ?state name '{state_name}' .
                      ?state dcid ?geoId .
                    }}
                    LIMIT 1
                 """
    try:
        # Execute the simple query
        dcid_dict = dc.query(simple_query)
        dcid = [item['?geoId'] for item in dcid_dict]
        return dcid[0]
    except Exception as ex:
        return None


def get_dcid_from_country_name(country_name):
    simple_query = f"""
                    SELECT ?geoId
                    WHERE {{
                      ?country typeOf Country .
                      ?country name '{country_name}' .
                      ?country dcid ?geoId .
                    }}
                    LIMIT 1
                 """
    try:
        # Execute the simple query
        dcid_dict = dc.query(simple_query)
        dcid = [item['?geoId'] for item in dcid_dict]
        return dcid[0]
    except Exception as ex:
        return None


class DataCommonsRetriever(DataFrameRetriever):

    def get_description(self):
        return f"""Dataframe Retriever: "{self.name}"
Data Commons contains data on tens of thousands of time-series variables for one or multiple counties, states, 
countries, or locations. Below are examples of these time series variables:
    Area_FloodEvent
    Count_Person (for population)
    Count_FireEvent
    Count_FlashFloodEvent
    Count_FloodEvent
    Count_HailEvent
    Count_HeatTemperatureEvent
    Count_HeatWaveEvent
    Count_HeavyRainEvent
    Count_Person_Employed
    Median_Income_Person
    CountOfClaims_NaturalHazardInsurance_BuildingStructureAndContents_FloodEvent
    Max_Rainfall
    Max_Snowfall
    SettlementAmount_NaturalHazardInsurance_BuildingContents_FloodEvent
    SettlementAmount_NaturalHazardInsurance_BuildingStructureAndContents_FloodEvent
    SettlementAmount_NaturalHazardInsurance_BuildingStructure_FloodEvent
    FemaSocialVulnerability_NaturalHazardImpact
    
Note that Data Commons can load a time serial variable in a state, or in a county, or in all counties in a state. 
"""

    def get_examples(self):
        return """
Find the populations for the Ross county and Pike county in Ohio.
Load the flood event counts for all counties in the Ohio State.
Load the populations for all counties Scioto river flows through.
        """

    def get_dataframe_annotation(self, data_repo: DataRepository, atomic_request: str):
        prompt = PromptTemplate(
            template="""
[ Data Repository ]
The Data Repository contains the following dataframes:
   {data_repo}
The meaning of each dataframe is defined by its title. Each dataframe can be accessed by the expression
given in "How to Access". For example, data_repo.dataframe_annotations[0].df can access the first dataframe. 
 
In Data Commons, dcid is used as index to access data. A dcid has the following format, 
for example, "geoid/39" is the dcid for the Ohio State and "geoid/06" is the dcid for the
California State. 

We have the following functions to get dcid from a state/county name:
    get_dcid_from_state_name(state_name)
    get_dcid_from_county_name(county_name) 
    get_dcid_from_country_name(country_name)
To call get_dcid_from_county_name, the county name must be in the format "San Diego County". 
Don't miss "County" in the name. 

Data Commons has the following statistical variables available for a particular place: {dc_variables}

The following code can fetch some variables data for some dcid from Data Commons:

    import datacommons_pandas as dc
    
    def get_time_series_dataframe_for_dcid(dcid_list, variable_name):
        _df = dc.build_time_series_dataframe(dcid_list, variable_name)
        _df.insert(0, 'Name', _df.index.map(dc.get_property_values(_df.index, 'name')))
        _df['Name'] = _df['Name'].str[0]
        return _df

[Example 1] 
Find the populations for all counties in Ohio, we can run the following code:

    # Get dcid for all counties in Ohio
    ohio_county_dcid = dc.get_places_in(["geoId/39"], 'County')["geoId/39"]

    # Get Count_Person (i.e., population) for all counties in Ohio
    df = get_time_series_dataframe_for_dcid(ohio_county_dcid, "Count_Person")
    title = "The Populations for All Counties in Ohio"

[Example 2]
Find the populations for the Ross county and Pike county in Ohio, we can run the 
following code:

    ross_pike_dcid = ['geoId/39131', 'geoId/39141']
    df = get_time_series_dataframe_for_dcid(ross_pike_dcid, "Count_Person")
    title = "The Populations for the Ross county and Pike county in Ohio"

[Example 3]
Find the populations of Ross county and Scioto county

    ross_scioto_dcid = [ get_dcid_from_county_name('Ross County'), get_dcid_from_county_name('Scioto County') ]
    df = get_time_series_dataframe_for_dcid(ross_scioto_dcid, "Count_Person")
    title = "The Populations for the Ross county and Scioto county in Ohio"

[Example 4] 
Find the populations of all counties where Scioto River flows through.

First, you should look at the meaning of each dataframe's title in the Data Repository, determine if the Data Repository 
contains a GeoDataFrame containing all counties Scioto River passes.

[ Case 1]
If the Data Repository is empty or doesn't contain a GeoDataFrame containing all counties Scioto River passes, then 
return the following code:
   raise Exception("The data for all counties Scioto River passes is missing. Please load it first.")

[ Case 2]
If the Data Repository is not empty and contains a DataFrame containing all counties Scioto River passes through with a column
"name" for county names, return the Python code in the following format:
    gdf = <Use the expression specified in "How to Access" to load the dataframe for all counties Scioto River passes through, for example, data_repo.dataframe_annotations[0].df>
    scioto_river_dcid = [ get_dcid_from_county_name(county_name) for county_name in gdf['name']]
    df = get_time_series_dataframe_for_dcid(scioto_river_dcid, "Count_Person")  
    title = "The Populations for All Counties where Scioto River Flows Through"

Please note that you need to decide whether the Data Repository contains a GeoDataFrame containing all counties Scioto 
River passes; don't include the judgement code in the returned code. You must return the code in Case 1 or Case 2 
exclusively.

Please note that for counties, gdb['name'] already has a value like "Greenup County". Don't add another "County" to these names. 

[Example 5]
Find the social vulnerability for all counties downstream of the coal mine with the name "Century Mine" along 
Ohio River.

First, you should look at the meaning of each dataframe's title in the Data Repository, determine if the Data Repository 
contains a GeoDataFrame containing all counties downstream of the coal mine with the name "Century Mine" along Ohio River 
with a column "Name" for county names.

[ Case 1 ]
If the Data Repository is empty or doesn't contain a GeoDataFrame containing all counties downstream of the coal mine 
with the name "Century Mine" along Ohio River, then return the following code:
   raise Exception("The data for all counties downstream of the coal mine with the name "Century Mine" along Ohio River is missing. Please load it first.")

[ Case 2 ] 
If the Data Repository is not empty and contains a GeoDataFrame containing all counties downstream of the coal mine 
with the name "Century Mine" along Ohio River, then return the following code: return the following 
code:
    gdf = <Use the expression specified in "How to Access" to load the dataframe for all counties downstream of the coal mine with the name "Century Mine" along Ohio River, for example, data_repo.dataframe_annotations[0].df>
    counties_dcid = [ get_dcid_from_county_name(county_name) for county_name in gdf['Name']]
    df = get_time_series_dataframe_for_dcid(counties_dcid, "FemaSocialVulnerability_NaturalHazardImpact")  
    title = "The Social Vulnerability for All Counties Downstream of the Coal Mine with the Name \"Century Mine\" along Ohio River"

Please note that you need to decide whether the Data Repository contains a GeoDataFrame containing all counties Scioto 
River passes; don't include the judgement code in the returned code. You must return the code in Case 1 or Case 2 
exclusively.

If the example data from gdf has a county name like 'Ross', then need to convert it to 'Ross County' to call 
get_dcid_from_county_name.

[ Request ]
The following is the request from the user:
    {request}

Please use pd.merge(df1, df2, on=df1.columns.to_list[:-1]) to merge two dataframes if needed. 

Please return the complete Python code only to implement the user's request without preamble or 
explanation. Don't include any print statement. Don't add ``` around the code. Make a title and 
save it to the variable title. 
""",
            input_variables=["request", "variables", "dc_variables"],
        )
        df_code_chain = prompt | self.llm | StrOutputParser()

        dc_variables = ""

        # response = requests.get(f"https://sparcal.sdsc.edu/api/v1/Utility/data_commons?search_terms={atomic_request}")
        # items = json.loads(response.text)

        items = ndp_search(atomic_request)
        for item in items:
            dc_variables = f"""{dc_variables}\n
    variable: {item['variable']}
    description: {item['name']}"""

        data_repo_description = f""
        if len(data_repo.dataframe_annotations) > 0:
            for index, adf in enumerate(data_repo.dataframe_annotations):
                data_repo_description = f"{data_repo_description}\n{str(adf)}\n" \
                                        f"How to Access: data_repo.dataframe_annotations[{index}].df\n"
        else:
            data_repo_description = "The Data Repository is empty."

        # prompt_instance = prompt.format(data_repo=data_repo_description,
        #                                request=atomic_request,
        #                                dc_variables=dc_variables)
        # print('-' * 70)
        # print(prompt_instance)

        code = df_code_chain.invoke({"request": atomic_request,
                                     "data_repo": data_repo_description,
                                     "dc_variables": dc_variables})
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
            "dc": dc,
            "data_repo": data_repo,
            "get_variables_for_dcid": get_variables_for_dcid,
            "get_time_series_dataframe_for_dcid": get_time_series_dataframe_for_dcid,
            "get_dcid_from_county_name": get_dcid_from_county_name,
            "get_dcid_from_state_name": get_dcid_from_state_name,
            "get_dcid_from_country_name": get_dcid_from_country_name
        }
        exec(code, global_vars)
        title = global_vars.get("title")
        df = global_vars.get("df")
        return DataFrameAnnotation(df, title)
