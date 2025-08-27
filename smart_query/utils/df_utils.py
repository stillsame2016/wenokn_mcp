
import re
import sys
import geopandas as gpd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from shapely import wkt
from datetime import datetime


def get_column_name_parts(column_name):
    return re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', column_name)


def df_to_gdf(df):
    column_names = df.columns.tolist()
    geometry_column_names = [x for x in column_names if x.endswith('Geometry')]
    df['geometry'] = df[geometry_column_names[0]].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.drop(columns=[geometry_column_names[0]], inplace=True)

    column_name_parts = get_column_name_parts(column_names[0])
    column_name_parts.pop()
    gdf.attrs['data_name'] = " ".join(column_name_parts).capitalize()

    for column_name in column_names:
        tmp_column_name_parts = get_column_name_parts(column_name)
        tmp_name = tmp_column_name_parts.pop()
        tmp_data_name = " ".join(column_name_parts).capitalize()
        if gdf.attrs['data_name'] == tmp_data_name:
            gdf.rename(columns={column_name: tmp_name}, inplace=True)
    return gdf


def create_title_from_request(llm, request):
    prompt = PromptTemplate(
        template="""
We loaded data based on the following user request:
{request}

Please create a title to the loaded data based on the user's request. Don't include any explanations.
Return a title only. 

For example, if the request is "Find the Ohio river", just return "Ohio river".

For example, if the request is "Find all counties the Scioto River flows through", just return "All counties the Scioto River flows through"

Answer: """,
        input_variables=["request"],
    )

    request_router = prompt | llm | StrOutputParser()
    return request_router.invoke({"request": request})


def remove_consecutive_system_objects(json_list, data_source):
    result = []
    temp_group = []

    for obj in json_list:
        if obj['data_source'] == data_source:
            temp_group.append(obj)
            if obj['origin'] == 'User':
                if len(temp_group) > 1 and all(item['origin'] == 'System' for item in temp_group[:-1]):
                    result.extend([item for item in temp_group if item['origin'] != 'System'])
                else:
                    result.extend(temp_group)
                temp_group = []
        else:
            if temp_group:
                result.extend(temp_group)
                temp_group = []
            result.append(obj)

    if temp_group:
        result.extend(temp_group)

    return result


def find_and_remove_consecutive(data, data_source):
    for i in range(0, len(data)-1):
        element = data[i]
        next_element = data[i+1]
        if element["data_source"] == data_source and element["origin"] == "System" and \
                next_element["data_source"] == data_source:
            del data[i]
            return find_and_remove_consecutive(data, data_source)
    return data