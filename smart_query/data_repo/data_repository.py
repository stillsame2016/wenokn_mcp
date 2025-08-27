from datetime import datetime, timedelta
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from smart_query.data_repo.dataframe_annotation import DataFrameAnnotation


class DataRepository:
    def __init__(self, llm):
        self.llm = llm
        self.dataframe_annotations = []

    def add_dataframe_annotation(self, df):
        if not isinstance(df, DataFrameAnnotation):
            raise TypeError("Only DataFrameAnnotation instances can be added")
        self.dataframe_annotations.append(df)

    def get_dataframe_annotation(self, index: int):
        try:
            return self.dataframe_annotations[index]
        except IndexError:
            raise IndexError(f"No dataframe at index '{index}'")

    def remove_dataframe_annotation(self, index: int):
        try:
            del self.dataframe_annotations[index]
        except IndexError:
            raise IndexError(f"No dataframe at index '{index}'")

    def remove_annotations_older_than(self, age_seconds):
        """Remove all dataframe annotations older than `age_seconds`."""
        cutoff_time = datetime.now() - timedelta(seconds=age_seconds)
        self.dataframe_annotations = [
            ann for ann in self.dataframe_annotations if ann.created_at >= cutoff_time
        ]

    def contain_dataframe_annotation(self, description: str):
        prompt = PromptTemplate(
            template="""
{data_repo}

Please determine if the dataframe by the following user request exists in the Data Repository. 
If it already exists, please return True, otherwise please return False. just return True or False. 
do not return any other explanation.

Note: Please note that “San Diego County” is not equivalent to “Southern San Diego County”.
       
Note: Please note that "Scioto River" is NOT semantically equivalent to the processed request 
"All basins that Scioto River flows through" because "Scioto River" gives a river with 
the name "Scioto" and "All basins that Scioto River flows through" tries to find all basins that the
Scioto River flows through".

Note: please note that "Find all watersheds in the Kanawha basin" is not equivalent to "Kanawha Basin".
"Find all watersheds in the Kanawha basin" tries to give all watershes intersect with the Kanawha basin.


User Request: {request}""",
            input_variables=["data_repo", "request"],
        )

        if len(self.dataframe_annotations) == 0:
            data_repo_description = "The Data Repository is empty."
        else:
            data_repo_description = f"""
The Data Repository contains the following dataframes:
   {self.__repr__()}
The meaning of each dataframe is defined by its title."""
        request_router = prompt | self.llm | StrOutputParser()
        result = request_router.invoke({"data_repo": data_repo_description, "request": description})
        return result == 'True'

    def remove_dataframe_annotation(self, description: str):
        pass

    def get_dataframe_annotation(self, description: str):
        prompt = PromptTemplate(
            template="""
{data_repo}

Please determine if the dataframe by the following user request exists in the Data Repository. 
If it already exists, please return the index, otherwise please return -1. just return -1 or an index. 
do not return any other explanation.

User Request: {request}""",
            input_variables=["data_repo", "request"],
        )

        if len(self.dataframe_annotations) == 0:
            data_repo_description = "The Data Repository is empty."
        else:
            data_repo_description = f"""
The Data Repository contains the following dataframes:
   {self.__repr__()}
The meaning of each dataframe is defined by its title."""
        request_router = prompt | self.llm | StrOutputParser()
        result = request_router.invoke({"data_repo": data_repo_description, "request": description})
        return result

    def list_titles(self):
        return [adf.get_title_and_ref() for adf in self.dataframe_annotations]

    def __repr__(self):
        data_repo_string = ""
        for index, adf in enumerate(self.dataframe_annotations):
            data_repo_string = f"{data_repo_string}\n{str(adf)}\n"
        return data_repo_string
