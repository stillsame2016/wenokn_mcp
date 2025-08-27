import json
from datetime import datetime

import pandas as pd
import geopandas as gpd


class DataFrameAnnotation:
    def __init__(self, df, title, creator="User", column_descriptions=None, additional_metadata=None):
        if not isinstance(df, pd.DataFrame) and not isinstance(df, gpd.GeoDataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if not title:
            raise ValueError("Title must be a non-empty string")

        self.df = df
        self.title = title
        self.creator = creator
        self.created_at = datetime.now()
        self._column_descriptions = column_descriptions if column_descriptions else {}
        self._additional_metadata = additional_metadata if additional_metadata else {}

    def get_creator(self):
        return self.creator

    def set_creator(self, creator):
        if creator == 'User' or creator == 'System':
            self.creator = creator
        else:
            raise ValueError("creator must be 'User' or 'System'")

    def get_created_at(self):
        return self.creator

    def set_column_description(self, column_name, description):
        if column_name not in self.df.columns:
            raise ValueError(f"Column {column_name} does not exist in the DataFrame")
        self._column_descriptions[column_name] = description

    def get_column_description(self, column_name):
        return self._column_descriptions.get(column_name, None)

    def set_metadata(self, key, value):
        self._additional_metadata[key] = value

    def get_metadata(self, key):
        return self._additional_metadata.get(key, None)

    def first_three_rows_as_json(self):
        if isinstance(self.df, gpd.GeoDataFrame):
            gdf = self.df.head(3).drop(columns=['geometry'])
            df = pd.DataFrame(gdf)
            return df.to_json(orient='records')
        else:
            return self.df.head(3).to_json(orient='records')

    def get_title_and_ref(self):
        return f"DataFrame Title: {self.title}\nReference: {self.creator}\n"

    def __repr__(self):
        column_metadata_list = []
        for col, dtype in self.df.dtypes.items():
            column_metadata = {
                "column name": col,
                "column type": str(dtype),
            }
            column_description = self.get_column_description(col)
            if column_description:
                column_metadata["column description"] = column_description
            column_metadata_list.append(column_metadata)
        return f"""DataFrame Title: {self.title}
Creator: {self.creator}
Created At: {self.created_at}
Column Descriptions: {json.dumps(column_metadata_list)}
First Three Rows: {self.first_three_rows_as_json()}"""

