from smart_query.data_repo.data_repository import DataRepository
from smart_query.data_repo.dataframe_annotation import DataFrameAnnotation

# Example Usage
data_repo = DataRepository(None)

import pandas as pd
data1 = {'Name': ['Tom', 'Bob', 'Alice'], 'Math_Score': [4, 5, 3]}
df1 = pd.DataFrame(data1)
adf = DataFrameAnnotation(df1, title="Scores of Math 150")
data_repo.add_dataframe(adf)

data2 = {'Name': ['Tom', 'Bob', 'Alice'], 'Height': [176, 182, 168]}
df2 = pd.DataFrame(data2)
adf2 = DataFrameAnnotation(df2, title="Student Heights (cm)")
data_repo.add_dataframe(adf2)

print(str(data_repo))