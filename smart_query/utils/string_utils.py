
def strip_sparql_decoartion(data: str):
    if data.startswith("\"```sparql"):
        start_index = data.find("```sparql") + len("```sparql")
        end_index = data.find("```", start_index)
        sparql_query = data[start_index:end_index].strip()
    elif data.startswith("\"```code"):
        start_index = data.find("```code") + len("```code")
        end_index = data.find("```", start_index)
        sparql_query = data[start_index:end_index].strip()
    elif data.startswith("\"```"):
        start_index = data.find("```") + len("```")
        end_index = data.find("```", start_index)
        sparql_query = data[start_index:end_index].strip()
    elif data.startswith('"') and data.endswith('"'):
        # Remove leading and trailing double quotes
        sparql_query = data[1:-1]
    else:
        sparql_query = data
    return sparql_query