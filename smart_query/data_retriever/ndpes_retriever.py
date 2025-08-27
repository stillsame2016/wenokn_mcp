import json

import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from smart_query.data_retriever.base_retriever import TextRetriever


class NDPESRetriever(TextRetriever):

    def get_description(self):
        return f"""Text Retriever: "{self.name}"
This retriever can be used to get an answer for any question related to "National Pollution Discharge Elimination 
System (NPDES)" and "Kentucky Pollutant Discharge Elimination System (KPDES)".

The National Pollutant Discharge Elimination System (NPDES) is a regulatory program implemented by the United 
States Environmental Protection Agency (EPA) to control water pollution. It was established under the Clean 
Water Act (CWA) to address the discharge of pollutants into the waters of the United States.

The NPDES program requires permits for any point source that discharges pollutants into navigable waters, 
which include rivers, lakes, streams, coastal areas, and other bodies of water. Point sources are discrete 
conveyances such as pipes, ditches, or channels.

Under the NPDES program, permits are issued to regulate the quantity, quality, and timing of the pollutants 
discharged into water bodies. These permits include limits on the types and amounts of pollutants that can 
be discharged, monitoring and reporting requirements, and other conditions to ensure compliance with water 
quality standards and protect the environment and public health.

The goal of the NPDES program is to eliminate or minimize the discharge of pollutants into water bodies, 
thereby improving and maintaining water quality, protecting aquatic ecosystems, and safeguarding human health. 
It plays a critical role in preventing water pollution and maintaining the integrity of the nation's water resources."""

    def get_examples(self):
        return """
I want to propose a new type of wastewater discharge. what should I do?
What form should I use if I am creating a new facility for discharging processed wastewater?
I have an incident related to wastewater discharge. what should I do?
What wastewater discharge rules or regulations are relevant for flooding in Kentucky?
Could you please list some rules for wastewater discharge during flooding events in Kentucky?
"""

    def get_text(self, atomic_request: str):
        VDB_URL = "https://sparcal.sdsc.edu/api/v1/Utility/regulations"
        KPDES_URL = "https://sparcal.sdsc.edu/api/v1/Utility/kpdes"
        template = PromptTemplate(
            template="""You are the expert of National Pollution Discharge Elimination System (NPDES) and Kentucky 
Pollutant Discharge Elimination System (KPDES). 

The National Pollutant Discharge Elimination System (NPDES) is a regulatory program implemented by the United 
States Environmental Protection Agency (EPA) to control water pollution. It was established under the Clean 
Water Act (CWA) to address the discharge of pollutants into the waters of the United States.

The NPDES program requires permits for any point source that discharges pollutants into navigable waters, 
which include rivers, lakes, streams, coastal areas, and other bodies of water. Point sources are discrete 
conveyances such as pipes, ditches, or channels.

Under the NPDES program, permits are issued to regulate the quantity, quality, and timing of the pollutants 
discharged into water bodies. These permits include limits on the types and amounts of pollutants that can 
be discharged, monitoring and reporting requirements, and other conditions to ensure compliance with water 
quality standards and protect the environment and public health.

The goal of the NPDES program is to eliminate or minimize the discharge of pollutants into water bodies, 
thereby improving and maintaining water quality, protecting aquatic ecosystems, and safeguarding human health. 
It plays a critical role in preventing water pollution and maintaining the integrity of the nation's water 
resources.

Based on the provided context, use easy understanding language to answer the question clear and precise with 
references and explanations. If the local regulations (for example, KPDES for Kentucky Pollutant Discharge 
Elimination System) can be applied, please include the details of both NPDES rules and KPDES rules, and make 
clear indications of the sources of the rules.

If no information is provided in the context, return the result as "Sorry I dont know the answer", don't provide 
the wrong answer or a contradictory answer.

Context:{context}

Question:{question}?

Answer: """,
            input_variables=["question", "context"],
        )
        rag_chain = template | self.llm | StrOutputParser()

        if "kentucky" in atomic_request.lower() or "KPDES" in atomic_request:
            response = requests.get(f"{VDB_URL}?search_terms={atomic_request}")
            datasets = json.loads(response.text)
            datasets = datasets[0:4]
            context = "NPDES regulations: "
            context += "\n".join([dataset["description"] for dataset in datasets])

            response = requests.get(f"{KPDES_URL}?search_terms={atomic_request}")
            datasets = json.loads(response.text)
            datasets = datasets[0:4]
            context += "\nKPDES (Kentucky Pollutant Discharge Elimination System) regulations: "
            context += "\n".join([dataset["description"] for dataset in datasets])
        else:
            response = requests.get(f"{VDB_URL}?search_terms={atomic_request}")
            datasets = json.loads(response.text)
            datasets = datasets[0:5]
            context = "\n".join([dataset["description"] for dataset in datasets])

        result = rag_chain.invoke({"question": atomic_request, "context": context})
        return result
