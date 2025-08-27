from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel

from smart_query.data_repo.data_repository import DataRepository


class Retriever(ABC):
    def __init__(self, name: str, llm: BaseChatModel):
        self.name = name
        self.llm = llm

    @abstractmethod
    def get_description(self):
        pass

    def get_name(self):
        return self.name

    def get_llm(self):
        return self.llm

    @abstractmethod
    def get_examples(self):
        pass


class TextRetriever(Retriever):
    @abstractmethod
    def get_text(self, atomic_request: str):
        pass


class DataFrameRetriever(Retriever):
    def __init__(self, name: str, llm: BaseChatModel, join_query_compatible: bool = False):
        super().__init__(name, llm)
        self.join_query_compatible = join_query_compatible

    def set_inner_join(self, inner_join):
        self.join_query_compatible = inner_join

    def get_inner_join(self):
        return self.join_query_compatible

    @abstractmethod
    def get_dataframe_annotation(self, data_repo: DataRepository, atomic_request: str):
        pass
