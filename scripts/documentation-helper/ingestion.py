from dotenv import load_dotenv
import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    download_loader,
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.readers.file import UnstructuredReader
from pinecone import Pinecone

load_dotenv()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

if __name__ == "__main__":
    print("Going to ingest pinecone docs...")

    # path_docs = "./scripts/documentation-helper/llamaindex-docs-sample"  # only for debug purposes
    path_docs = "./llamaindex-docs"
    parser = UnstructuredReader()
    file_extractor = {".html": parser}
    documents = SimpleDirectoryReader(
        path_docs, file_extractor=file_extractor
    ).load_data()
    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, node_parser=node_parser
    )

    vector_store = PineconeVectorStore(pinecone_index=index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    vector_store_index = VectorStoreIndex.from_documents(
        documents=documents,
        service_context=service_context,
        storage_context=storage_context,
        show_progress=True,
    )

    print("Ingestion complete!")
