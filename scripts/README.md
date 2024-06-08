# Setting up enviroment
We will prepare a python enviroment with pipenv:

```powershell
> pip install pipenv
```

Checking it's version:
```powershell
> pipenv --version
pipenv, version 2023.12.1
```
Now we can use bash or cmd/powershell to enter in pipenv following theese commands:
```powershell
> pipenv shell
> pipenv install llama-index python-dotenv llama-index-readers-web bs4 openai pinecone-client llama-index-embeddings-openai black llama-index-readers-file unstructured streamlit
```
