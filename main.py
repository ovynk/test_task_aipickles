import os
import uvicorn

from fastapi import FastAPI, Request
from langchain_community.llms import HuggingFaceHub
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from huggingface_hub.utils import HfHubHTTPError


HF_TOKEN = 'YOUR HUGGINGFACE TOKEN'
HF_MODEL = "facebook/bart-large-cnn"

os.environ['HUGGINGFACEHUB_API_TOKEN'] = HF_TOKEN

app = FastAPI()
llm = HuggingFaceHub(
    repo_id=HF_MODEL,
    task='summarization',
    model_kwargs={'max_length': 1024, 'max_new_tokens': 120}
)
chain = load_summarize_chain(llm, chain_type='map_reduce')


@app.post("/summarize")
async def summarize(request: Request):
    data = await request.json()
    text = data.get("text", "")

    # split big text into chunks, so that it can fit in model max length
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=900, chunk_overlap=30)
    texts = text_splitter.split_text(text)

    # convert the split chunks into document format (the chain needs it)
    docs = [Document(page_content=t) for t in texts]

    try:
        summary = await chain.ainvoke(docs)
    except HfHubHTTPError as e:
        return {
            "error_description": "LLM Error. Use english text or try reduce text's length.",
            "detailed": str(e.server_message)
        }

    return {"summary": summary['output_text']}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8888)
