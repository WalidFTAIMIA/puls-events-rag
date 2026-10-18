from pathlib import Path

import pytest
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"


@pytest.fixture(scope="module")
def vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def test_vectorstore_files_exist():
    assert (VECTORSTORE_DIR / "index.faiss").is_file()
    assert (VECTORSTORE_DIR / "index.pkl").is_file()


def test_index_and_documents_have_same_size(vectorstore):
    assert vectorstore.index.ntotal > 0
    assert vectorstore.index.ntotal == len(vectorstore.index_to_docstore_id)


def test_semantic_search_returns_event_metadata(vectorstore):
    results = vectorstore.similarity_search(
        "Festival de musique à Paris",
        k=3,
    )

    assert 0 < len(results) <= 3

    for document in results:
        assert document.page_content
        assert document.metadata["uid"]
        assert document.metadata["title"]
        assert document.metadata["region"] == "Île-de-France"
