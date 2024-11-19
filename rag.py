from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Iterator

# モデルとパスの設定
EMBEDDING_MODEL = "jeffh/intfloat-multilingual-e5-large:f16"
GENERATION_MODEL = "schroneko/gemma-2-2b-jpn-it"
PERSIST_DIR = "./chroma_db"
GPU_SERVER = "http://127.0.0.1:11434"

# プロンプトテンプレート
RAG_TEMPLATE = """
以下のコンテキストを使用して、質問に日本語で答えてください。
コンテキストに含まれていない情報については、その旨を明確に伝えてください。

コンテキスト:
{context}

質問: {question}

回答:
"""

def format_docs(docs):
    """文書のコンテンツを結合"""
    return "\n---\n".join(doc.page_content for doc in docs)

def search_and_generate(query: str) -> Iterator[str]:
    """
    RAGロジックを実行します：
    1. ベクトルDBから関連文書を検索
    2. 検索結果を表示（スコア情報含む）
    3. プロンプトを生成
    4. 回答を生成

    Args:
        query: ユーザーからの質問

    Returns:
        str: 生成された回答
    """

    # 埋め込みモデルのmultilingual-e5で必要なprefixを追加 (query: )
    query = "query: " + query

    # ベクトルストアの準備
    embeddings = OllamaEmbeddings(base_url=GPU_SERVER, model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    # 文書検索の実行
    # 検索方式を選択（どちらかをコメントアウト）

    # ----------------------------------------------

    # 方式1: Similarity Search
    results = vectorstore.similarity_search_with_relevance_scores(
        query,
        k=4
    )
    docs = [doc for doc, _ in results]

    print("\n=== 検索結果（Similarity Search）===")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n[文書 {i}] 類似度スコア: {score:.4f}")
        print(f"内容: {doc.page_content}")
        print("-" * 80)

    # ----------------------------------------------

    # 方式2: MMR Search
    # retriever = vectorstore.as_retriever(
    #     search_type="mmr",
    #     search_kwargs={
    #         'k': 4,              # 取得する文書数
    #         'fetch_k': 10,       # 候補として検索する文書数
    #         'lambda_mult': 0.7   # 類似度(1.0)と多様性(0.0)のバランス
    #     }
    # )
    # docs = retriever.get_relevant_documents(query)

    # print("\n=== 検索結果（MMR Search）===")
    # for i, doc in enumerate(docs, 1):
    #     print(f"\n[文書 {i}]")
    #     print(f"内容: {doc.page_content}")
    #     print("-" * 80)

    # ----------------------------------------------

    # プロンプトの準備
    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    # プロンプトの表示用関数
    def print_prompt(inputs):
        formatted_prompt = prompt.format(**inputs)
        print("\n=== 生成プロンプト ===")
        print(formatted_prompt)
        print("-" * 80)
        print("\n回答: ", end='', flush=True)
        return formatted_prompt

    # RAGチェーンの実行
    chain = (
        RunnableParallel({
            "context": lambda _: format_docs(docs),
            "question": RunnablePassthrough()
        })
        | print_prompt
        | OllamaLLM(base_url=GPU_SERVER, model=GENERATION_MODEL)
        | StrOutputParser()
    )

    # ストリーミング実行
    return chain.stream(query)

def main():
    print("質問を入力してください（終了するには「quit」と入力）")

    while True:
        query = input("\n質問: ").strip()
        if query.lower() == 'quit':
            break

        # ストリーミング出力を逐次表示
        for chunk in search_and_generate(query):
            print(chunk, end='', flush=True)
        print("\n")
        print("-" * 80)

if __name__ == "__main__":
    main()
