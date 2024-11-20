from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 入出力パスの設定
PDF_PATH = "./pdf/sample_rules.pdf"  # 入力PDFファイルのパス
PERSIST_DIR = "./chroma_db"      # ベクトルDBの保存先

# 埋め込みモデルの設定
MODEL_NAME = "jeffh/intfloat-multilingual-e5-large:f16"  # Ollamaの埋め込みモデル名
GPU_SERVER = "http://127.0.0.1:11434"

def process_pdf(pdf_path: str, persist_dir: str) -> None:
    """
    RAGのための文書処理を行います：
    1. PDFからテキストを抽出
    2. テキストをチャンクに分割
    3. チャンクをベクトル化して保存
    4. 処理したチャンクの内容を確認用ファイルに出力

    Args:
        pdf_path: 処理対象のPDFファイルパス
        persist_dir: ベクトルDBの保存先ディレクトリ
    """
    # 1. PDFからテキストを抽出
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. テキストをチャンクに分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,          # チャンクの最大サイズ（文字数）
        chunk_overlap=50,        # チャンク間のオーバーラップ（文字数）
        separators=[             # 分割位置の優先順位
            "\n\n",  # 段落
            "\n",    # 改行
            "。",    # 文末
            "、",    # 読点
            " ",     # スペース
            ""      # 文字単位
        ],
    )
    chunks = text_splitter.split_documents(documents)

    # 3. チャンクをベクトル化してChromaDBに保存
    embeddings = OllamaEmbeddings(base_url=GPU_SERVER, model=MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    # 埋め込みモデルのmultilingual-e5で必要なprefixを追加 (passage: )
    for chunk in chunks:
        chunk.page_content = "passage: " + chunk.page_content
    vectorstore.add_documents(chunks)

    # 4. チャンクの内容を確認用ファイルに出力
    output_file = Path(pdf_path).stem + "_chunks.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, 1):
            f.write(f"[Chunk {i}]\n{chunk.page_content}\n\n")

    # 処理結果の表示
    print(f"✓ PDF処理完了: {pdf_path}")
    print(f"  - 生成チャンク数: {len(chunks)}")
    print(f"  - チャンク確認用ファイル: {output_file}")
    print(f"  - ベクトルDB保存先: {persist_dir}")

if __name__ == "__main__":
    process_pdf(
        pdf_path=PDF_PATH,
        persist_dir=PERSIST_DIR
    )
