import pandas as pd
import os
from dotenv import load_dotenv
import sys
import llama_index
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,

)
from IPython.display import Markdown, display
import nest_asyncio
import openai
import pandas as pd

nest_asyncio.apply()

# .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数から必要な情報を取得
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region = os.getenv("AWS_DEFAULT_REGION", None)
openai_api_key = os.getenv("OPENAI_API_KEY")

# クエリデータとドキュメント関連度のデータを読み込む
testset_path = "localgovfaq/testset.txt"
testset_df = pd.read_csv(testset_path, sep='\t', header=None).fillna("None")
testset_df.columns = ['Query', 'A_ID', 'B_ID', 'C_ID']
testset_df.head()

#、QAペアのIDとQAの回答が含まれています
answers_path = 'localgovfaq/qas/answers_in_Amagasaki.txt'
answers_df = pd.read_csv(answers_path, sep='\t', header=None, names=['ID', 'Answer'])
answers = answers_df["Answer"].values.tolist()
answers_df.head()

# 指定されたフォーマットの辞書に変換
# qrelsとは(Quality Relevance）は、情報検索システムの評価
level_to_int = {'A': 3, 'B': 2, 'C': 1}
qrels_dict = {}
for index, row in testset_df.iterrows():
    sub_dict = {}
    for level in level_to_int.keys():
        id_column = f'{level}_ID'
        doc_ids = row[id_column].split(' ')
        level_to_int = {'A': 3, 'B': 2, 'C': 1}
        sub_dict.update({f'd_{doc_id}': level_to_int[level] for doc_id in doc_ids if doc_id != 'None'})
    qrels_dict[f'q_{index+1}'] = sub_dict

from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
from ranx import Qrels, Run, evaluate

# Titanモデルのパラメータを設定します
model_kwargs_titan = {
        "stopSequences": [],  # 停止シーケンスを設定します
        "temperature":0.0,  # temperatureを設定します
        "topP":0.5  # topPを設定します
    }

# OpenAIモデルのパラメータを設定します
model_kwargs_openai = {
    "stopSequences": [],  # 停止シーケンスを設定します
    "temperature": 0.0,  # temperatureを設定します
    "topP": 0.5  # topPを設定します
}

# 各embeddingモデルを使って、nodeからインデックス作成
model_dict = {
    "titan_v1": BedrockEmbedding().from_credentials(aws_profile=None, model_name='amazon.titan-embed-text-v1'),
    "titan_v2": BedrockEmbedding().from_credentials(aws_profile=None, model_name='amazon.titan-embed-text-v2'),
    "openai_002": OpenAIEmbedding(model="text-embedding-ada-002", api_key=openai_api_key),
    "openai_003_small": OpenAIEmbedding(model="text-embedding-3-small", api_key=openai_api_key),
    "openai_003_large": OpenAIEmbedding(model="text-embedding-3-large", api_key=openai_api_key)
}

# indexのみ作成。 TextEmbeddingのインスタンスを作成するときに、indexを作成する
docs = [Document(text=answer, metadata={"id": str(i)}) for i, answer in enumerate(answers_df['Answer'].values.tolist())]

# クエリを取得して検索を行い、結果を辞書に格納する関数
def run_search_and_store_results(testset_df, index, k):
    run_dict = {}
    for i, query in enumerate(testset_df["Query"].values.tolist(), start=1):
        # 検索実行
        base_retriever = index.as_retriever(similarity_top_k=k)
        docs = base_retriever.retrieve(query)  # llama-index retrieveメソッドを使用
        
        # 結果を辞書に格納
        run_dict[f'q_{i}'] = {}
        for j, doc in enumerate(docs):
            doc_id = doc.metadata["id"]
            score = len(docs) - j  # スコアはドキュメントの位置に基づく
            run_dict[f'q_{i}'] = run_dict[f'q_{i}'] | {f"d_{doc_id}": score}
    
    return run_dict

#kはトップ検索結果の順位
top_k_values = [3, 5, 7, 10]

for model_name, embed_model in model_dict.items():
    # 選択したモデルを使用してインデックスを作成
    index = VectorStoreIndex(docs, embed_model=embed_model)
    for k in top_k_values:
        run_dict = run_search_and_store_results(testset_df, index, k)
        qrels = Qrels(qrels_dict)
        run_large = Run(run_dict)
        print(f"Model: {model_name}, Top K: {k}")
        print(evaluate(qrels, run_large, [f"hit_rate@{k}", f"mrr@{k}", f"ndcg@{k}"]))
