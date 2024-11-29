from tqdm import tqdm
from LLaMA import LLaMA
import pandas as pd
from datasets import Dataset
import tiktoken

llama = LLaMA("/data/Logic/llama-2-7b-chat-hf")
stances = pd.read_csv("/data1/mwy/NLP/fnc-1/competition_test_stances.csv")
bodies = pd.read_csv("/data1/mwy/NLP/fnc-1/competition_test_bodies.csv")
data = pd.merge(stances, bodies)
dataset = Dataset.from_pandas(data)

def prompt(body):
    return {"prompt": f"""You are an advanced language model tasked with summarizing news articles. Your summary should be less than 300 words and capture the main points of the article. This summary will be used to determine the relationship between the article body and its headline, so it should be comprehensive yet concise. Follow these guidelines:
    1. Identify and include the primary subject of the article.
    2. Highlight key facts, events, and statements.
    3. Avoid minor details and tangential information.
    4. Maintain a neutral and objective tone.

    Article Body:{body}
    
    Summary:
"""}

summ = []
tokens = 0

# for _, each in tqdm(data.iterrows(), total=len(data)):
#     summ.append(llama.generate(prompt(each['articleBody'])))
    # encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    # tokens += len(encoding.encode(prompt(each['articleBody'])))

dataset = dataset.map(prompt, input_columns='articleBody')
# print(dataset[0])
summ += llama.batch_generate(dataset, 'prompt', 16)

# print(tokens)
pd.DataFrame({'summary': summ}).to_csv('./test_summary.csv')