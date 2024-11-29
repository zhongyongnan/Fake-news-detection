from tqdm import tqdm
from LLaMA import LLaMA
import pandas as pd
import re

llama = LLaMA("/data1/mwy/NLP/llama-2-7b-chat-hf_fnc")
stances = pd.read_csv("./fnc-1/competition_test_stances.csv")
bodies = pd.read_csv("./fnc-1/competition_test_bodies.csv")
data = pd.merge(stances, bodies)

def prompt(title, body):
    return f"""You are a sophisticated language model trained to detect the stance of a news body relative to its headline. Given a news title and its corresponding body text, classify the stance into one of the following categories:

    Agrees: The body text agrees with the headline.
    Disagrees: The body text disagrees with the headline.
    Discusses: The body text discusses the same topic as the headline but does not take a position.
    Unrelated: The body text discusses a different topic than the headline.
    Input:

    News Title: {title}

    News Body: {body}

    Task: Classify the stance of the body text relative to the claim made in the headline into one of the four categories: Agrees, Disagrees, Discusses, Unrelated.

    Output Format:

    Stance: {{Agrees/Disagrees/Discusses/Unrelated}}"""


def chop_classify_text(text):
    pattern = re.compile(r'agree|disagree|discuss|unrelated', re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group().lower()
    else:
        return "other type"

headline = []
bodyid = []
stance = []

for _, each in tqdm(data.iterrows(), total=len(data)):
    stance.append(chop_classify_text(llama.generate(prompt(each['Headline'], each['articleBody']))))
    headline.append(each['Headline'])
    bodyid.append(each['Body ID'])
    # print(headline, bodyid, stance)

pd.DataFrame({'Headline': headline, 'Body ID': bodyid, 'Stance': stance}).to_csv('./test_result.csv', index=False)