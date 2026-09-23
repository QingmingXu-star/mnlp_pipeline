import re
from .setup import SUPPORTED_LANGS
from utils import (
    load_schema_configs)


def build_row_prompts(row, gendered_row, eval_lang, scaffolds, punc_map, language_test_configs):
    # 1. Prepare Sentence Variations (Gendered vs Neutral)

    if gendered_row:
        m_sent, f_sent = row['Masculine'], row['Feminine']
        condition = "G"
    else:
        m_sent, m_sent_noun, f_sent, f_sent_noun = wrap_neutral_sentence(row['Neutral'], eval_lang, scaffolds, punc_map)
        if eval_lang in language_test_configs["language_types"]["gendered_pronouns"] and language_test_configs["use_pronouns_if_available"] == "True":
            condition = "P"
            pass
        else:
            m_sent, f_sent = m_sent_noun, f_sent_noun
            condition = "N"

    model_inputs = {}

    masc_words = m_sent.split()
    fem_words = f_sent.split()
        
    fem_word = None
    masc_word = None
    
    for idx, word in enumerate(fem_words):
        if idx < len(masc_words) and word != masc_words[idx]:
            fem_word = word
            masc_word = masc_words[idx]
            break

    model_inputs["baseline"] = (m_sent, f_sent)

    if not hasattr(build_row_prompts, "_already_printed"):
        print(f"\nID: baseline\n{model_inputs['baseline']}\n")
        print("-" * 30)

    if not hasattr(build_row_prompts, "_already_printed"):
        print("="*50 + "\n")
        build_row_prompts._already_printed = True

    return model_inputs, condition, masc_word, fem_word


def wrap_neutral_sentence(sentence, eval_lang, SCAFFOLDS, PUNC_MAP):

    m_pron, f_pron = "he said", "she said"
    m_noun, f_noun = "the man said", "the woman said"
    if eval_lang != "English":
        m_pron, f_pron = SCAFFOLDS[m_pron][eval_lang], SCAFFOLDS[f_pron][eval_lang]
        m_noun, f_noun = SCAFFOLDS[m_noun][eval_lang], SCAFFOLDS[f_noun][eval_lang]

    sentence = re.sub(r'^[^\w\s]+|[^\w\s]+\Z', '', sentence.strip())
    p_start, p_end = PUNC_MAP[eval_lang]
    base = f"{p_start}{sentence}{p_end}"

    return (f"{base} {m_pron}", f"{base} {m_noun}", f"{base} {f_pron}", f"{base} {f_noun}")