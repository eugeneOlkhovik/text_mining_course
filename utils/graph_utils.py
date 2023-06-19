import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')


def get_entities(sent):
    # chunk 1
    ent1 = ""
    ent2 = ""
    prv_tok_dep = ""
    prv_tok_text = ""
    prefix = ""
    modifier = ""
    for tok in nlp(sent):
        # chunk 2
        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound'
                # then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = f"{prv_tok_text} {tok.text}"

    # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
        # if the previous word was also a 'compound'
        # then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = f"{prv_tok_text} {tok.text}"

            # chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = f"{modifier} {prefix} {tok.text}"
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""
            # chunk 4
            if tok.dep_.find("obj"):
                ent2 = f"{modifier} {prefix} {tok.text}"

            # chunk 5
            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text

    return [ent1.strip(), ent2.strip()]


def get_relation(sent):
    doc = nlp(sent)
    matcher = Matcher(nlp.vocab)

    pattern = [[{'DEP': 'ROOT'},
        {'DEP': 'prep', 'OP': "?"},
        {'DEP': 'agent', 'OP': "?"},
        {'POS': 'ADJ', 'OP': "?"}]]
    matcher.add("matching_1", pattern, on_match=None)
    matches = matcher(doc)
    span = doc[matches[len(matches) - 1][1]:matches[len(matches) - 1][2]]
    return (span.text)
