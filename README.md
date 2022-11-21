# latin-bert-huggingface
This repo contains tokenizer config files to integrate [Latin BERT](https://github.com/dbamman/latin-bert) in 🤗 Transformers.

## Installation
The tokenizer config only works with the fast tokenizers from 🤗 Tokenizers.
```bash
pip install transformers tokenizers
```
To download the pytorch model and the model.config from Google docs, you can use the included script.
```bash
./download_latinbert.sh
```

## Usage
Warning: the tokenizer used by latin-bert is not identical with the WordPiece tokenizer used by standard BERT models.
In order to run the WordPiece algorithm with end-of-word suffixes, I had to implement most steps of the tokenization
via regular expressions in the pre-tokenizer. There might be some cases where the results differ from the original.

A feature that has not been implemented here is the splitting of enclitics (-que etc.). In accordance with the original paper and
code you might want to use [cltk.tokenizers.lat](https://docs.cltk.org/en/latest/cltk.tokenizers.lat.html) for that.

### Text infilling
```python
>>> from transformers import pipeline
>>> path_to_latin_bert = "./bert-base-latin-uncased/"
>>> unmasker = pipeline('fill-mask', model=path_to_latin_bert)
>>> unmasker("""dvces et reges carthaginiensivm hanno et mago qui [MASK] punico bello 
                cornelium consulem aput liparas ceperunt""")
[{'score': 0.4497990906238556,
  'token': 241,
  'token_str': 'primo',
  'sequence': 'dvces et reges carthaginiensivm hanno et mago qui primo punico bello cornelium consulem aput liparas ceperunt'},
 {'score': 0.4091891050338745,
  'token': 422,
  'token_str': 'secundo',
  'sequence': 'dvces et reges carthaginiensivm hanno et mago qui secundo punico bello cornelium consulem aput liparas ceperunt'},
 {'score': 0.0678754523396492,
  'token': 674,
  'token_str': 'tertio',
  'sequence': 'dvces et reges carthaginiensivm hanno et mago qui tertio punico bello cornelium consulem aput liparas ceperunt'},
 {'score': 0.02497234009206295,
  'token': 1651,
  'token_str': 'altero',
  'sequence': 'dvces et reges carthaginiensivm hanno et mago qui altero punico bello cornelium consulem aput liparas ceperunt'},
 {'score': 0.0188859011977911,
  'token': 4565,
  'token_str': 'priore',
  'sequence': 'dvces et reges carthaginiensivm hanno et mago qui priore punico bello cornelium consulem aput liparas ceperunt'}]
```

### Generate word representations
This corresponds to the [minimal working example](https://github.com/dbamman/latin-bert/tree/master#minimal-example) from the original repo.
```python
from transformers import AutoModel, AutoTokenizer
path_to_latin_bert = "./bert-base-latin-uncased/"

tokenizer = AutoTokenizer.from_pretrained(path_to_latin_bert)
model = AutoModel.from_pretrained(path_to_latin_bert)

# examples from scripts/gen_berts.py in original repo
sents = ["arma virum -que cano", "arma gravi numero violenta -que bella parabam"]
sents = [s.split() for s in sents]  # enclitics and word splitting...

# tokenize and run model
inputs = tokenizer(sents, return_tensors="pt", is_split_into_words=True, padding=True)
outputs = model(**inputs)

# average over subtokens and print results
for batch_ix, (sent, out) in enumerate(zip(sents, outputs.last_hidden_state)):
    for n, word in enumerate(sent):
        token_span = inputs.word_to_tokens(batch_ix, n)
        embedding = out[slice(*token_span)].mean(axis=0)
        print(word, ' '.join(f"{x:.5f}" for x in embedding), sep="\t")
    print()
```
