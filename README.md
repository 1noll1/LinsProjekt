# Place name classifier

This is a silly binary place name classifier, trained to distinguish a "tätort" from "småort".
Please see the excel_data folder for reference. This data was fetched from [Statistiska centralbyråns website](https://www.scb.se) in December 2019.

## Instructions

### The pipeline
The architecture works in the following way:
1. `SALDO_sammansattningar.py` runs SALDO compound analysis (which takes a while!) on all the place names contained in the xlsx-files in the `datasets`folder.
This script also scrapes Wikipedia for relevant place name suffixes for the compound analysis to be more accurate.
2. `load_fasttext.py` loads the pretrained FastText vectors, adds vectors for OOV words and saves them as `fasttext_vectors.pkl`.
3. Running `main.py` with no additional arguments trains the GRU on a character level.
Running `main_fasttext.py` with no additional arguments trains the GRU on the compounds instead. Running `main_fasttext.py`
with `--pretrained True` should load the pretrained fasttext vectors – in theory.
4. If you run `eval.py` without arguments, the `trained_model` trained in `main.py` will be evaluated by default. If you wish to evaluate the fastText model, please run `eval.py --modelfile fastText_trained_model --dataset fastText_dataset`.

*Nota bene: You will also need the fastText .bin and .vec files for Swedish which are too large to upload to this repo. You can find them here: https://fasttext.cc/docs/en/crawl-vectors.html*
