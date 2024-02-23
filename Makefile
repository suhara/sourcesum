all: outputs/xsum_reference__report.csv \
outputs/xsum_pegasus__report.csv \
outputs/cnn_reference__report.csv \
outputs/cnn_pegasus__report.csv \
outputs/xsum_reference__corr.pdf \
outputs/xsum_pegasus__corr.pdf \
outputs/cnn_reference__corr.pdf \
outputs/cnn_pegasus__corr.pdf \
outputs/dataset_stats.csv \
outputs/xsum_sourcesentsumm__1__table.csv \
outputs/xsum_sourcesentsumm__2__table.csv \
outputs/cnn_sourcesentsumm__1__table.csv \
outputs/cnn_sourcesentsumm__2__table.csv \
outputs/xsum_reference__examples.csv \
outputs/xsum_pegasus__examples.csv \
outputs/cnn_reference__examples.csv \
outputs/cnn_pegasus__examples.csv \
outputs/main_table.tex


# Examples analysis

outputs/xsum_reference__examples.csv: outputs/xsum_reference__rouge1.json outputs/xsum_reference__bertscore.json outputs/xsum_reference__lexrank.json outputs/xsum_reference__perplexity.json
	python scripts/analyze_examples.py --input xsum_reference

outputs/xsum_pegasus__examples.csv: outputs/xsum_pegasus__rouge1.json outputs/xsum_pegasus__bertscore.json outputs/xsum_pegasus__lexrank.json outputs/xsum_pegasus__perplexity.json outputs/xsum_pegasus__attention.json
	python scripts/analyze_examples.py --input xsum_pegasus

outputs/cnn_reference__examples.csv: outputs/cnn_reference__rouge1.json outputs/cnn_reference__bertscore.json outputs/cnn_reference__lexrank.json outputs/cnn_reference__perplexity.json
	python scripts/analyze_examples.py --input cnn_reference

outputs/cnn_pegasus__examples.csv: outputs/cnn_pegasus__rouge1.json outputs/cnn_pegasus__bertscore.json outputs/cnn_pegasus__lexrank.json outputs/cnn_pegasus__perplexity.json outputs/cnn_pegasus__attention.json
	python scripts/analyze_examples.py --input cnn_pegasus


# Source sentence summarization experiments

## Aggregate

outputs/xsum_sourcesentsumm__1__table.csv: outputs/xsum_sourcesentsumm__lexrank__1__pred.csv outputs/xsum_sourcesentsumm__pegasus__1__pred.csv outputs/xsum_sourcesentsumm__bart__1__pred.csv
	python scripts/aggregate_sourcesent.py --dataset xsum --label_threshold 1

outputs/xsum_sourcesentsumm__2__table.csv: outputs/xsum_sourcesentsumm__lexrank__2__pred.csv outputs/xsum_sourcesentsumm__pegasus__2__pred.csv outputs/xsum_sourcesentsumm__bart__2__pred.csv
	python scripts/aggregate_sourcesent.py --dataset xsum --label_threshold 2

outputs/cnn_sourcesentsumm__1__table.csv: outputs/cnn_sourcesentsumm__lexrank__1__pred.csv outputs/cnn_sourcesentsumm__pegasus__1__pred.csv outputs/cnn_sourcesentsumm__bart__1__pred.csv
	python scripts/aggregate_sourcesent.py --dataset cnn --label_threshold 1

outputs/cnn_sourcesentsumm__2__table.csv: outputs/cnn_sourcesentsumm__lexrank__2__pred.csv outputs/cnn_sourcesentsumm__pegasus__2__pred.csv outputs/cnn_sourcesentsumm__bart__2__pred.csv
	python scripts/aggregate_sourcesent.py --dataset cnn --label_threshold 2


## Xsum

outputs/xsum_sourcesentsumm__lexrank__1__pred.csv: data/xsum_reference.json
	python scripts/summarize_sourcesent_xsum.py --method lexrank --label_threshold 1

outputs/xsum_sourcesentsumm__pegasus__1__pred.csv: data/xsum_reference.json
	python scripts/summarize_sourcesent_xsum.py --method pegasus --label_threshold 1

outputs/xsum_sourcesentsumm__bart__1__pred.csv: data/xsum_reference.json
	python scripts/summarize_sourcesent_xsum.py --method bart --label_threshold 1

outputs/xsum_sourcesentsumm__lexrank__2__pred.csv: data/xsum_reference.json
	python scripts/summarize_sourcesent_xsum.py --method lexrank --label_threshold 2

outputs/xsum_sourcesentsumm__pegasus__2__pred.csv: data/xsum_reference.json
	python scripts/summarize_sourcesent_xsum.py --method pegasus --label_threshold 2

outputs/xsum_sourcesentsumm__bart__2__pred.csv: data/xsum_reference.json
	python scripts/summarize_sourcesent_xsum.py --method bart --label_threshold 2

## CNN

outputs/cnn_sourcesentsumm__lexrank__1__pred.csv: data/cnn_reference.json
	python scripts/summarize_sourcesent_cnn.py --method lexrank --label_threshold 1

outputs/cnn_sourcesentsumm__pegasus__1__pred.csv: data/cnn_reference.json
	python scripts/summarize_sourcesent_cnn.py --method pegasus --label_threshold 1

outputs/cnn_sourcesentsumm__bart__1__pred.csv: data/cnn_reference.json
	python scripts/summarize_sourcesent_cnn.py --method bart --label_threshold 1

outputs/cnn_sourcesentsumm__lexrank__2__pred.csv: data/cnn_reference.json
	python scripts/summarize_sourcesent_cnn.py --method lexrank --label_threshold 2

outputs/cnn_sourcesentsumm__pegasus__2__pred.csv: data/cnn_reference.json
	python scripts/summarize_sourcesent_cnn.py --method pegasus --label_threshold 2

outputs/cnn_sourcesentsumm__bart__2__pred.csv: data/cnn_reference.json
	python scripts/summarize_sourcesent_cnn.py --method bart --label_threshold 2



# Dataset stats
outputs/dataset_stats.csv: data/cnn_pegasus.json data/cnn_reference.json data/xsum_pegasus.json data/xsum_reference.json
	python scripts/calculate_dataset_stats.py


# Correlation

outputs/xsum_reference__corr.pdf: outputs/xsum_reference__rouge1.json outputs/xsum_reference__bertscore.json outputs/xsum_reference__lexrank.json outputs/xsum_reference__perplexity.json outputs/xsum_reference__simcse.json outputs/xsum_reference__gptpmi.json outputs/xsum_reference__text-davinci-003.json
	python scripts/calculate_correlation.py --input xsum_reference

outputs/xsum_pegasus__corr.pdf: outputs/xsum_pegasus__rouge1.json outputs/xsum_pegasus__bertscore.json outputs/xsum_pegasus__lexrank.json outputs/xsum_pegasus__perplexity.json outputs/xsum_pegasus__attention.json outputs/xsum_pegasus__simcse.json outputs/xsum_pegasus__gptpmi.json outputs/xsum_pegasus__text-davinci-003.json
	python scripts/calculate_correlation.py --input xsum_pegasus

outputs/cnn_reference__corr.pdf: outputs/cnn_reference__rouge1.json outputs/cnn_reference__bertscore.json outputs/cnn_reference__lexrank.json outputs/cnn_reference__perplexity.json outputs/cnn_reference__simcse.json outputs/cnn_reference__gptpmi.json outputs/cnn_reference__text-davinci-003.json
	python scripts/calculate_correlation.py --input cnn_reference

outputs/cnn_pegasus__corr.pdf: outputs/cnn_pegasus__rouge1.json outputs/cnn_pegasus__bertscore.json outputs/cnn_pegasus__lexrank.json outputs/cnn_pegasus__perplexity.json outputs/cnn_pegasus__attention.json outputs/cnn_pegasus__simcse.json outputs/cnn_pegasus__gptpmi.json outputs/cnn_pegasus__text-davinci-003.json
	python scripts/calculate_correlation.py --input cnn_pegasus



# Report

outputs/main_table.tex: outputs/main_table.csv
	python scripts/csv2latex.py --num_header 2 --input outputs/main_table.csv > outputs/main_table.tex

outputs/main_table.csv: outputs/xsum_reference__report.csv outputs/xsum_pegasus__report.csv outputs/cnn_reference__report.csv outputs/cnn_pegasus__report.csv
	python scripts/create_table.py --output outputs/main_table.csv

outputs/xsum_reference__report.csv: outputs/xsum_reference__rouge1__eval.csv outputs/xsum_reference__bertscore__eval.csv outputs/xsum_reference__lexrank__eval.csv outputs/xsum_reference__perplexity__eval.csv outputs/xsum_reference__simcse__eval.csv outputs/xsum_reference__gptpmi__eval.csv outputs/xsum_reference__text-davinci-003__eval.csv
	python scripts/create_report.py --input xsum_reference

outputs/xsum_pegasus__report.csv: outputs/xsum_pegasus__rouge1__eval.csv outputs/xsum_pegasus__bertscore__eval.csv outputs/xsum_pegasus__lexrank__eval.csv outputs/xsum_pegasus__perplexity__eval.csv outputs/xsum_pegasus__attention__eval.csv outputs/xsum_pegasus__simcse__eval.csv outputs/xsum_pegasus__gptpmi__eval.csv outputs/xsum_pegasus__text-davinci-003__eval.csv
	python scripts/create_report.py --input xsum_pegasus

outputs/cnn_reference__report.csv: outputs/cnn_reference__rouge1__eval.csv outputs/cnn_reference__bertscore__eval.csv outputs/cnn_reference__lexrank__eval.csv outputs/cnn_reference__perplexity__eval.csv outputs/cnn_reference__simcse__eval.csv outputs/cnn_reference__gptpmi__eval.csv outputs/cnn_reference__text-davinci-003__eval.csv
	python scripts/create_report.py --input cnn_reference

outputs/cnn_pegasus__report.csv: outputs/cnn_pegasus__rouge1__eval.csv outputs/cnn_pegasus__bertscore__eval.csv outputs/cnn_pegasus__lexrank__eval.csv outputs/cnn_pegasus__perplexity__eval.csv outputs/cnn_pegasus__attention__eval.csv outputs/cnn_pegasus__simcse__eval.csv outputs/cnn_pegasus__gptpmi__eval.csv outputs/cnn_pegasus__text-davinci-003__eval.csv
	python scripts/create_report.py --input cnn_pegasus



# Evaluation

## XSum

### Reference

outputs/xsum_reference__rouge1__eval.csv: outputs/xsum_reference__rouge1.json
	python scripts/evaluate.py --input outputs/xsum_reference__rouge1.json --method rouge1

outputs/xsum_reference__bertscore__eval.csv: outputs/xsum_reference__bertscore.json
	python scripts/evaluate.py --input outputs/xsum_reference__bertscore.json --method bertscore

outputs/xsum_reference__lexrank__eval.csv: outputs/xsum_reference__lexrank.json
	python scripts/evaluate.py --input outputs/xsum_reference__lexrank.json --method lexrank

outputs/xsum_reference__perplexity__eval.csv: outputs/xsum_reference__perplexity.json
	python scripts/evaluate.py --input outputs/xsum_reference__perplexity.json --method perplexity

outputs/xsum_reference__simcse__eval.csv: outputs/xsum_reference__simcse.json
	python scripts/evaluate.py --input outputs/xsum_reference__simcse.json --method simcse

outputs/xsum_reference__gptpmi__eval.csv: outputs/xsum_reference__gptpmi.json
	python scripts/evaluate.py --input outputs/xsum_reference__gptpmi.json --method gptpmi

outputs/xsum_reference__text-davinci-003__eval.csv: outputs/xsum_reference__text-davinci-003.json
	python scripts/evaluate.py --input outputs/xsum_reference__text-davinci-003.json --method text-davinci-003


### PEGASUS

outputs/xsum_pegasus__rouge1__eval.csv: outputs/xsum_pegasus__rouge1.json
	python scripts/evaluate.py --input outputs/xsum_pegasus__rouge1.json --method rouge1

outputs/xsum_pegasus__bertscore__eval.csv: outputs/xsum_pegasus__bertscore.json
	python scripts/evaluate.py --input outputs/xsum_pegasus__bertscore.json --method bertscore

outputs/xsum_pegasus__lexrank__eval.csv: outputs/xsum_pegasus__lexrank.json
	python scripts/evaluate.py --input outputs/xsum_pegasus__lexrank.json --method lexrank

outputs/xsum_pegasus__perplexity__eval.csv: outputs/xsum_pegasus__perplexity.json
	python scripts/evaluate.py --input outputs/xsum_pegasus__perplexity.json --method perplexity

outputs/xsum_pegasus__attention__eval.csv: outputs/xsum_pegasus__attention.json
	python scripts/evaluate.py --input outputs/xsum_pegasus__attention.json --method attention

outputs/xsum_pegasus__simcse__eval.csv: outputs/xsum_pegasus__simcse.json
	python scripts/evaluate.py --input outputs/xsum_pegasus__simcse.json --method simcse

outputs/xsum_pegasus__gptpmi__eval.csv: outputs/xsum_pegasus__gptpmi.json
	python scripts/evaluate.py --input outputs/xsum_pegasus__gptpmi.json --method gptpmi

outputs/xsum_pegasus__text-davinci-003__eval.csv: outputs/xsum_pegasus__text-davinci-003.json
	python scripts/evaluate.py --input outputs/xsum_pegasus__text-davinci-003.json --method text-davinci-003


## CNN/DM

### Reference

outputs/cnn_reference__rouge1__eval.csv: outputs/cnn_reference__rouge1.json
	python scripts/evaluate.py --input outputs/cnn_reference__rouge1.json --method rouge1

outputs/cnn_reference__bertscore__eval.csv: outputs/cnn_reference__bertscore.json
	python scripts/evaluate.py --input outputs/cnn_reference__bertscore.json --method bertscore

outputs/cnn_reference__lexrank__eval.csv: outputs/cnn_reference__lexrank.json
	python scripts/evaluate.py --input outputs/cnn_reference__lexrank.json --method lexrank

outputs/cnn_reference__perplexity__eval.csv: outputs/cnn_reference__perplexity.json
	python scripts/evaluate.py --input outputs/cnn_reference__perplexity.json --method perplexity

outputs/cnn_reference__simcse__eval.csv: outputs/cnn_reference__simcse.json
	python scripts/evaluate.py --input outputs/cnn_reference__simcse.json --method simcse

outputs/cnn_reference__gptpmi__eval.csv: outputs/cnn_reference__gptpmi.json
	python scripts/evaluate.py --input outputs/cnn_reference__gptpmi.json --method gptpmi

outputs/cnn_reference__text-davinci-003__eval.csv: outputs/cnn_reference__text-davinci-003.json
	python scripts/evaluate.py --input outputs/cnn_reference__text-davinci-003.json --method text-davinci-003


### PEGASUS

outputs/cnn_pegasus__rouge1__eval.csv: outputs/cnn_pegasus__rouge1.json
	python scripts/evaluate.py --input outputs/cnn_pegasus__rouge1.json --method rouge1

outputs/cnn_pegasus__bertscore__eval.csv: outputs/cnn_pegasus__bertscore.json
	python scripts/evaluate.py --input outputs/cnn_pegasus__bertscore.json --method bertscore

outputs/cnn_pegasus__lexrank__eval.csv: outputs/cnn_pegasus__lexrank.json
	python scripts/evaluate.py --input outputs/cnn_pegasus__lexrank.json --method lexrank

outputs/cnn_pegasus__perplexity__eval.csv: outputs/cnn_pegasus__perplexity.json
	python scripts/evaluate.py --input outputs/cnn_pegasus__perplexity.json --method perplexity

outputs/cnn_pegasus__attention__eval.csv: outputs/cnn_pegasus__attention.json
	python scripts/evaluate.py --input outputs/cnn_pegasus__attention.json --method attention

outputs/cnn_pegasus__simcse__eval.csv: outputs/cnn_pegasus__simcse.json
	python scripts/evaluate.py --input outputs/cnn_pegasus__simcse.json --method simcse

outputs/cnn_pegasus__gptpmi__eval.csv: outputs/cnn_pegasus__gptpmi.json
	python scripts/evaluate.py --input outputs/cnn_pegasus__gptpmi.json --method gptpmi

outputs/cnn_pegasus__text-davinci-003__eval.csv: outputs/cnn_pegasus__text-davinci-003.json
	python scripts/evaluate.py --input outputs/cnn_pegasus__text-davinci-003.json --method text-davinci-003


# Prediction

## XSUM

### Reference
outputs/xsum_reference__rouge1.json:
	python scripts/evaluate_xsum.py --input data/xsum_reference.json --method rouge1

outputs/xsum_reference__bertscore.json:
	python scripts/evaluate_xsum.py --input data/xsum_reference.json --method bertscore

outputs/xsum_reference__lexrank.json:
	python scripts/evaluate_xsum.py --input data/xsum_reference.json --method lexrank

outputs/xsum_reference__perplexity.json:
	python scripts/evaluate_xsum.py --input data/xsum_reference.json --method perplexity

outputs/xsum_reference__simcse.json:
	python scripts/evaluate_xsum.py --input data/xsum_reference.json --method simcse

outputs/xsum_reference__gptpmi.json:
	python scripts/evaluate_xsum.py --input data/xsum_reference.json --method gptpmi

outputs/xsum_reference__text-davinci-003.json:
	python scripts/evaluate_gpt_xsum.py --input data/xsum_reference.json --method text-davinci-003


### Pegasus
outputs/xsum_pegasus__rouge1.json:
	python scripts/evaluate_xsum.py --input data/xsum_pegasus.json --method rouge1

outputs/xsum_pegasus__bertscore.json:
	python scripts/evaluate_xsum.py --input data/xsum_pegasus.json --method bertscore

outputs/xsum_pegasus__lexrank.json:
	python scripts/evaluate_xsum.py --input data/xsum_pegasus.json --method lexrank

outputs/xsum_pegasus__perplexity.json:
	python scripts/evaluate_xsum.py --input data/xsum_pegasus.json --method perplexity

outputs/xsum_pegasus__attention.json:
	python scripts/evaluate_xsum.py --input data/xsum_pegasus.json --method attention

outputs/xsum_pegasus__simcse.json:
	python scripts/evaluate_xsum.py --input data/xsum_pegasus.json --method simcse

outputs/xsum_pegasus__gptpmi.json:
	python scripts/evaluate_xsum.py --input data/xsum_pegasus.json --method gptpmi

outputs/xsum_pegasus__text-davinci-003.json:
	python scripts/evaluate_gpt_xsum.py --input data/xsum_pegasus.json --method text-davinci-003


## CNN/DM

### Reference
outputs/cnn_reference__rouge1.json:
	python scripts/evaluate_cnn.py --input data/cnn_reference.json --method rouge1

outputs/cnn_reference__bertscore.json:
	python scripts/evaluate_cnn.py --input data/cnn_reference.json --method bertscore

outputs/cnn_reference__lexrank.json:
	python scripts/evaluate_cnn.py --input data/cnn_reference.json --method lexrank

outputs/cnn_reference__perplexity.json:
	python scripts/evaluate_cnn.py --input data/cnn_reference.json --method perplexity

outputs/cnn_reference__simcse.json:
	python scripts/evaluate_cnn.py --input data/cnn_reference.json --method simcse

outputs/cnn_reference__gptpmi.json:
	python scripts/evaluate_cnn.py --input data/cnn_reference.json --method gptpmi

outputs/cnn_reference__text-davinci-003.json:
	python scripts/evaluate_gpt_cnn.py --input data/cnn_reference.json --method text-davinci-003


### Pegasus
outputs/cnn_pegasus__rouge1.json:
	python scripts/evaluate_cnn.py --input data/cnn_pegasus.json --method rouge1

outputs/cnn_pegasus__bertscore.json:
	python scripts/evaluate_cnn.py --input data/cnn_pegasus.json --method bertscore

outputs/cnn_pegasus__lexrank.json:
	python scripts/evaluate_cnn.py --input data/cnn_pegasus.json --method lexrank

outputs/cnn_pegasus__perplexity.json:
	python scripts/evaluate_cnn.py --input data/cnn_pegasus.json --method perplexity

outputs/cnn_pegasus__attention.json:
	python scripts/evaluate_cnn.py --input data/cnn_pegasus.json --method attention

outputs/cnn_pegasus__simcse.json:
	python scripts/evaluate_cnn.py --input data/cnn_pegasus.json --method simcse

outputs/cnn_pegasus__gptpmi.json:
	python scripts/evaluate_cnn.py --input data/cnn_pegasus.json --method gptpmi

outputs/cnn_pegasus__text-davinci-003.json:
	python scripts/evaluate_gpt_cnn.py --input data/cnn_pegasus.json --method text-davinci-003
