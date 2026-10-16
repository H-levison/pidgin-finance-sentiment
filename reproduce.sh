#!/bin/bash
# Full pipeline, in order. Each stage writes into results/ so the report's
# numbers can be traced back to a command.
#
# Expects the virtual environment active, or PY pointed at its interpreter.
set -e
PY="${PY:-python}"

echo "== 1. download the AfriSenti pcm splits =="
$PY src/download_data.py

echo "== 2. audit and clean, write CSVs + data_stats =="
$PY src/prepare_data.py

echo "== 3. baselines (TF-IDF + logistic regression) =="
$PY src/baseline.py --features word --class-weight none
$PY src/baseline.py --features word --class-weight balanced
$PY src/baseline.py --features char --class-weight balanced
$PY src/baseline.py --features word --class-weight balanced --train train_full

echo "== 4. Word2Vec + BiLSTM =="
$PY src/bilstm.py --epochs 12 --class-weight
$PY src/bilstm.py --epochs 12 --class-weight --random-embeddings
$PY src/bilstm.py --epochs 12

echo "== 5. transformer comparison (one at a time: 6 GB GPU) =="
# Fixed 6-epoch budget with best-epoch selection on dev macro F1. At 3 epochs
# AfroXLM-R had already peaked while XLM-R was still improving, so 3 was not a
# fair shared budget. Early-converging models are unaffected by the larger one.
$PY src/finetune.py --epochs 6 --model Davlan/afro-xlmr-base --class-weight --save
$PY src/finetune.py --epochs 6 --model Davlan/afro-xlmr-base
$PY src/finetune.py --epochs 6 --model xlm-roberta-base --class-weight
$PY src/finetune.py --epochs 6 --model roberta-base --class-weight
$PY src/finetune.py --epochs 6 --model castorini/afriberta_base --class-weight
$PY src/finetune.py --epochs 6 --model Davlan/afro-xlmr-base --class-weight --train train_full

echo "== 5b. AfriBERTa (needs a local copy, see README) =="
# castorini/afriberta_base ships only pytorch_model.bin. transformers 5.x looks
# for safetensors, follows a pull-request revision, and caches a fragment that
# torch cannot read. Download the files directly and load from a local folder.
if [ -d "C:/ml/afriberta_base" ]; then
  $PY src/finetune.py --epochs 6 --model "C:/ml/afriberta_base" --class-weight --tag afriberta
else
  echo "  skipped: see the AfriBERTa note in README.md"
fi

echo "== 5c. leakage controls =="
$PY src/finetune.py --epochs 6 --model Davlan/afro-xlmr-base --class-weight --train train_full_sub --tag sub

echo "== 6. dataset figure =="
$PY src/data_figures.py

echo "== 7. final test-set evaluation of the model chosen on dev =="
# Test is scored once, after the dev comparison has picked the configuration.
$PY src/finetune.py --epochs 6 --model Davlan/afro-xlmr-base --class-weight --eval-test --save

echo "== 8. results table =="
$PY src/results_table.py

echo "== 9. error analysis =="
$PY src/error_analysis.py --predictions results/predictions_afro-xlmr-base-weighted_test.csv

echo "== 10. seed summary =="
$PY src/seed_summary.py --pattern seed4 --split test

echo
echo "Done. Figures in results/figures, tables in results/."
echo
echo "The finance and crypto test set is collected and labelled separately:"
echo "  python src/collect_posts.py --harvest-board 15 --max-pages 4 \\"
echo "         --min-words 15 --max-words 60    # gather candidates"
echo "  python src/make_annotation_sheet.py     # build the labelling sheet"
echo "  ... label by hand, following data/annotation/GUIDELINES.md ..."
echo "  python src/import_labels.py <exported labels dir>"
echo "  python src/build_finance_test.py        # sheet -> finance_test.csv"
echo "  python src/annotation_agreement.py      # Cohen kappa on the double-labelled subset"
echo "  python src/evaluate_finance.py --checkpoint afro-xlmr-base-weighted-seed42"
