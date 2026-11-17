# Final Project: Who Does the Classifier Misread? Auditing Annotation Ambiguity in GoEmotions
Piper Altman and Helen Cui

We train TF-IDF and MiniLM-embedding emotion classifiers on GoEmotions and ask how much of their error is explained by annotator disagreement, and which emotions and subreddits they misread most. See the final report for results.

**Note: Files within this repository were initially developed on Google CoLab then exported to local machines before being uploaded to Github, the commit history and how to run the files in this repo reflect that**

# Files
## Repository contents
| File | Purpose |
|---|---|
| CS74_Dataset_Pipeline.ipynb | Preprocessing, feature construction, training, validation tuning and test predictions for all model variants [confirm earlier notebook names] |
| CS74_Final_Evaluation_Error.ipynb | Ambiguity measures, error analysis, JSD, aggregation sensitivity, subgroup analysis, qualitative-review sampling |
| eval_figs | Figures produced by both notebooks used in the final report |
| eval_tables | Other results produced by both notebooks used in the final report |
| qualitative_review_sample_completed.csv | The 40 sampled errors with both reviewers' labels and the resolved label |

## Environment
- Google Colab, Python 3.13, scikit-learn 1.6.1, datasets 5.1.0, plus pandas, numpy, scipy, matplotlib, pyarrow and torch [confirm versions]
- The pipeline notebook installs: langchain-core, langchain-huggingface, huggingface_hub, sentence-transformers, datasets>=2.19, pyarrow
- A Hugging Face token is optional. Without one, downloads work but are rate-limited.
- A GPU is recommended for the embeddings (about 1-2 minutes, against about 11 minutes on CPU).

## Data
- Simplified GoEmotions (train/validation/test, 43,410 / 5,426 / 5,427 comments), used for training and gold labels.
- Raw GoEmotions (one row per comment and rater), loaded from Hugging Face inside 04_evaluation.
  It is used only to compute entropy, agreement, subreddit and the alternative gold definitions.

## How to run
Both notebooks should be run in Google CoLab for best results, as they contain code to have /outputs be mounted within Google Drive.

1. Run CS74_Dataset_Pipeline.ipynb. It saves predictions_<tag>.parquet, thresholds_<tag>.npy and
   model_summary.csv to `/content/drive/MyDrive/outputs` (change the path in the setup cell if needed).
2. Run CS74_Final_Evaluation_Error.ipynb from top to bottom. It reads those files, downloads the raw data,
   and writes figures to `eval_figs/` and tables to `eval_tables/` in the same folder.
3. Section 8 of CS74_Final_Evaluation_Error writes qualitative_review_sample.csv once and never overwrites it.
   Two people, label_piper and label_helen in this case, label it independently, and the completed file is read by the next cell.
   The completed sheet is included in this repo.

## Outputs and the report
In the output folder: predictions_<tag>.parquet (gold, probability and prediction for 28 labels, all splits),
thresholds_<tag>.npy, emb_all-MiniLM-L6-v2_<split>.npy, model_summary.csv, and
predictions_d_embed_OOF.parquet 
CS74_Final_Evaluation_Error writes eval_figs/ and eval_tables/. File prefixes match the report's figure and table numbers.

## Reproducibility notes
- Seed 42 and 1,000 bootstrap resamples (set in the CS74_Final_Evaluation_Error setup cell).
- The two audited models are chosen by validation macro-F1 unless TFIDF_TAG and EMB_TAG are set.
  The report uses c_word_char_hand and e_word_plus_embed.
- The evaluation reads saved predictions and does not retrain.
- Model training seeds and library versions: [confirm from 03_model].
- Bootstrap intervals and the qualitative sample depend on the seed, so changing it changes them slightly.

## Authors and AI use
See the AI-use statement in the report.
