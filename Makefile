DATASET := webtext
MODEL := small-117M
MODEL := small-117M-p0.9
# MODEL := medium-345M
# MODEL := large-762M
# MODEL := xl-1542M
LANGMODEL := ngram
REPRESENTATIONS := gpt2-xl
REPRESENTATIONS_TYPE := final
DATA_DIR := data
CHECKPOINT_DIR := checkpoints
BATCH_SIZE_REPRESENTATIONS = 4
BATCH_SIZE_SURPRISALS = 4
SEED := 0

MODELS_BASE := small-117M medium-345M large-762M xl-1542M

HUMAN_EVALS_FILE := $(DATA_DIR)/other/mauve-human-eval-anon.csv


RAW_DATA_DIR_BASE := $(DATA_DIR)/raw/
PROCESSED_DIR_BASE := $(DATA_DIR)/processed/

ifneq ($(filter $(MODEL),human),)
RAW_DATA_DIR := $(RAW_DATA_DIR_BASE)/$(DATASET)/
PROCESSED_DIR_MODEL := $(PROCESSED_DIR_BASE)/$(DATASET)/$(MODEL)
CHECKPOINT_DIR_MODEL := $(CHECKPOINT_DIR)/$(DATASET)/$(MODEL)
else
RAW_DATA_DIR := $(RAW_DATA_DIR_BASE)/$(DATASET)/seed_$(SEED)/
PROCESSED_DIR_MODEL := $(PROCESSED_DIR_BASE)/$(DATASET)/seed_$(SEED)/$(MODEL)
CHECKPOINT_DIR_MODEL := $(CHECKPOINT_DIR)/$(DATASET)/seed_$(SEED)/$(MODEL)
endif

# Raw data
RAW_DATA_FILE_TEST := $(RAW_DATA_DIR)/$(MODEL).test.jsonl
RAW_DATA_FILE_HUMAN_TEST := $(RAW_DATA_DIR)/human.test.jsonl

#AW Processed data file names
BARETEXT_FNAME := model.baretext.txt
TEXT_FNAME := model.text.pickle
BASE_INFO_FNAME := model.base.pickle
SURPRISALS_FNAME := model.surps.pickle
REPRESENTATIONS_FNAME := model.rep.pickle
CLUSTERS_FNAME := model.clusters.pickle
KMEANS_MODEL_FNAME := kmeans.faiss
PCA_MODEL_FNAME := pca.pickle

# Processed data test
PROCESSED_DIR_TEST := $(PROCESSED_DIR_MODEL)/test
BARETEXT_FILE_TEST := $(PROCESSED_DIR_TEST)/$(BARETEXT_FNAME)
TEXT_FILE_TEST := $(PROCESSED_DIR_TEST)/$(TEXT_FNAME)
BASE_INFO_FILE_TEST := $(PROCESSED_DIR_TEST)/$(BASE_INFO_FNAME)
SURPRISALS_FILE_TEST := $(PROCESSED_DIR_TEST)/$(SURPRISALS_FNAME)
REPRESENTATIONS_DIR_TEST := $(PROCESSED_DIR_TEST)/$(REPRESENTATIONS)/$(REPRESENTATIONS_TYPE)
REPRESENTATIONS_FILE_TEST := $(REPRESENTATIONS_DIR_TEST)/$(REPRESENTATIONS_FNAME)
CLUSTERS_FILE_TEST := $(REPRESENTATIONS_DIR_TEST)/$(CLUSTERS_FNAME)

# Mauve clustering models
KMEANS_MODEL_FILE := $(REPRESENTATIONS_DIR_TEST)/$(KMEANS_MODEL_FNAME)
PCA_MODEL_FILE := $(REPRESENTATIONS_DIR_TEST)/$(PCA_MODEL_FNAME)


# Human data file names
BARETEXT_FNAME_HUMAN := model.baretext.txt
TEXT_FNAME_HUMAN := model.text.pickle
BASE_INFO_FNAME_HUMAN := model.base.pickle
REPRESENTATIONS_FNAME_HUMAN := model.rep.pickle
CLUSTERS_FNAME_HUMAN := human.clusters.pickle

# Human data files test
PROCESSED_DIR_MODEL_HUMAN := $(PROCESSED_DIR_BASE)/$(DATASET)/human
PROCESSED_DIR_HUMAN_TEST := $(PROCESSED_DIR_MODEL_HUMAN)/test
REPRESENTATIONS_DIR_HUMAN_TEST := $(PROCESSED_DIR_HUMAN_TEST)/$(REPRESENTATIONS)/$(REPRESENTATIONS_TYPE)
BARETEXT_FILE_HUMAN_TEST := $(PROCESSED_DIR_HUMAN_TEST)/$(BARETEXT_FNAME_HUMAN)
TEXT_FILE_HUMAN_TEST := $(PROCESSED_DIR_HUMAN_TEST)/$(TEXT_FNAME_HUMAN)
BASE_INFO_FILE_HUMAN_TEST := $(PROCESSED_DIR_HUMAN_TEST)/$(BASE_INFO_FNAME_HUMAN)
REPRESENTATIONS_FILE_HUMAN_TEST := $(REPRESENTATIONS_DIR_HUMAN_TEST)/$(REPRESENTATIONS_FNAME_HUMAN)
CLUSTERS_FILE_HUMAN_TEST := $(REPRESENTATIONS_DIR_TEST)/$(CLUSTERS_FNAME_HUMAN)


# Trained Models
CHECKPOINT_DIR_LANGMODEL := $(CHECKPOINT_DIR_MODEL)/$(LANGMODEL)
NGRAM_MODEL_FILE_ARPA := $(CHECKPOINT_DIR_LANGMODEL)/ngram.model.arpa
NGRAM_MODEL_FILE := $(CHECKPOINT_DIR_LANGMODEL)/ngram.model.binary
LSTM_MODEL_FILE := $(CHECKPOINT_DIR_LANGMODEL)/lstm.model.torch
ifneq ($(filter $(LANGMODEL),ngram),)
LANGMODEL_FILE := $(NGRAM_MODEL_FILE)
else
LANGMODEL_FILE := $(LSTM_MODEL_FILE)
endif
LANGMODEL_SURPRISALS_FILE := $(CHECKPOINT_DIR_LANGMODEL)/langmodel.model.surps.pickle

CHECKPOINT_DIR_REPRESENTATIONS := $(CHECKPOINT_DIR_MODEL)/$(REPRESENTATIONS)/$(REPRESENTATIONS_TYPE)
CLUSTER_MODEL_FILE := $(CHECKPOINT_DIR_REPRESENTATIONS)/cluster.model.pickle
CLUSTER_SURPRISALS_FILE := $(CHECKPOINT_DIR_REPRESENTATIONS)/cluster.model.surps.pickle


CHECKPOINT_DIR_MODEL_HUMAN := $(CHECKPOINT_DIR)/$(DATASET)/human
CHECKPOINT_DIR_LANGMODEL_HUMAN := $(CHECKPOINT_DIR_MODEL_HUMAN)/$(LANGMODEL)
NGRAM_MODEL_FILE_HUMAN := $(CHECKPOINT_DIR_LANGMODEL_HUMAN)/ngram.model.binary
LSTM_MODEL_FILE_HUMAN := $(CHECKPOINT_DIR_LANGMODEL_HUMAN)/lstm.model.torch
ifneq ($(filter $(LANGMODEL),ngram),)
LANGMODEL_FILE_HUMAN := $(NGRAM_MODEL_FILE_HUMAN)
else
LANGMODEL_FILE_HUMAN := $(LSTM_MODEL_FILE_HUMAN)
endif
LANGMODEL_SURPRISALS_FILE_HUMAN := $(CHECKPOINT_DIR_LANGMODEL)/langmodel.human.surps.pickle
CLUSTER_MODEL_FILE_HUMAN := $(CHECKPOINT_DIR_REPRESENTATIONS)/cluster.human.pickle
CLUSTER_SURPRISALS_FILE_HUMAN := $(CHECKPOINT_DIR_REPRESENTATIONS)/cluster.human.surps.pickle

TEXT_EMPIRICAL_SCORES_FILE := $(CHECKPOINT_DIR_LANGMODEL)/empirical_scores.text.pickle
TEXT_JENSEN_SCORES_FILE := $(CHECKPOINT_DIR_LANGMODEL)/divergences_scores.text.pickle
CLUSTER_EMPIRICAL_SCORES_FILE := $(CHECKPOINT_DIR_REPRESENTATIONS)/empirical_scores.cluster.pickle
CLUSTER_JENSEN_SCORES_FILE := $(CHECKPOINT_DIR_REPRESENTATIONS)/divergences_scores.cluster.pickle

# Analysis result files
CHECKPOINT_DIR_FULL := $(CHECKPOINT_DIR_MODEL)/$(LANGMODEL)--$(REPRESENTATIONS)_$(REPRESENTATIONS_TYPE)
SURPRISALS_CORRELATIONS_FILE := $(CHECKPOINT_DIR_FULL)/surps.corr.tsv

# Results files
RESULTS_DIR := results
COMPILED_RESULTS := $(RESULTS_DIR)/compiled_$(DATASET).tsv
COMPILED_SURPRISAL_CORR := $(RESULTS_DIR)/surprisal_corr_$(DATASET).tsv
COMPILED_SCORES_CORR := $(RESULTS_DIR)/humanscores_corr_$(DATASET).tsv

ifneq ($(filter $(MODEL),human),)
all: get_text get_representations train_text

all_text: get_text train_text

all_clusters: get_representations

else
all: get_text get_representations get_clusters train_text train_clusters eval_text eval_clusters analyse

all_text: get_text train_text eval_text analyse_texts

all_clusters: get_representations get_clusters train_clusters eval_clusters analyse_clusters

ifneq ($(filter $(MODEL),$(MODELS_BASE)),)
analyse: $(SURPRISALS_CORRELATIONS_FILE) analyse_texts analyse_clusters
else
analyse: analyse_texts analyse_clusters
endif

analyse_clusters: $(CLUSTER_EMPIRICAL_SCORES_FILE) $(CLUSTER_JENSEN_SCORES_FILE)

analyse_texts: $(TEXT_EMPIRICAL_SCORES_FILE) $(TEXT_JENSEN_SCORES_FILE)

eval_clusters: $(CLUSTER_SURPRISALS_FILE) $(CLUSTER_SURPRISALS_FILE_HUMAN)

eval_text: $(LANGMODEL_SURPRISALS_FILE) $(LANGMODEL_SURPRISALS_FILE_HUMAN)

train_clusters: $(CLUSTER_MODEL_FILE) $(CLUSTER_MODEL_FILE_HUMAN)

get_clusters: $(KMEANS_MODEL_FILE) get_clusters_model get_clusters_human

get_clusters_model: $(CLUSTERS_FILE_TEST)

get_clusters_human: $(CLUSTERS_FILE_HUMAN_TEST)

endif

train_text: $(LANGMODEL_FILE)

get_representations: $(REPRESENTATIONS_FILE_TEST)

ifneq ($(filter $(MODEL),$(MODELS_BASE)),)
get_text: $(TEXT_FILE_TEST) $(SURPRISALS_FILE_TEST)
else
get_text: $(TEXT_FILE_TEST)
endif

get_results: $(COMPILED_RESULTS) $(COMPILED_SURPRISAL_CORR)

compare: $(COMPILED_RESULTS)
	python -u src/h03_analysis/compare_to_human_eval.py --results-fpath $(COMPILED_RESULTS) --correlations-fpath $(COMPILED_SCORES_CORR)

plot: $(COMPILED_SURPRISAL_CORR)
	python -u src/h04_paper/plot_scores_full.py --correlations-fpath $(COMPILED_SCORES_CORR) --results-fpath $(RESULTS_DIR)

$(COMPILED_SURPRISAL_CORR):
	mkdir -p $(RESULTS_DIR)
	python -u src/h03_analysis/compile_surprisals.py --checkpoints-dir $(CHECKPOINT_DIR) --dataset $(DATASET) --src-fname surps.corr.tsv --results-fpath $(COMPILED_SURPRISAL_CORR)


$(COMPILED_RESULTS):
	mkdir -p $(RESULTS_DIR)
	python -u src/h03_analysis/compile_results.py --checkpoints-dir $(CHECKPOINT_DIR) --dataset $(DATASET) --results-fpath $(COMPILED_RESULTS)

# Langmodel + Cluster results

$(CLUSTER_JENSEN_SCORES_FILE):
	python -u src/h03_analysis/get_divergences_empirical.py --data-type cluster --surprisals-model-fpath $(CLUSTER_SURPRISALS_FILE) \
		--surprisals-human-fpath $(CLUSTER_SURPRISALS_FILE_HUMAN) --results-fpath $(CLUSTER_JENSEN_SCORES_FILE)

$(TEXT_JENSEN_SCORES_FILE):
	python -u src/h03_analysis/get_divergences_empirical.py --data-type text --surprisals-model-fpath $(LANGMODEL_SURPRISALS_FILE) \
		--surprisals-human-fpath $(LANGMODEL_SURPRISALS_FILE_HUMAN) --results-fpath $(TEXT_JENSEN_SCORES_FILE)

$(CLUSTER_EMPIRICAL_SCORES_FILE):
	python -u src/h03_analysis/get_mauve_scores_empirical.py --data-type cluster --surprisals-model-fpath $(CLUSTER_SURPRISALS_FILE) \
		--surprisals-human-fpath $(CLUSTER_SURPRISALS_FILE_HUMAN) --results-fpath $(CLUSTER_EMPIRICAL_SCORES_FILE)

$(TEXT_EMPIRICAL_SCORES_FILE):
	python -u src/h03_analysis/get_mauve_scores_empirical.py --data-type text --surprisals-model-fpath $(LANGMODEL_SURPRISALS_FILE) \
		--surprisals-human-fpath $(LANGMODEL_SURPRISALS_FILE_HUMAN) --results-fpath $(TEXT_EMPIRICAL_SCORES_FILE)

$(SURPRISALS_CORRELATIONS_FILE):
	mkdir -p $(CHECKPOINT_DIR_FULL)
	python -u src/h03_analysis/get_surprisal_correlations.py --surprisals-orig-fpath $(SURPRISALS_FILE_TEST) \
		--surprisals-cluster-fpath $(CLUSTER_SURPRISALS_FILE) --surprisals-text-fpath $(LANGMODEL_SURPRISALS_FILE) \
		--results-fpath $(SURPRISALS_CORRELATIONS_FILE)

# Cluster results

$(CLUSTER_SURPRISALS_FILE):
	python -u src/h02_learn/eval.py --data-type clusters --model-type categorical --model-data-fpath $(CLUSTERS_FILE_TEST) --human-data-fpath $(CLUSTERS_FILE_HUMAN_TEST) --model-fpath $(CLUSTER_MODEL_FILE) --surprisals-fpath $(CLUSTER_SURPRISALS_FILE)

$(CLUSTER_MODEL_FILE):
	mkdir -p $(CHECKPOINT_DIR_REPRESENTATIONS)
	python -u src/h02_learn/learn.py --data-type clusters --data-fpath $(CLUSTERS_FILE_TEST) --model-fpath $(CLUSTER_MODEL_FILE)

$(CLUSTER_SURPRISALS_FILE_HUMAN):
	python -u src/h02_learn/eval.py --data-type clusters --model-type categorical --model-data-fpath $(CLUSTERS_FILE_TEST) --human-data-fpath $(CLUSTERS_FILE_HUMAN_TEST) --model-fpath $(CLUSTER_MODEL_FILE_HUMAN) --surprisals-fpath $(CLUSTER_SURPRISALS_FILE_HUMAN)

$(CLUSTER_MODEL_FILE_HUMAN):
	mkdir -p $(CHECKPOINT_DIR_REPRESENTATIONS)
	python -u src/h02_learn/learn.py --data-type clusters --data-fpath $(CLUSTERS_FILE_HUMAN_TEST) --model-fpath $(CLUSTER_MODEL_FILE_HUMAN)

# Language model results

$(LANGMODEL_SURPRISALS_FILE_HUMAN):
	python -u src/h02_learn/eval.py --data-type text --model-type $(LANGMODEL) --model-data-fpath $(TEXT_FILE_TEST) --human-data-fpath $(TEXT_FILE_HUMAN_TEST) --model-fpath $(LANGMODEL_FILE_HUMAN) --surprisals-fpath $(LANGMODEL_SURPRISALS_FILE_HUMAN)

$(LANGMODEL_SURPRISALS_FILE):
	python -u src/h02_learn/eval.py --data-type text --model-type $(LANGMODEL) --model-data-fpath $(TEXT_FILE_TEST) --human-data-fpath $(TEXT_FILE_HUMAN_TEST) --model-fpath $(LANGMODEL_FILE) --surprisals-fpath $(LANGMODEL_SURPRISALS_FILE)

# Train model

$(LSTM_MODEL_FILE):
	mkdir -p $(CHECKPOINT_DIR_LANGMODEL)
	python -u src/h02_learn/learn.py --data-type text --data-fpath $(TEXT_FILE_TEST) --model-fpath $(LSTM_MODEL_FILE)

$(NGRAM_MODEL_FILE): $(NGRAM_MODEL_FILE_ARPA)
	kenlm/build/bin/build_binary $(NGRAM_MODEL_FILE_ARPA) $(NGRAM_MODEL_FILE)

$(NGRAM_MODEL_FILE_ARPA):
	mkdir -p $(CHECKPOINT_DIR_LANGMODEL)
	mkdir -p ./.kenlm_temp
	kenlm/build/bin/lmplz -o 5 -T ./.kenlm_temp/ --skip_symbols < $(BARETEXT_FILE_TEST) > $(NGRAM_MODEL_FILE_ARPA)


# Get model text cluster labels
$(CLUSTERS_FILE_TEST):
	python src/h01_data/get_clusters.py \
		--representations-fpath $(REPRESENTATIONS_FILE_TEST) --clusters-fpath $(CLUSTERS_FILE_TEST) \
		--kmeans-fpath $(KMEANS_MODEL_FILE) --pca-fpath $(PCA_MODEL_FILE)

# Get human text cluster labels
$(CLUSTERS_FILE_HUMAN_TEST):
	python src/h01_data/get_clusters.py \
		--representations-fpath $(REPRESENTATIONS_FILE_HUMAN_TEST) --clusters-fpath $(CLUSTERS_FILE_HUMAN_TEST) \
		--kmeans-fpath $(KMEANS_MODEL_FILE) --pca-fpath $(PCA_MODEL_FILE)

# Get clustering model
$(KMEANS_MODEL_FILE):
	python src/h01_data/train_clusters.py \
		--p-representations-fpath $(REPRESENTATIONS_FILE_HUMAN_TEST) --q-representations-fpath $(REPRESENTATIONS_FILE_TEST) \
		--kmeans-fpath $(KMEANS_MODEL_FILE) --pca-fpath $(PCA_MODEL_FILE) --seed $(SEED)

# Preprocess representations data
$(REPRESENTATIONS_FILE_TEST):
	mkdir -p $(REPRESENTATIONS_DIR_TEST)
	python src/h01_data/get_representations.py --batch-size $(BATCH_SIZE_REPRESENTATIONS) --representations-fpath $(REPRESENTATIONS_FILE_TEST) \
		--text-fpath $(TEXT_FILE_TEST) --representations-model $(REPRESENTATIONS) --representations-type $(REPRESENTATIONS_TYPE)
	rm $(REPRESENTATIONS_FILE_TEST).temp

# Preprocess surprisals data
$(SURPRISALS_FILE_TEST):
	python src/h01_data/get_surprisals.py --batch-size $(BATCH_SIZE_SURPRISALS) --surprisals-fpath $(SURPRISALS_FILE_TEST) \
		--text-fpath $(TEXT_FILE_TEST) --model $(MODEL)

# Preprocess text data
$(TEXT_FILE_TEST):
	mkdir -p $(PROCESSED_DIR_TEST)
	python src/h01_data/get_text.py --text-fpath $(TEXT_FILE_TEST) --base-info-fpath $(BASE_INFO_FILE_TEST) \
		 --baretext-fpath $(BARETEXT_FILE_TEST) --raw-data-fpath $(RAW_DATA_FILE_TEST)
