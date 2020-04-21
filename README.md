### Cai dat anaconda https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

### Tao folder moi truong
conda create -p=./.env

### Them tep w2v vao folder '/Word-Similarity/word2vec'
#### Link: https://drive.google.com/open?id=1z1IDKaZuJXw5g7Yebr1GDq2UfvVR_aGx

### Cap nhat moi truong anaconda
conda env update -f environment.yml -p ./.env --prune

### Su dung moi truong
conda activate ./.env

### Chay test
python main/run.py

### Cac task va vi du
#### Test cosine similarity
python main/run.py --task=0 --word1=chị --word2=anh --k=10 

#### Test Cosine Similarity with Visim-400 dataset
python main/run.py --task=1 

#### Test K nearest words
python main/run.py --task=2 --word=anh --k=10 
python main/run.py --task=2 --word=chị --k=10 

#### Test Synonym-Antonym Classification
python main/run.py --task=3

#### Test Model Prediction
python main/run.py --task=4 --word1=anh --word2=kết_thúc
