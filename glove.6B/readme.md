# Using Stanford's Glove 6B with Gensim

You can download the Stanford's Glove vectors from here (822 MB): https://nlp.stanford.edu/projects/glove/. <br>
Once the zip file is downloaded and extracted, an easy way is to convert to word2vec format, the 100d format. <br>
This the one time process referred in `get_glove_model()` function in `extract_features.py` file. <br>
The process is inspired from: https://machinelearningmastery.com/develop-word-embeddings-python-gensim/.
