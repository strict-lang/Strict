# Strict.MachineLearning
Classic machine learning next to the neural networks of `NeuralNetworks/`, written in Strict. The data comes from `NeuralNetworks/iris.csv` or is downloaded on the first run, nothing big is committed.

- `IrisClusters.strict`: k-means (`KMeans`) groups the 150 Iris flowers into 3 clusters that match their species to 88.7%.
- `IrisClassifiers.strict`: k nearest neighbors (`NearestNeighbors`), Gaussian naive Bayes (`NaiveBayes`), a Gini decision tree (`DecisionTree`) and a random forest of 25 trees (`RandomForest`, bootstrap samples and a random square root of the features per split) all classify 29 of the 30 held out flowers right.
- `MarkovText.strict`: downloads Tiny Shakespeare (1.1 MB from karpathy/char-rnn on GitHub) on the first run and writes 400 characters with an order 5 Markov chain (`Markov`, the text itself is the model: the next character is drawn from all places where the last 5 characters occur). A failed download is an Error (`Evaluation.Downloaded`) and leaves no corpus behind, `python MachineLearning/check_download.py` (also with `native`) checks that.
