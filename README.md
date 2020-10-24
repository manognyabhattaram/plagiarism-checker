# plagiarism-checker

Setup and compilation:
The script consists of a Jupyter notebook.
The training files should be moved into the "texts" folder, and the testing files should be moved into the "test" folder.
The script checks similarity for ".txt" files as of now.

Functionality:
The plagiarism checker does the following: <br/>
1. Loads files<br/>
2. Preprocessing (lowercase, removing punctuation, changing numbers to words, lemmatization, stemming)<br/>
3. Performs tf-idf vectorization on the contents of the files<br/>
4. Find cosine similarity between the resulting vectors<br/>
5. Ranks the training documents based on similarity to the testing set<br/>
6. Return the rank list with similarity percentage<br/>
7. Calculate precision and recall<br/>

Citation for the corpus:
The training and test sets used in this project are selected texxt files from the following corpus:
Clough, P. and Stevenson, M. Developing A Corpus of Plagiarised Short Answers, Language Resources and Evaluation: Special Issue on Plagiarism and Authorship Analysis, In Press
