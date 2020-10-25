# plagiarism-checker

<h2>Setup and compilation:</h2>
The script consists of a Jupyter notebook.<br/>
The file plag_check.py is the same script as the Jupyter notebook, This file was used to generate the documentation<br/>
The training files should be moved into the "texts" folder, and the testing files should be moved into the "test" folder.<br/>
The script checks similarity for ".txt" files as of now.<br/>
<br/>
<h2>Functionality:</h2>
<b>The plagiarism checker does the following:</b><br/>
1. Loads files<br/>
2. Preprocessing (lowercase, removing punctuation, changing numbers to words, lemmatization, stemming)<br/>
3. Performs tf-idf vectorization on the contents of the files<br/>
4. Find cosine similarity between the resulting vectors<br/>
5. Ranks the training documents based on similarity to the testing set<br/>
6. Return the rank list with similarity percentage<br/>
7. Calculate precision and recall<br/>
<br/>
<h2>Citation for the corpus:</h2>
<b>The training and test sets used in this project are selected text files from the following corpus:</b><br/>
Clough, P. and Stevenson, M. Developing A Corpus of Plagiarised Short Answers, Language Resources and Evaluation: Special Issue on Plagiarism and Authorship Analysis, In Press
