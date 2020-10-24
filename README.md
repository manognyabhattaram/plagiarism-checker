# plagiarism-checker

The plagiarism checker does the following: (also consider this as our todos list)<br/>
1.loads files<br/>
2. preprocessing (lowercase, removing punctuation, changing numbers to words, lemmatization, stemming)</br/>
3. performs tf-idf vectorization on the contents of the files<br/>
4. Find cosine similarity between the resulting vectors<br/>
5. return similarity score, calculate precision and recall<br/>

Citation for the corpus:
The training and test sets used in this project are selected texxt files from the following corpus:
Clough, P. and Stevenson, M. Developing A Corpus of Plagiarised Short Answers, Language Resources and Evaluation: Special Issue on Plagiarism and Authorship Analysis, In Press
