# plagiarism-checker

The plagiarism checker does the following: (also consider this as our todos list)\n
1.loads files 
2. preprocessing (lowercase, removing punctuation, changing numbers to words, lemmatization, stemming) 
3. performs tf-idf vectorization on the contents of the files
4. Find cosine similarity between the resulting vectors
5. return similarity score, calculate precision and recall
