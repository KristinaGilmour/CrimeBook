# CrimeBook

[CrimeBook](https://crimebook.streamlit.app/) is a book recommender focused on recommending crime, thriller and mystery book series based on the user input.  Each recommendation includes the name of the series, brief description of the series, number of books in the series and a link to the GoodReads page which contains a list of the books in the series in the correct reading order.

## Instructions:
1. Click into the text input box below the "I feel like reading:" and type in description of what you are looking for. Hit enter.
2. Once you receive five recommendations you can filter based on the number of the books in the series, using sliders on the left.
3. If you receive less than five recommendations, please expand your description or lower minimum number of books / increase maximum number of books.

## Modeling:
### Data:

Datasets used are:

* goodreads_book_series.json.gz
* goodreads_books_mystery_thriller_crime.json.gz
* goodreads_reviews_mystery_thriller_crime.json.gz

from [Good Reads Book Graph](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) site. Data sets were preprocessed and combined in order to identify crime, thriller and mystery book series, and group individual books based on the series they belong to. 

### Model:
1. Some basic text cleaning is applied to each book series description, stop words are removed and text is lemmatized and tokenized using nltk package. Finally cleaned and tokenized text is converted into numeric values using TF-IDF vectorization. When a user provides an input, it goes through the same process. Then cosine similarity is calculated between the user input vector and all of the book series vectors and a list of the similarities is saved.
2. Same process is repeated but this time for each book series text used contains descriptions of individual books innthe series. Again list of cosine similarity between the user input and each book series is calculated and saved.
3. Two cosine similarity vectors are combined using weighted average and only values above certain cutoff are saved.
4. Remaining values are sorted and top five results are selected and returned. If final list contains less then five results all of those are reported, together with suggestion that user needs to provide more detailed description or adjust filtering.  
