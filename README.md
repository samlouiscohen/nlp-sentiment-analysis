# nlp-sentiment-analysis

A program to preform sentiment analysis on IMDB movie reviews. 
The classifier writes each prediction to a file where '1' corresponds to a positive review and '0' corresponds to a negative review.

I used both a unigram and bigram representation of the data in order to draw conclusions about which one was better suited to the set given my constraints. 

Also, alongside each vanilla representation I included one implementing tf–idf, or "short for term frequency–inverse document frequency".

After performing cross validation on the training set, I noted that suprisingly vanilla unigram had the bes performance with an accuracy hovering around 84%.

