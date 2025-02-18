import math  # import math for calculations
from collections import defaultdict, Counter  # imports for frequency counting

class NGramModel:  # ngram model class
    def __init__(self, n, smoothing=None, alpha=1):  # constructor
        self.n = n  # store n gram size
        self.smoothing = smoothing  # store smoothing method
        self.alpha = alpha  # store smoothing parameter
        self.ngram_freqs = defaultdict(Counter)  # dict to store n gram counts
        self.uni_freqs = Counter()  # counter to store unigram counts
        self.total_tokens = 0  # var to track total tokens in corpus
        self.vocab = set()  # set for unique words in corpus
    
    # splitting and normalizing text
    def tokenize(self, text): 
        words = text.strip().lower().split()  # lowercase/split into words
        return words  # return tokenized words
    
    # 1. training unsmoothed unigram/bigram lm's
    def train(self, data):  # train n gram model
        for sentence in data:  # loop through sentences in training set
            words = self.tokenize(sentence)  # tokenize sentence
            self.total_tokens += len(words)  # update total token count
            self.vocab.update(words)  # update vocabulary set
            
            for word in words:  # loop through words
                self.uni_freqs[word] += 1  # unigram freq count
            
            for i in range(len(words) - self.n + 1):  # iterate to extract n grams
                prefix = tuple(words[i:i+self.n-1])  # get n-1 prefix
                suffix = words[i+self.n-1]  # get last word of n gram
                self.ngram_freqs[prefix][suffix] += 1  # n gram freq count
    
    # 2. implementing smoothing: unknown word handling
    def laplace_prob(self, prefix, word):  # find laplace smoothed probability
        if self.n == 1:  # check if unigram
            return (self.uni_freqs[word] + 1) / (self.total_tokens + len(self.vocab))  # laplace smoothed unigram probability
        else:  # bigram, trigram, etc. 
            prefix_count = sum(self.ngram_freqs[prefix].values())  # prefix occurrences count
            word_count = self.ngram_freqs[prefix][word]  # n gram occurrence count
            return (word_count + 1) / (prefix_count + len(self.vocab))  # laplace smoothed probability
    
    def add_k_prob(self, prefix, word, k=0.5):  # compute add k smoothed probability
        if self.n == 1:  # check if unigram 
            return (self.uni_freqs[word] + k) / (self.total_tokens + k * len(self.vocab))  # add k smoothed unigram probability
        else:  # bigram, trigram, etc.
            prefix_count = sum(self.ngram_freqs[prefix].values())  # prefix occurrences count
            word_count = self.ngram_freqs[prefix][word]  # n gram occurrence count
            return (word_count + k) / (prefix_count + k * len(self.vocab))  # add k smoothed probability
    
    def get_prob(self, prefix, word):  # find probability with/without smoothing
        if self.smoothing == 'laplace':  # if laplace smoothing 
            return self.laplace_prob(prefix, word)  # return laplace probability
        elif self.smoothing == 'add-k':  # if add k smoothing
            return self.add_k_prob(prefix, word, self.alpha)  # return add k probability
        else:  # no smoothing applied
            if self.n == 1:  # unigram 
                return self.uni_freqs[word] / self.total_tokens if word in self.uni_freqs else 0  # return probability
            else:  # bigram, trigram, etc. 
                prefix_count = sum(self.ngram_freqs[prefix].values())  # prefix occurrences count
                word_count = self.ngram_freqs[prefix][word]  # n gram occurrence count
                return word_count / prefix_count if prefix_count > 0 else 0  # return probability
    
    # 3. finding perplexity on validation set
    def find_perplexity(self, data): 
        log_prob_sum = 0  # init sum of log %
        total_words = 0  # total words in validation set

        for sentence in data:  # loop through validation sentences
            words = self.tokenize(sentence)  # tokenize sentence
            total_words += len(words)  # update word count

            for i in range(len(words) - self.n + 1):  # loop through n grams
                prefix = tuple(words[i:i+self.n-1])  # n-1 prefix extraction
                word = words[i+self.n-1]  # last word of n gram extraction
                prob = self.get_prob(prefix, word)  # compute %
                prob = max(prob, 1e-10)  # log0 avoiding
                log_prob_sum += math.log(prob)  # add log %

        if total_words == 0:  # if empty validation set
            print("empty validation set.")  # warning
            return float('inf')  # return that there is high perplexity 
        
        return math.exp(-log_prob_sum / total_words)  # find perplexity and return

    def display_model_probabilities(self):  # display model probabilities for unigram and bigram
        for word, count in self.uni_freqs.items():  # loop through unigrams
            print(f"P({word}) = {count/self.total_tokens:.3f}")  # unigram %
        
        for prefix, words in self.ngram_freqs.items():  # loop through bigrams
            for word, count in words.items():  # loop through suffixes
                print(f"P({word}|{' '.join(prefix)}) = {count/sum(words.values()):.3f}")  # bigram %
           
if __name__ == "__main__":  # main 
    with open("train.txt", "r", encoding="utf-8") as file:  # open training set
        train_data = file.readlines()  # read training data
    
    with open("val.txt", "r", encoding="utf-8") as file:  # open validation set
        val_data = file.readlines()  # read validation data
    
    unigram = NGramModel(n=1)  # unigram model instance
    unigram.train(train_data)  # train unigram model
    unigram.display_model_probabilities()  # display unigram %
    
    bigram = NGramModel(n=2)  # bigram model instance
    bigram.train(train_data)  # train bigram model
    bigram.display_model_probabilities()  # display bigram %
    print("unigram perplexity:", unigram.find_perplexity(val_data))  # find unigram perplexity
    print("bigram perplexity:", bigram.find_perplexity(val_data))  # find bigram perplexity
    
    laplace_unigram = NGramModel(n=1, smoothing='laplace')  # unigram model with laplace smoothing
    laplace_unigram.train(train_data)  # train 
    print("laplace unigram perplexity:", laplace_unigram.find_perplexity(val_data))  # find perplexity
    
    add_k_bigram = NGramModel(n=2, smoothing='add-k', alpha=0.5)  # bigram model with add k smoothing
    add_k_bigram.train(train_data)  # train 
    print("add-k bigram perplexity:", add_k_bigram.find_perplexity(val_data))  # compute perplexity

    print("unigram training perplexity:", unigram.find_perplexity(train_data))  
    print("bigram training perplexity:", bigram.find_perplexity(train_data))  
    print("laplace unigram training perplexity:", laplace_unigram.find_perplexity(train_data))  
    print("add-k bigram training perplexity:", add_k_bigram.find_perplexity(train_data))  

