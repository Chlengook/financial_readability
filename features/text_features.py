# -*- coding: utf-8 -*-
"""
Creates textual features from an intput paragraph
"""

# Load Packages
import textstat
import nltk 
import sklearn
import pandas as pd
import pkg_resources

# set word lists to be used

## This corpus comes from the Cambridge English Corpus of spoken English and includes
## all the NGSL and SUP words needed to get 90% coverage.
NGSL_wordlist = set([
    ln.decode('utf-8').strip() for ln in
    pkg_resources.resource_stream('financial_readability.features', 'NGSL_wordlist.txt')
])

## The Business Service List 1.0, also known as the BSL (Browne, C. & Culligan, B., 2016) is a list of approximately 1700 words 
## that occur with very high frequency within the domain of general business English. Based on a 64.5 million word corpus of business 
## texts, newspapers, journals and websites, the BSL 1.0 version gives approximately 97% coverage of general business English materials 
## when learned in combination with the 2800 words of core general English in the New General Service List or NGSL (Browne, C., Culligan, B., and Phillips, J. 2013) 
BSL_wordlist = set([
    ln.decode('utf-8').strip() for ln in
    pkg_resources.resource_stream('financial_readability.features', 'BSL_wordlist.txt')
])

## New Academic Word List (NAWL): The NAWL is based on a carefully selected academic corpus of 288 million words.
NAWL_wordlist = set([
    ln.decode('utf-8').strip() for ln in
    pkg_resources.resource_stream('financial_readability.features', 'NAWL_wordlist.txt')
])

#%%
# Create Features
class text_features:
    """
    Creates various text features for a paragraph
    """
    
    def __init__(self, paragraph):
        self.paragraph = paragraph
        
        # create sentences, word tokens as well as part-of-speech tags
        self.sentences = nltk.sent_tokenize(paragraph)
        self.tokens_sent =  [nltk.word_tokenize(sent) for sent in self.sentences]
        self.tokens =  nltk.word_tokenize(paragraph)
        self.pos_tag = [nltk.pos_tag(sent) for sent in self.tokens_sent]
        
        # create the standard readability measures
        self.flesch = textstat.flesch_reading_ease(self.paragraph)
        
    def pos_tag_onehot(self):
        """
        Creates the POS tag per token in a one-hot encoding. (To be agregated ofer the paragraph or used as input into a RNN.)
        """
        pos_tags_classes = ['CC', 'CD', 'DT','EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM','TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
        lst = []
        # create list with POS tags
        for item in self.pos_tag:
            lst += item
        pos_tag_data = [item[1] for item in lst]
        # one hot encoding of the different POS tags
        x = sklearn.preprocessing.label_binarize(pos_tag_data, classes=pos_tags_classes)
        output = pd.DataFrame(x, columns=pos_tags_classes)
        return output
    
    def word_features(self):
        """
        Creates additional word features.
        """
        stop_words = set(nltk.corpus.stopwords.words('english'))
        word_lenght = [len(token) for token in self.tokens]
        is_stopword = [1 if token.lower() in stop_words else 0 for token in self.tokens]
        syllables = [textstat.syllable_count(token) for token in self.tokens]
        NGSL = [self.check_word_list(token.lower(), word_list='NGSL') for token in self.tokens]
        BSL = [self.check_word_list(token.lower(), word_list='BSL') for token in self.tokens]
        NAWL = [self.check_word_list(token.lower(), word_list='NAWL') for token in self.tokens]
        output = pd.DataFrame({'wlen': word_lenght,'stopw': is_stopword,'syll': syllables,'ngsl': NGSL,'bsl': BSL,'nawl': NAWL})
        return output
    
    def check_word_list(self, token, word_list='NGSL'):
        if word_list=='NGSL':
            word_set = NGSL_wordlist
        elif word_list=='BSL':
            word_set = BSL_wordlist
        else:
            word_set = NAWL_wordlist
       
        if token not in word_set:
            x = 0
        else:
            x=1
        return x
 
#%%    
# Testing    
if __name__ == "__main__":  
    test_parapraph = ('The high functionality of the products enables optimal performance of HVAC systems.'
                      + ' The power consumption of the actuators is reduced by means of energy-optimizing algorithms.')
    
    test_text = text_features(test_parapraph)
    print('-------------------')
    print(test_text.sentences)
    print('-------------------')
    print(test_text.tokens)
    print('-------------------')
    print(test_text.pos_tag)
    print('-------------------')
    print(test_text.flesch)
    print('-------------------')
    test_pos = test_text.pos_tag_onehot()
    print('-------------------')
    test_other = test_text.word_features()
    print('-------------------')
    test_text.check_word_list('regression')

    
