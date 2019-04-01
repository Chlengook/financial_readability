# -*- coding: utf-8 -*-
"""
Creates textual features from an intput paragraph
"""

# Load Packages
import textstat
import nltk 

# Create Features
class text_features:
    """
    Creates various text features for a paragraph
    """
    
    def __init__(self, paragraph):
        self.paragraph = paragraph
        
        # create sentences, word tokens as well as part-of-speech tags
        self.sentences = nltk.sent_tokenize(paragraph)
        self.tokens =  [nltk.word_tokenize(sent) for sent in self.sentences]
        self.pos_tag = [nltk.pos_tag(sent) for sent in self.tokens]
        
        # create the standard readability measures
        self.flesch = textstat.flesch_reading_ease(self.paragraph)
        

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
    
