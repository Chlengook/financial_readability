# -*- coding: utf-8 -*-
"""
Creates textual features from an intput paragraph
"""

# Load Packages
import textstat
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import pkg_resources
import ast
import spacy
#from collections import Counter
from pyphen import Pyphen
import pickle
#import xgboost 

# lead the language model from spay. this must be downloaded
nlp = spacy.load('en_core_web_md')
pyphen_dic = Pyphen(lang='en')

# set word lists to be used

## This corpus comes from the Cambridge English Corpus of spoken English and includes
## all the NGSL and SUP words needed to get 90% coverage.
NGSL_wordlist = set([
    ln.decode('utf-8').strip() for ln in
    pkg_resources.resource_stream('financial_readability', 'word_lists/NGSL_wordlist.txt')
])

## The Business Service List 1.0, also known as the BSL (Browne, C. & Culligan, B., 2016) is a list of approximately 1700 words 
## that occur with very high frequency within the domain of general business English. Based on a 64.5 million word corpus of business 
## texts, newspapers, journals and websites, the BSL 1.0 version gives approximately 97% coverage of general business English materials 
## when learned in combination with the 2800 words of core general English in the New General Service List or NGSL (Browne, C., Culligan, B., and Phillips, J. 2013) 
BSL_wordlist = set([
    ln.decode('utf-8').strip() for ln in
    pkg_resources.resource_stream('financial_readability', 'word_lists/BSL_wordlist.txt')
])

## New Academic Word List (NAWL): The NAWL is based on a carefully selected academic corpus of 288 million words.
NAWL_wordlist = set([
    ln.decode('utf-8').strip() for ln in
    pkg_resources.resource_stream('financial_readability', 'word_lists/NAWL_wordlist.txt')
])

## Load tf_idf score list
idf_list = list([
    ln.decode('utf-8').strip() for ln in
    pkg_resources.resource_stream('financial_readability', 'word_lists/dict_idf.txt')
])

idf_dict = ast.literal_eval(idf_list[0])

## Load the BOFIR model

with pkg_resources.resource_stream('financial_readability', 'models/bofir_model_5c.pickle.dat') as f:
            bofir_model_5c = pickle.load(f)
            
with pkg_resources.resource_stream('financial_readability', 'models/bofir_model_3c.pickle.dat') as f:
            bofir_model_3c = pickle.load(f)

#bofir_model_5c = pickle.load(open("bofir_model_5c.pickle.dat", "rb"))
#bofir_model_3c = pickle.load(open("bofir_model_3c.pickle.dat", "rb"))

#%%
# Create Features
class text_features:
    """
    Creates various text features for a paragraph
    
    Methods
    -------
    syl_count(word=None)
        Counts the number of syllables for a given word
        
    linguistic_features(as_dict = False)
        Returns the tokens and their linguistic features based on the spacy doc container
    
    pos_onehot():
        Creates the POS tag per token in a one-hot encoding.
    
    dep_onehot():
        Creates the dependency tag per token in a one-hot encoding.
        
    wordlist_features(as_dict=False):
        Creates word features based on word lists and the calculated tf-idf scores.
        
    other_features(as_dict=False):
        Returns dummies for the remaining spacy word features
        
    classic_features(as_dict=False):
        Returns the classic word features
    
    check_word_list(token, word_list='NGSL'):
        Function to check if token exists in specific word list.
     
    check_tfidf(token):
        Function to check if token exists in tf_idf list and return idf score.
    
    tree_features(as_dict=False):
        Function to create the tree based features.
    
    """
    
    def __init__(self, paragraph):
        self.paragraph = paragraph
        
        # create the standard readability measures
        self.flesch = textstat.flesch_reading_ease(self.paragraph)
        
        # create a spacy doc container
        self.doc = nlp(paragraph)
        
        # Spacy text variables
        self.token = [token.text for token in self.doc]
        self.sent = [sentence.text for sentence in self.doc.sents]
        self.lenght = [len(token.text) for token in self.doc]
        self.lemma = [token.lemma_ for token in self.doc]
        self.pos = [token.pos_ for token in self.doc]
        self.tag = [token.tag_ for token in self.doc]
        self.dep = [token.dep_ for token in self.doc]
        self.like_email = [token.like_email for token in self.doc]
        self.like_url = [token.like_url for token in self.doc]
        self.is_alpha = [token.is_alpha for token in self.doc]
        self.is_stop = [token.is_stop for token in self.doc]
        self.ent_type = [token.ent_type_ for token in self.doc]
        self.ent_pos = [token.ent_iob_ for token in self.doc]
        self.word_vectors = [token.vector for token in self.doc]
        self.vector_norm = [token.vector_norm for token in self.doc]
        self.is_oov = [token.is_oov for token in self.doc]
        
        # lexical chain - dependencies of words:
        self.subtree_lenght = [len(list(token.subtree)) for token in self.doc]
        self.n_left = [len(list(token.lefts)) for token in self.doc]
        self.n_right = [len(list(token.rights)) for token in self.doc]
        self.ancestors = [len(list(token.ancestors)) for token in self.doc] 
        self.children = [len(list(token.children)) for token in self.doc]
        
        # count syllables per token
        self.syllables = [self.syl_count(token.text) for token in self.doc]
        
        # number of sentences and tokens
        self.n_sentences = len(self.sent)
        self.n_tokens = len(self.token)
    
    def syl_count(self, word):
        """
        Counts the number of syllables for a given word
        
        Parameters
        ----------
        word : str
            The token to be analyzed
            
         Returns
         -------
         count: integer
             The number of syllables
        """
        
        count = 0
        split_word = pyphen_dic.inserted(word.lower())
        count += max(1, split_word.count("-") + 1)
        return count

    def linguistic_features(self, as_dict = False):
        """
        Function that returns the tokens and their linguistic features based on the spacy doc container
        doc: spacy doc input
        
        Parameters
        ----------
        as_dict : boolean
            Defines if output is a dataframe or dict
            
        Returns
        -------
        d: pandas DataFrame
            Dataframe with all the Output Variables
        
        Output Variables:
        -------
        
        Text: The original word text.
        Lemma: The base form of the word.
        POS: The simple part-of-speech tag.
        Tag: The detailed part-of-speech tag.
        Dep: Syntactic dependency, i.e. the relation between tokens.
        
        like_email: Does the token resemble an email address?
        is_alpha: Does the token consist of alphabetic characters?
        is stop: Is the token part of a stop list, i.e. the most common words of the language?
        ent_type: Named entity type
        ent_pos: IOB code of named entity tag.
        
        vector_norm: The L2 norm of the token’s vector (the square root of 
                                                        the sum of the values squared)
        is_oov: Out-of-vocabulary
        
        lexical chain variables determine the dependency tree:
        subtree_lenght: total number of suptrees
        n_left: number of connections left
        n_left: number of connections right
        ancestors: number of nodes above
        children: number of nodes below
        
        syllables: number of syllables (only for words found in the dictionary)
        """
        
        d = {'token':self.token,'lenght':self.lenght,'lemma':self.lemma, 
             'pos':self.pos,'tag':self.tag,
             'dep':self.dep,'like_email':self.like_email,'like_url':self.like_url,
             'stop':self.is_stop, 'alpha':self.is_alpha, 
             'ent_type':self.ent_type,'ent_pos':self.ent_pos,
             'vector_norm':self.vector_norm,'oov':self.is_oov,
             'subtree_lenght':self.subtree_lenght, 'n_left':self.n_left,
             'n_right':self.n_right,'ancestors':self.ancestors,
             'children':self.children,'syllables': self.syllables}
        
        if as_dict:
            return d
        else:
            return pd.DataFrame(d)
    
    def pos_onehot(self):
        """
        Creates the POS tag per token in a one-hot encoding. (To be agregated 
        over the paragraph or used as input into a RNN.)
        
        Returns
        -------
        d: pandas DataFrame
            Dataframe with all the Output Variables
        
        Output Variables:
        -------
        ADJ	    adjective	
        ADP	    adposition	
        ADV	    adverb	
        AUX	    auxiliary	
        CONJ	conjunction	
        CCONJ	coordinating conjunction	
        DET	    determiner	
        INTJ	interjection	
        NOUN	noun	
        NUM	    numeral	
        PART	particle	
        PRON	pronoun	
        PROPN	proper noun
        PUNCT	punctuation	
        SCONJ	subordinating conjunction	
        SYM	    symbol	
        VERB	verb	
        X	    other	
        SPACE	space
        """
        pos_tags_classes = ['ADJ', 'ADP', 'ADV','AUX', 'CONJ', 'CCONJ', 'DET', 
                            'INTJ', 'JJS', 'NOUN', 'NUM', 'PART', 
                            'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 
                            'X', 'SPACE']
        pos_tag_data = self.pos
        # one hot encoding of the different POS tags
        x = label_binarize(pos_tag_data, classes=pos_tags_classes)
        output = pd.DataFrame(x, columns=pos_tags_classes)
        
        return output
    
    def dep_onehot(self):
        """
        Creates the dependency tag per token in a one-hot encoding.
        
        Returns
        -------
        d: pandas DataFrame
            Dataframe with all the Output Variables
        
        Output Variables:
        -------
        acl	clausal modifier of noun (adjectival clause)
        acomp	adjectival complement
        advcl	adverbial clause modifier
        advmod	adverbial modifier
        agent	agent
        amod	adjectival modifier
        appos	appositional modifier
        attr	attribute
        aux	auxiliary
        auxpass	auxiliary (passive)
        case	case marking
        cc	coordinating conjunction
        ccomp	clausal complement
        compound	compound
        conj	conjunct
        cop	copula
        csubj	clausal subject
        csubjpass	clausal subject (passive)
        dative	dative
        dep	unclassified dependent
        det	determiner
        dobj	direct object
        expl	expletive
        intj	interjection
        mark	marker
        meta	meta modifier
        neg	negation modifier
        nn	noun compound modifier
        nounmod	modifier of nominal
        npmod	noun phrase as adverbial modifier
        nsubj	nominal subject
        nsubjpass	nominal subject (passive)
        nummod	numeric modifier
        oprd	object predicate
        obj	object
        obl	oblique nominal
        parataxis	parataxis
        pcomp	complement of preposition
        pobj	object of preposition
        poss	possession modifier
        preconj	pre-correlative conjunction
        prep	prepositional modifier
        prt	particle
        punct	punctuation
        quantmod	modifier of quantifier
        relcl	relative clause modifier
        root	root
        xcomp	open clausal complement
        """
        dep_tags_classes = ['acl', 'acomp', 'advcl','advmod', 'agent', 'amod', 
                            'appos', 'attr', 'aux', 'auxpass', 'case', 'cc', 
                            'ccomp', 'compound', 'conj', 'cop', 'csubj', 'csubjpass', 
                            'dative', 'dep','det', 'dobj', 'expl',
                            'intj', 'mark', 'meta', 'neg', 'nn', 'nounmod', 'npmod', 
                            'nsubj','nsubjpass', 'nummod', 'oprd',
                            'obj', 'obl', 'parataxis', 'pcomp', 'pobj', 'poss', 'preconj', 
                            'prep','prt', 'punct', 'quantmod', 
                            'relcl','root', 'xcomp']
        # one hot encoding of the different DEP tags
        x = label_binarize(self.dep, classes=dep_tags_classes)
        output = pd.DataFrame(x, columns=dep_tags_classes)
        return output
    
    def wordlist_features(self, as_dict=False):
        """
        Creates word features based on word lists and the calculated tf-idf scores.
        
        Parameters
        ----------
        as_dict : boolean
            Defines if output is a dataframe or dict
            
        Returns
        -------
        d: pandas DataFrame
            Dataframe with all the Output Variables
            
        """
        NGSL = [self.check_word_list(token.lower(), word_list='NGSL') for token in self.token]
        BSL = [self.check_word_list(token.lower(), word_list='BSL') for token in self.token]
        NAWL = [self.check_word_list(token.lower(), word_list='NAWL') for token in self.token]
        idf = [self.check_tfidf(token.lower()) for token in self.token]
        
        d = {'ngsl': NGSL,'bsl': BSL,'nawl': NAWL, 'idf': idf}
        
        if as_dict:
            return d
        else:
            return pd.DataFrame(d)
    
    def other_features(self, as_dict=False):
        """
        Returns dummies for the remaining spacy word features
        
        Parameters
        ----------
        as_dict : boolean
            Defines if output is a dataframe or dict
            
        Returns
        -------
        d: pandas DataFrame
            Dataframe with all the Output Variables
        """
        
        # the dummy variables
        is_entity = [1 if token != 'O' else 0 for token in self.ent_pos]
        like_email = [1 if token == True else 0 for token in self.like_email]
        like_url = [1 if token == True else 0 for token in self.like_url]
        is_stop = [1 if token == True else 0 for token in self.is_stop]
        is_alpha = [1 if token == True else 0 for token in self.is_alpha]
        is_oov = [1 if token == True else 0 for token in self.is_oov]      
        
        d = {'is_entity': is_entity,'like_email': like_email,'like_url': like_url,
             'is_stop': is_stop, 'is_alpha': is_alpha, 'is_oov': is_oov, 
             'vector_norm':self.vector_norm}
        
        if as_dict:
            return d
        else:
            return pd.DataFrame(d) 
    
    def classic_features(self, as_dict=False):
        """
        Returns the classic word features
        
        Parameters
        ----------
        as_dict : boolean
            Defines if output is a dataframe or dict
            
        Returns
        -------
        d: pandas DataFrame
            Dataframe with all the Output Variables
            
        """
        large_words = [1 if syl >= 3 else 0 for syl in self.syllables]
        polsyll = [1 if syl > 1 else 0 for syl in self.syllables]
        
        # the dummy variables     
        d = {'syllables': self.syllables, 'large_word': large_words,
             'polsyll':polsyll, 'lenght':self.lenght}
        
        if as_dict:
            return d
        else:
            return pd.DataFrame(d)
    
    def check_word_list(self, token, word_list='NGSL'):
        """
        Function to check if token exists in specific word list.
        
        Parameters
        ----------
        token : str
            The token to be analyzed
        word_list : str
            Defines the wordlist to be considered (NGSL, BSL or NAWL) if nothing 
            is specified, NAWL is considered
            
        Returns
        -------
        x: integer 
            Dummy (0 or 1) if word is in the specified word list
            
        """
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
    
    def check_tfidf(self, token):
        """
        Function to check if token exists in tf_idf list and return idf score.
        
        Parameters
        ----------
        token : str
            The token to be analyzed
            
        Returns
        -------
        value: integer 
            IDF value
        """
        value = idf_dict.get(token, 0)
        
        return value 
    
    def tree_features(self, as_dict=False):
        """
        Function to create the tree based features.
        
        Parameters
        ----------
        as_dict : boolean
            Defines if output is a dataframe or dict
            
        Returns
        -------
        d: pandas DataFrame
            Dataframe with all the Output Variables
        
        Output Variables
        -------
        subtree_lenght
        n_left
        n_right
        ancestors
        children
        """
        # lexical chain - dependencies of words:
        self.subtree_lenght = [len(list(token.subtree)) for token in self.doc]
        self.n_left = [len(list(token.lefts)) for token in self.doc]
        self.n_right = [len(list(token.rights)) for token in self.doc]
        self.ancestors = [len(list(token.ancestors)) for token in self.doc] 
        self.children = [len(list(token.children)) for token in self.doc]
    
        d = {'subtree_lenght':self.subtree_lenght, 'n_left':self.n_left,'n_right':self.n_right,
             'ancestors':self.ancestors,'children':self.children}
        
        if as_dict:
            return d
        else:
            return pd.DataFrame(d) 
        
    def semantic_features(self, as_dict=False):
        """
        Function to calculate the cumulative explained variance for PCA on the word embeddings.
        
        Parameters
        ----------
        as_dict : boolean
            Defines if output is a dataframe or dict
            
        Returns
        -------
        d: pandas DataFrame
            Dataframe with all the Output Variables
        
        Output Variables
        -------
        wordvec_1pc
        wordvec_3pc
        wordvec_10pc
        """
        pca = PCA()
        pca.fit(self.word_vectors)
        explained_var = pd.DataFrame(pca.explained_variance_ratio_, columns=['expl_var'])
        
        wordvec_1pc = np.sum(explained_var.iloc[0])
        wordvec_3pc = np.sum(explained_var.iloc[0:3])
        wordvec_10pc = np.sum(explained_var.iloc[0:10])
        
        d = {'wordvec_1pc':wordvec_1pc,'wordvec_3pc':wordvec_3pc,'wordvec_10pc':wordvec_10pc}
        
        if as_dict:
            return d
        else:
            return pd.DataFrame(d) 
        
    
    def word_features(self, embeddings = False):
        """
        Combines the featuresets to a Dataframe with 
        all the 88 word-features. 
        
        Parameters
        ----------
        embeddings : boolean
            Defines if the word embeddings (n=300) are included or not
            
        Returns
        -------
        d: pandas DataFrame
            Dataframe with all the Output Variables. Each row represents a token 
            and the features are in the columns (n x 88) as there are 88 word-features
        """
        classic_features = self.classic_features()
        pos_features = self.pos_onehot()
        dep_features = self.dep_onehot()
        wordlist_features = self.wordlist_features()
        other_features = self.other_features()
        tree_features = self.tree_features()
        
        if embeddings:
            nameslist = ["V{:02d}".format(x+1) for x in range(300)]
            word_embeddings = pd.DataFrame(self.word_vectors, columns = nameslist)
            return pd.concat([classic_features,pos_features,dep_features, wordlist_features,
                              other_features,tree_features,word_embeddings], axis=1)
        else:
            return pd.concat([classic_features, pos_features,dep_features, wordlist_features,
                              other_features,tree_features], axis=1)
        
    def paragraph_features(self, embed = False, as_dict = False):
        """
        Create the feature set over the total paragraph based on the 
        features estimated per word.
        
        Parameters
        ----------
        embed : boolean
            Defines if the word embeddings (n=300) are included or not
        as_dict : boolean
            Defines if output is a dataframe or dict
            
        Returns
        -------
        d: pandas DataFrame
            Dataframe with all the Output Variables. Each row represents a feature.
            columns: 
                cat: feature category
                value: value of the feature
            
        """
        # word embeddings
        word_embeddings_raw = pd.DataFrame(self.word_vectors, columns = ["V{:02d}".format(x+1) for x in range(300)])
            
        # create all datasets with the mean word values
        classic_features = pd.DataFrame(self.classic_features().mean(), columns= ['value'])
        classic_features['cat'] = 'classic'
        dep_features = pd.DataFrame(self.dep_onehot().mean(), columns= ['value'])
        dep_features['cat'] = 'dep'
        wordlist_features = pd.DataFrame(self.wordlist_features().mean(), columns= ['value'])
        wordlist_features['cat'] = 'classic'
        pos_features = pd.DataFrame(self.pos_onehot().mean(), columns= ['value'])
        pos_features['cat'] = 'pos'
        tree_features = pd.DataFrame(self.tree_features().mean(), columns= ['value'])
        tree_features['cat'] = 'tree'
        other_features = pd.DataFrame(self.other_features().mean(), columns= ['value'])
        other_features['cat'] = 'classic'
        semantic_features = pd.DataFrame(self.semantic_features().mean(), columns= ['value'])
        semantic_features['cat'] = 'semantic'
        word_embeddings = pd.DataFrame(word_embeddings_raw.mean(), columns= ['value'])
        word_embeddings['cat'] = 'embeddings'
        
        if embed:
            temp_df = pd.concat([classic_features, dep_features, wordlist_features, pos_features, other_features,
                              tree_features,semantic_features, word_embeddings], axis=0)
        else:
            temp_df = pd.concat([classic_features, dep_features, wordlist_features, pos_features, other_features,
                              tree_features,semantic_features], axis=0)
        
        temp_df['var'] = temp_df.index
           
        # add standard features that are not based on word features
        paragraph_features = pd.DataFrame(columns=['var','value', 'cat']) 
        paragraph_features.loc[0] = ['n_sentences'] + [self.n_sentences] + ['classic']
        paragraph_features.loc[1] = ['sent_lenght'] +[self.n_tokens/self.n_sentences] + ['classic']
        paragraph_features.loc[3] = ['n_tokens'] +[self.n_tokens] + ['classic']
        
        # add the entitiy based features (in addition to the percentage of entities)
        paragraph_features.loc[4] = ['n_entities'] + [self.other_features()['is_entity'].sum()] + ['entity']
        paragraph_features.loc[5] = ['ent_per_sent'] + [self.other_features()['is_entity'].sum() /self.n_sentences ] + ['entity']
        
        # additional dependency tree features
        paragraph_features.loc[6] = ['max_treelenght'] + [self.tree_features()['subtree_lenght'].max()] + ['tree']
        paragraph_features.loc[7] = ['q80_treelenght'] + [self.tree_features()['subtree_lenght'].quantile(.8)] + ['tree']
        paragraph_features.loc[8] = ['var_treelenght'] + [self.tree_features()['subtree_lenght'].var()] + ['tree']
        
        full_df = pd.concat([temp_df,paragraph_features], axis=0, sort=True)
        
        full_df = full_df.set_index('var').sort_values(by=['cat','var'])
        
        # manually rename some of the category labels:
        full_df.at['idf','cat'] = 'semantic'
        full_df.at['is_entity','cat'] = 'entity'
        full_df.at['vector_norm','cat'] = 'semantic'
        # reorder the df for output
        full_df = full_df.sort_values(by=['cat','var'])
        
        if as_dict:
            return full_df['value'].to_dict()
        else:
            return full_df
        
    def bofir(self, cat5 = True):
        """
        Use the paragraph features to calculate the BOFIR score for a
        given paragraph.
        
        Parameters
        ----------
        cat5: boolean
            Defines if BOFIR schould be calculated on 3 or 5 category scale
            
        Returns
        -------
        pred: integer
            The BOFIR score
            
        """
        feature_list = []
        feature_list.append(self.paragraph_features(embed = True, as_dict = True))
        data_df = pd.DataFrame(feature_list)
        # get predicted readability score
        prediction_5c = bofir_model_5c.predict(data_df)[0]
        prediction_3c = bofir_model_3c.predict(data_df)[0]
        
        if cat5:
            pred = prediction_5c
        else:
            pred = prediction_3c
        
        return pred
    
    def readability_measures(self, as_dict = False):
        """
        Return the BOFIR score as well as other classic readability formulas for the paragraph.
        
        Parameters
        ----------
        as_dict : boolean
            Defines if output is a dataframe or dict
            
        Returns
        -------
        d: DataFrame
            DataFrame with the BOFIR score and additional readability measures
            
        """
        flesch = self.flesch
        smog = textstat.smog_index(self.paragraph)
        dale_chall = textstat.dale_chall_readability_score(self.paragraph)
        fog = textstat.gunning_fog(self.paragraph)
        bofir_5cat = self.bofir(cat5 = True)
        bofir_3cat = self.bofir(cat5 = False)
        
        d = {'bofir_5cat':bofir_5cat, 'bofir_3cat':bofir_3cat,'fog':fog,
             'dale_chall':dale_chall,'smog':smog,'flesch':flesch}
        
        if as_dict:
            return d
        else:
            return pd.DataFrame(d, index=['readability_score'])
        
            
#%%    
# Example  
if __name__ == "__main__":  
    
    test_parapraph = ('The high functionality of the ABB products enables optimal performance of HVAC systems.'
                      + ' The power consumption of the Swiss actuators is reduced by means of energy-optimizing algorithms by 2012.')
    
    # initiate the object:
    test_text = text_features(test_parapraph)
    
    print('The Raw Tokens:')
    print('-------------------')
    print(test_text.token)
    print('-------------------')    
    word_features = test_text.word_features()
    word_features_total = test_text.word_features(embeddings = True)
    print('-------------------')
    print('Paragraph features:')
    print('-------------------')
    par_features = test_text.paragraph_features(embed = False)
    print(par_features.head())
    print('-------------------')
    print('BOFIR score:')
    print('-------------------')
    print(test_text.bofir(cat5 = False))
    
