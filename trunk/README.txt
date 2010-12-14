First off this is a big hack to get basic nltk functionality on appengine.
All it supports is basic sentence tokenization, word tokenization, PoS tagging and a RegExpParser.
That was all I needed for an online NLP demo - http://www.google.com/buzz/sharunsanthosh/9E7UfxdVqgx

Uses nltk 0.9.9 for Windows
The sentence tokenizer is from nltk_data\tokenizers\punkt\english.pickle
The PoS tagger is from nltk_data\taggers\maxent_treebank_pos_tagger\english.pickle

The tagger pickle has been modified to reduce its size and remove a dependency on numpy
This was what was done
tag = nltk.data.load('taggers\maxent_treebank_pos_tagger\english.pickle')
c=tag.classifier()
These are the four structures required by the algo : 
c._encoding._labels, c._encoding._mapping, c._encoding._alwayson, c._weights
c._encoding._mapping  is a mapping from feature,value,tag -> index in c._weights 
c._weights is a numpy vector and to remove the numpy dependency use
c._weights.tolist() to get a python list. 
This list of weights is merged into mapping to reduce the overall size of the tagger pickle.
So now mapping is feature,value,tag -> weight

To get it all working on appengine the main 'trick' is to keep the classifer in memory. Memcache is used to get this done.
Good to remember this is distributed RAM. App Engine understandably doesn't make any guarantees about how long objects once in the cache stay in the cache. 
So a cron job is run periodically to keep everything in the cache. 
Another issue is the tag function makes a lot of calls to memcache on every request. 
So the algo code has been modified to reduce the number of calls to memcache.
More explaination on this issue here - http://www.google.com/buzz/sharunsanthosh/KqLoKEUDf9J

Feedback, patches, bug reports etc are most welcome. Thanks!

