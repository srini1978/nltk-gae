**EDIT** Aug 29 2013: For anyone looking for interesting work in Indian language NLP or speech recognition mail me your resumes. Uniphore is hiring - sharun@uniphore.com

This is a very small part of the python nltk library to support some basic NLP (Natural Language Processing) functionality on appengine, namely the default sentence tokenizer, word tokenizer, part of speech tagger and regex parser.

Based on nltk 0.9.9 using Python 2.5

Source - http://code.google.com/p/nltk-gae/source/browse/#svn%2Ftrunk

Demo - http://sharun-s.appspot.com/auto

Motivation - ~~http://www.google.com/buzz/sharunsanthosh/9E7UfxdVqgx~~~~

Since the Google Buzz link is dead, pasting the original posting below-

Nov 26, 2010
### Natural Language Processing (NLP) ###

I have been trying to get an NLP demo online for a while now. And it isn't easy cause most of the operations are resource intensive. But after much mutilation of the python nltk toolkit, I have managed to create a demo page here - imine.in/extract

Paste your text in the left textbox and click Phrases or PoS button. The server extracts and returns the'interesting' phrases or noun phrases to be precise. If you press PoS the part of speech tags are returned. It may take 3-4 seconds for a response so be patient. For developers who want an API there is a note at the end of the post.

A noun phrase is just a sequence of nouns or an adjective followed by a sequence of nouns. In order to do this 4 things need to happen. Extract sentences - Extract words - Run the words through a part of speech tagger - Run the part of speech tagged words through a parser that identifies sequences based on a set of rules (or a grammar). And so you get your phrases. These phrases are extremely useful. You can use them to categorize news, analyse intent, identify locations, people, organizations...endless possibilities. But getting back to NLP...

Where NLP helps:
If I want to extract all the names of people mentioned in a speech, one way to do it is to compare each word against all known names in the universe. Collecting all the names in the universe is a tad difficult and I still might not find everything if the para happens to have names from another universe, which it just might.

But the problem isn't that simple. Lets be honest, some people have ridiculously long names. So now I don't just compare every word, I have to compare all possible sequences of words in the speech against that uncompilable list of all known names. And this is just a simplistic description of the problem, there are an endless bunch of other complications that have kept and will keep many people employed for a long time to come.

Yet most kids have developed the ability to do it at a very young age. They can do it without any kind of prior exposure to any of those names. They don't need to be able to read or write to do it. They don't need to be taught the difference between a noun and a verb to do it. Yet it all magically works. And the answers lie in the workings of language.

NLP is a natural evolution of conventional text processing. It makes use of our existing knowledge of languagewhich can minimize the complexity of certain problem. It takes care of things like sentence boundary detection (yes there are many ways to end a sentence and often none are used, creating a whole bunch of headaches, also there are issues like abbreviations,quotations whose punctuations can be mistaken for the end of a sentence). word tokenization (the end of a word is not always a space), part of speech tagging and parsing. For developers interested the nltk toolkit [nltk.org] which also includes a free online book is a great place to start.

The phrases returned match a simple regex rule` <JJ.*>?<NN.*>+ `
Here are the list of possible pos tags http://bulba.sdsu.edu/jeanette/thesis/PennTags.html#Word
You may want to just retrieve the pos tags and make your own rules for phrase extraction.

### NLP Demo Update ###

There were a couple of issues with the demo ( imine.in/extract ) and requests were getting dropped for many reasons. This shouldn't be happening anymore (hopefully). Requests were taking around 5-10 seconds, painfully long for a webapp. It's now down to 1-2s (dependent on length of text). Just wanted to add a note here to explain what was and is going on.

I am using nltk's default sentence tokenizer and part-of-speech tagger. The tagger uses a maximum entropy classifier. Goal is to take a word and tell you if its a noun or a verb or an adjective etc. There are 46 differentparts of speech tags that the classifier needs to choose from. So it takes a word (or a token to use the right lingo), its position in the sentence and generates a bunch of features about it (previous token, prefix, suffix, is it capitalized etc). The classifier uses a precreated db of ~200000 (feature, tag) to weight mappings, that it uses to calculate, a probability for each tag, given the feature set of a token. So 46 probabilities are calculated for each token. The tag with highest probability wins.

Now if the mappings are kept in a db online and you are trying to tag text with say 30 tokens. Let's say 4 features are detected. For each token you get 30\*4\*46 calls to the db to compute the probabilities. Thats 5520 calls. Even with 5-10 millisecond db access that's a whole lot of seconds for each request.

So enter memcache. App Engine docs say its memory access (memcache) is 5 times faster than db access(reads) so currently the demo tries to keep the entire damn mapping table in memory. And memcache supports this quite beautifully. Requests are performing much better now.