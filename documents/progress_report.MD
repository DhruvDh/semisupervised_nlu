# Semi-supervised Natural Language Understanding

- Dhruv Dhamani
- ddhamani@uncc.edu

## Table of Contents

- [Semi-supervised Natural Language Understanding](#semi-supervised-natural-language-understanding)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [**Semi-supervised** Natural Language Understanding:](#semi-supervised-natural-language-understanding)
    - [Semi-supervised **Natural Language Understanding**:](#semi-supervised-natural-language-understanding)
  - [Background](#background)
    - [Language Models](#language-models)
    - [Related work](#related-work)
  - [Dataset](#dataset)
    - [Our Approach](#our-approach)
  - [Performance metrics](#performance-metrics)
  - [Logistics](#logistics)
    - [Updated Timeline](#updated-timeline)
  - [References and Citation](#references-and-citation)

## Introduction

Let's break down the project title in two for clarity -

### **Semi-supervised** Natural Language Understanding:
  
Machine learning approaches are tending to either be supervised or unsupervised.

  - A supervised approach as in when you have curated data with a clear expectation for what parts of the data are the *input* to the model and what part the model is supposed to predict/infer or present as an *output*.
  - In general, unsupervised approaches mean that the data being used has not been labelled, classified or categorized. Instead, we create some sort of a feedback mechanism for the model which helps it identify commonalities in the data and react based on the presence or absence of such commonalities in each new piece of data.
  
  When I say *semi-supervised* what I mean is that I intend going to take pre-existing labelled, classified or categorized data - the kind of data one uses for *supervised learning*, and transform and curate it into unlabeled data - the kind of data one uses for *unsupervised learning* - with the transformation or curation being done *on the basis of the labels of our pre-existing labelled data*, and then use this curated data for training our model.

  **Why would anyone do such a thing?** 
  
  Well, the motivation is simple. One could argue that as a general rule, there is a lot more unlabeled data in existence than labelled data. Thereby, if one creates a machine learning system that learns by use of unlabelled data, it is always going to have more data to learn from than a system that is based on learning from labelled data.

  And in most cases, the more data you have, the better machine learning systems can learn. Thusly, by transforming and curating labelled data for supervised learning approaches into unlabeled data for unsupervised learning approaches, we also manage to increase the available data for learning manifold; assuming the area of application of said machine learning system have availability of unlabeled data that can be learned from, and satisfactory feedback mechanisms for unsupervised learning to take place.

### Semi-supervised **Natural Language Understanding**:
  
  Natural language understanding or Natural language inference is a subtopic of natural-language processing in artificial intelligence that deals with machine reading comprehension. It can be thought of as what happens after natural language processing (NLP) - if a computer was presented with the sentence -
    
    This is a sentence.

  The results of performing NLP techniques and algorithms on this sentence would give us information about what the individual words in the sentence are relative to the grammar of that language, and what the relationship between the words in the sentence is. It would look something like -

  ![Example of spacy's NLP](spacy-viz.svg)

  Now taking this set of information produced by NLP algorithms and getting a machine to _comprehend_ or _understand_ what the sentence means is what Natural Language Understanding/Inference is.

  Our above discussion of the motivation behind doing "semi-supervised" learning lead us to the conclusion that semi-supervised learning might be a good thing to do in the following scenarios -

  - when there is availability of large amounts of raw data that can be learned from
  - when there exist satisfactory feedback mechanisms to facilitate unsupervised learning
  - (and also, when labelled data can be transformed and curated into labelled data without a whole lot of hiccups)

  One field that satisfies all these requirements is Natural Language processing -

  1. Availability of large amounts of raw data - Large amounts of written down, digital representations of language, i.e. text, is basically what the internet is. I don't think more needs to be said here.
  2. Existence of satisfactory feedback mechanisms to facilitate unsupervised learning - Academia has been doing unsupervised learning in the field of Natural Language Processing for years. The process in which word vectors and language models are learnt - the process of capturing meaning of words in a vector of floating-point numbers by providing lots of examples of how the words are used in the natural language *is* unsupervised learning.
  3. Ease of transformation and curation of labelled data into unlabeled data - more on this in the review of related work and the approach and summary of methods sections below.

## Background

### Language Models

Traditionally, language models refer to statistical language models which can be thought of as a probability distribution over a sequence of words. Given a sequence of words (sentence, sentences, etc.), a statistical language model tells us the probability of that sequence of words existing in this language.

For instance, an English language model would assign `I love dogs` a higher probability than `I dogs love`. 

__Why is this important?__

In a way, language models assign a sensibility score or a naturalness score. This is due to the fact that the language models are trained with lots of sensible and natural sentences (ideally), and if so it is obvious that sentences like `I love dogs` (a sensible sentence) has a much higher probability in occurring the data the model has been trained with than `I dogs love` (a nonsensical one).

This is an important concept to remember for understanding the rest of this document. And as such we'll explain it again in a different way.

Consider the following two sentences -
```
A. I love dogs so I got one.
B. I dogs love in my car.
```

Now let us assume there exists a perfect, ideal language model. Would that model assign a higher score to sentence `A` or `B`? The answer is obviously `A`.
Let's say the model assigns a probability `0.0000000473` to the `A` and `0.00000000000000000000000000823` to `B`. On account of this score, we can conclude that `A` is a sentence that is more likely to *naturally occur* in the English language than `B`; and since sentences that *naturally occur* in the English language tend to be sentences that make sense, it is also okay to assume that `A` is a more sensible sentence than `B`.

Let's take things a step further and consider the next two sentences -
```
A. I love dogs. That is why I got one.
B. I love dogs. That is why Canada sucks.
```

What sort of score would a perfect, ideal language model assign to these sentences? They both seem grammatically correct, so the scores of both would probably be on the higher side. But which sentence is more _natural_? Probably the same sentence which is more sensible?

Let's read sentence `A` again. Getting a dog because you love dogs is a sensible thing to do. And at first glance `B` is a nonsensical sentence. But then again, we don't know the context of sentence `B`. Maybe the sentence is from an article about lawmakers in Canada proposing a bill that demands mandatory neutering of all dogs. In that case, `B` is a sensible sentence too.

But `A` is still more *natural* or more *sensible* than `B`, since `A` would make sense in most contexts, while `B` only makes sense in some contexts.

What I am trying to argue is that if language models are good enough, they can also be thought of as *sensibility models* or at least *common-sense models*.

Even if you do agree with everything that was just said, you would still doubt the feasibility or practicality of training a model that was actually a good "sensibility" model or "common-sense" model. These are good doubts to have, but thankfully there are researchers who have done work to credence this sort of thought process.

> Neural language models are different than statistical language models in the sense that the task neural language models are trained to do is to predict the next word after a sequence of words or predict the missing words in a sequence. 
> 
> The word that the neural language model predicts is the word which when added to the given input sequence, gives us a sequence with the highest probability of occurring in the language. A subtle but important difference. 
> 
> Neural language models can also thusly generate language by predicting the next word again and again. For instance, given a sequence `I love dogs so I got`, a neural language model might generate the next few words as `a dog`, etc. For the rest of the document whenever a language model is mentioned we refer to it being as something that predicts the next word(s), rather than something that gives a probability score for the sequence of input words.

### Related work

- **[The Natural Language Decathlon: Multitask Learning as Question Answering](https://arxiv.org/abs/1806.08730):**

 [5] The focus of this paper was on introducing a new benchmark for measuring the performance of NLP models. They presented MQAN model for simple question answering which capitalizes on questions with the help of a multi-pointer-generator decoder. They demonstrated how labelled data can be used to train a language model to perform multiple tasks by casting all tasks as question-answers over a context. We are using this concept to understand the question answer mechanism in order to make the model extract the correct query to perform slot filling.

  However, the MQAN model this paper used takes the questions and the context as inputs separately, while we ideally do not want to explicitly tell the model which part of the sequence is the question and which part is the context so as to retain "naturalness" and also to minimize the amount work that needs to be done to adapt a model to a different task. 

  | ![MQAN model](MQAN.png) | 
  |:--:| 
  | The MQAN model - Questions and context are passed separately. |

- **[Learning and Evaluating General Linguistic Intelligence](https://arxiv.org/abs/1901.11373):**

 [9] This paper defines general linguistic intelligence as the ability to reuse previously acquired knowledge about a language’s lexicon, syntax, semantics, and pragmatic conventions to adapt to new tasks quickly. Using this definition, they analyze state-of-the-art natural language understanding models and perform experiments to evaluate them against these criteria through a series of experiments that assess the task-independence of the knowledge being acquired by the learning process. The results show that while the field has made impressive progress in terms of model architectures that generalize to many tasks, these models still require a lot of in-domain training examples (fine tuning, training task-specific modules), and are prone to catastrophic forgetting. Moreover, they find that far from solving general tasks (e.g., document question answering), the models are overfitting to the quirks of particular datasets (like SQuAD).

  The authors also write that they believe *"generative language models will drive progress on general linguistic intelligence"*. What they call general linguistic intelligence is conceptually not all that different from what I attempted to describe as a model of sensibility above.

- **[Language models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf):**

 [3] This paper is where the feasibility and practicality of language models being "intelligent" is demonstrated. The authors first construct a high-quality dataset of millions of human-curated web-pages called WebText. They then proceed to train a modified version of the original Generative Pre-trained Transformer (GPT), called Generative Pre-Trained Transformer 2 on WebText, and demonstrate how the model - without any explicit supervision - achieves state of art results on 7 out of 8 tested language modeling datasets in a zero-shot setting but still underfits WebText.


- **[Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146):**

 [6] This paper is directed towards understanding how lack of knowledge about fine tuning can be a hindrance. The researchers have presented various novel fine tuning techniques to achieve effective results like discriminative fine-tuning, target task classifier fine-tuning, etc. They have also defined a Universal Language Model Fine-Tuning(ULMFiT) method to make robust models. This method comprises of three stages, firstly, LM is trained on a general domain corpus for capturing general features of the language, then the LM is fine-tuned using discriminative fine-tuning and slanted triangular learning rates. Finally the classifier is fine-tuned on target task using gradual freezing. The ULMFiT was able to achieve state-of-the-art performance on widely used text classification tasks.   

- __[XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237):__

 [7] Here, the authors hypothesize on a possible problem with BERT's pretraining objective which they believe causes "data corruption". They do not like the fact that BERT's pretraining objective assumes the masked words are independent given the unmasked words. As a solution to this problem they propose a novel "dual-masking-plus-caching mechanism to induce an attention-based model to learn to predict tokens from all possible permutations of the factorization order of all other tokens in the same input sequence." At the end they demonstrate that their model achieves modest gains over results from the orignal BERT paper.

 > While I agree that BERTs masking of multiple input words might entail that sometimes the required information to predict a certain masked word lie in another masked word, I do not believe it is as much of a problem as the authors make it out to be. There are other, far more convinient, ways to work this problem as we'll see in the next paper. It must also be noted that while XLNet boasts of modest gains over BERT, they've trained on significantly more data (~16GB vs ~33GB) and used significantly more compute to acheive said gains, making any comparison meaningless. I do not believe this to be a fair comparison.

 - __[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692):__

  [11] RoBERTa iterates on BERT's pretraining procedure, including training the model longer, with bigger batches over more data; removing the next sentence prediction objective; training on longer sequences; and dynamically changing the masking pattern applied to the training data.

> As mentioned in the previous paper, BERT's pretraining objective does make a faulty assumption that the input words that it masks during training are independent of each other given the respective sequence of unmasked words. What I like the most about this work is that it works around this problem by changing the set of masked words for each iteration of training ensuring that a new set of masked words is presented to the model each time, reducing the negative effects of any loss of dependency information between masked words. It is also something I am somewhat skeptical of, since it also means that after enough iterations the model "sees" basically all of the training data, making it more likely to overfit. The paper boasts of consistent improvement on BERT on many tasks, so my concerns are probably unfounded.

## Dataset

The dataset focuses on seven *intents* (things a user could want to do) -
* SearchCreativeWork (e.g. *Find me the I, Robot television show*),
* GetWeather (e.g. *Is it windy in Boston, MA right now?*),
* BookRestaurant (e.g. *I want to book a highly rated restaurant for me and my boyfriend tomorrow night*),
* PlayMusic (e.g. *Play the last track from Beyoncé off Spotify*),
* AddToPlaylist (e.g. *Add Diamonds to my roadtrip playlist*)
* RateBook (e.g. *Give 6 stars to Of Mice and Men*)
* SearchScreeningEvent (e.g. *Check the showtimes for Wonder Woman in Paris*)

More than 2000 samples exist for each intent. The dataset here on slot filling (figuring out what are the entities in the utterance that are relevant to carrying out the user's intent), for example -

*“Is it gonna be sunny on **Sunday after lunch**, so we can go to **Coney Island**?”*

Here the slot of `date-time` needs to be filled with *Sunday after lunch*, and `location` needs to be filled with *Coney Island*.

### Our Approach

To understand the approach easily, let's consider a sample from the dataset -

*"Book spot for three at Maid-Rite Sandwich Shop in Antigua and Barbuda"*

It is stored in the dataset as the following JSON object.

```javascript
[
  {
    "text": "Book spot for "
  },
  {
    "text": "three",
    "entity": "party_size_number"
  },
  {
    "text": " at "
  },
  {
    "text": "Maid-Rite Sandwich Shop",
    "entity": "restaurant_name"
  },
  {
    "text": " in "
  },
  {
    "text": "Antigua and Barbuda",
    "entity": "country"
  }
]
```

The relevant slots that need to be filled here are `party_size_number`, and `restaurant_name`. As such, this can be casted as the following question-answer pairs -

- `Book a spot for three at Maid-Rite Sandwich Shop in Antigua and Barbuda. Which restaurant? [...]`
- `Book a spot for three at Maid-Rite Sandwich Shop in Antigua and Barbuda. What is the party size? [...]` 

The `[...]` part is where the generative language model starts predicting the next few words, which ends up telling us how to fill the `restaurant_name` slot for the first example, and `party_size_number` for the second example.

However, the dataset our generative language model is trained on probably does not have many examples of such question-answer pairs occurring in it. It is composed mostly of news articles, blog posts, etc. As such we hypothesize such an approach would not end up working out that well, without fine-tuning. How the model would perform after fine-tuning, well, I think we'll know only after doing it.

Another way to accomplish this same task would be to prompt the model into giving us an answer. I hypothesize that this would work better. For example -

- `Book a spot for three at Maid-Rite Sandwich Shop in Antigua and Barbuda. I'll book a table for you at [...]`
- `Book a spot for three at Maid-Rite Sandwich Shop in Antigua and Barbuda. I'll book a table for [...]` 


I would be using a pretrained RoBERTa as our language model, and attempt to fine-tune it and see how much better we can get it to work.

## Performance metrics

Evaluating how a generative language model is performing at such tasks isn't straight forward. For instance, consider the utterance -

*"book The Middle East restaurant in IN for noon"*

This is an actual sample from the `BookRestaurant` intent. We paired this with the question `Which restaurant?`, and fed it to GPT2-117M (117M represents the number of parameters - 117 million) as-is without any fine-tuning.

This is what I got back from the model -

```
book The Middle East restaurant in IN for noon
Which restaurant? 
. new east: . newg $
```

The prediction the model made was `. new east: . newg $`. While it is nonsensical, it is still commendable that the model managed to get one part of the restaurant's name `east` in its answer - without any actual training about restaurants or question answering, etc.

I intend to use BLEU (bilingual evaluation understudy) score as a metric for performance. This algorithm would be used to compare the generated answer with the actual answer.

## Logistics

### Updated Timeline
<!-- gantt
    title Timeline
    dateFormat  YYYY-MM-DD
    section Learning
    NLP, Language Models           :2019-03-01, 14d
    Neural Networks, Best Practices     :2019-03-13  , 13d
    section Data Prep
    Preparing data for training      :2019-03-17  , 12d
    section Fine Tuning and Experimentation
    For intents AddToPlaylist and BookRestaurant      :2019-03-27  , 24d
    For intents GetWeather and PlayMusic :2019-03-29  , 24d
    For intent RateBook      :2019-03-29 , 24d
    For intents SearchCreativeWork, SearchScreeningEvent      :2019-03-29  , 24d
     section Analysis and Report
     Progress Report              :2019-03-29, 7d
     Final written Report                :2019-04-15, 15d
     Analysing Results  :2019-04-15, 10d
     Poster :2019-04-25, 5d 
      -->

![New Timeline](timeline%20(2).svg)

## References and Citation

[1] “Better Language Models and Their Implications.” OpenAI, 14 Feb. 2019, https://openai.com/blog/better-language-models/.

[2] Coucke, Alice. “Benchmarking Natural Language Understanding Systems: Google, Facebook, Microsoft, Amazon, and Snips.” Medium, 2 June 2017, https://medium.com/snips-ai/benchmarking-natural-language-understanding-systems-google-facebook-microsoft-and-snips-2b8ddcf9fb19.

[3] Radford, A.; Wu, J.; Child, R.; Luan, D.; Amodei, D. & Sutskever, I. (2018), 'Language Models are Unsupervised Multitask Learners', .

[4] Radosavovic, Ilija et al. “Data Distillation: Towards Omni-Supervised Learning.” 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (2018): 4119-4128.

[5] McCann, Bryan et al. “The Natural Language Decathlon: Multitask Learning as Question Answering.” CoRR abs/1806.08730 (2018): n. Pag.

[6] Ruder, Sebastian and Jeremy Howard. “Universal Language Model Fine-tuning for Text Classification.” ACL (2018).

[7] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le. XLNet: Generalized Autoregressive Pretraining for Language Understanding. arXiv arXiv:1906.08237v1.

[8] Dai, A. M. and Le, Q. V. Semi-supervised sequence learning. In Advances in neural information processing systems, pp. 3079– 3087, 2015.

[9] Yogatama, Dani et al. “Learning and Evaluating General Linguistic Intelligence.” CoRRabs/1901.11373 (2019): n. Pag.

[10] Contribute to Snipsco/Nlu-Benchmark Development by Creating an Account on GitHub. 2016. Snips, 2019. GitHub, https://github.com/snipsco/nlu-benchmark.

[11] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. RoBERTa: A Robustly Optimized BERT Pretraining Approach. eprint arXiv:1907.11692.