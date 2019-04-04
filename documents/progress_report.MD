# Semi-supervised Natural Language Understanding

## Table of Contents

- [Semi-supervised Natural Language Understanding](#semi-supervised-natural-language-understanding)
  - [Table of Contents](#table-of-contents)
  - [Team](#team)
  - [Introduction](#introduction)
    - [**Semi-supervised** Natural Language Understanding:](#semi-supervised-natural-language-understanding)
    - [Semi-supervised **Natural Language Understanding**:](#semi-supervised-natural-language-understanding)
  - [Background](#background)
    - [Language Models](#language-models)
    - [Related work](#related-work)
  - [The Problem](#the-problem)
    - [Introduction](#introduction-1)
    - [Motivation](#motivation)
    - [Dataset](#dataset)
    - [Approach and Summary of Methods](#approach-and-summary-of-methods)
  - [Logistics](#logistics)
    - [Updated Timeline](#updated-timeline)
    - [Updated Division of Work](#updated-division-of-work)
  - [Reflections](#reflections)
    - [Differences and novelty](#differences-and-novelty)
  - [References and Citation](#references-and-citation)

## Team

- Team Name: Regrayshun
- Team Members:
  - Dhruv Dhamani (ddhamani@uncc.edu)
  - Saloni Gupta (sgupta38@uncc.edu)
  - Himanshu Sunil Dhawale (hdhawale@uncc.edu)
  - Bhavya Chawla (bchawla@uncc.edu)

## Introduction

Let's break down the project title in two for clarity -

### **Semi-supervised** Natural Language Understanding:
  
Machine learning approaches are tend to either be supervised or unsupervised.

  - A supervised approach as in we when have curated data with a clear expectation for what parts of the data are the *input* to the model and what part the model is supposed to predict/infer or present as an *output*.
  - In general, unsupervised approaches mean that the data we have has not been labelled, classified or categorized. Instead, we create some sort of a feedback mechanism for the model which helps it identify commonalities in the data and react based on the presence or absence of such commonalities in each new piece of data.
  
  When we say *semi-supervised* what we mean is that we're going to take pre-existing labelled, classified or categorized data - the kind of data one uses for *supervised learning*, and transform and curate it into unlabelled data - the kind of data one uses for *unsupervised learning* - with the transformation or curation being done *on the basis of the labels of our pre-existing labelled data*, and then use this curated data for training our model.

  **Why would anyone do such a thing?** 
  
  Well, the motivation is simple. One could argue that as a general rule, there is a lot more unlabelled data in existence than labelled data. Thereby, if one creates a machine learning system that learns by use of unlabelled data, it is always going to have more data to learn from than a system that is based on learning from labelled data.

  And in most cases, the more data you have, the better machine learning systems can learn. Thusly, by transforming and curating labelled data for supervised learning approaches into unlabelled data for unsupervised learning approaches, we also manage to increase the available data for learning manifold; assuming the area of application of said machine learning system have availability of unlabelled data that can be learned from, and satisfactory feedback mechanisms for unsupervised learning to take place.

### Semi-supervised **Natural Language Understanding**:
  
  Natural language understanding or Natural language inference is a subtopic of natural-language processing in artificial intelligence that deals with machine reading comprehension. It can be thought of as what happens after natural language processing (NLP) - if a computer was presented with the sentence -
    
    This is a sentence.

  The results of performing NLP techniques and algorithms on this sentence would give us information about what the individual words in the sentence are relative to the grammar of that language, and what the relationship between the words in the sentence is. It would look something like -

  ![Example of spacy's NLP](spacy-viz.svg)

  Now taking this set of information produced by NLP algorithms and getting a machine to _comprehend_ or _understand_ what the sentence means is what Natural Language Understanding/Inference is.

  Our above discussion of the motivation behind doing "semi-supervised" learning lead us to the conclusion that semi-supervised learning might be a good thing to do in the following scenarios -

  - when there is availability of large amounts of raw data that can be learned from
  - when there exist satisfactory feedback mechanisms to facilitate unsupervised learning
  - (and also when labelled data can be transformed and curated into labelled data without a whole lot of hiccups)

  One field that satisfies all of these requirements is Natural Language processing -

  1. Availability of large amounts of raw data - Large amounts of written down, digital representations of language, i.e. text, is basically what the internet is. I don't think more needs to be said here.
  2. Existence of satisfactory feedback mechanisms to facilitate unsupervised learning - Academia has been doing unsupervised learning in the field of Natural Language Processing for years. The process in which word vectors and language models are learnt - the process of capturing meaning of words in a vector of floating point numbers by providing lots of examples of how the words are used in the natural language *is* unsupervised learning.
  3. Ease of transformation and curation of labelled data into unlabelled data - more on this in the review of related work and the approach and summary of methods sections below.

## Background

### Language Models

Traditionally, language models refer to statistical language models which can be thought of as a probability distribution over a sequences of words. Given a sequence of words (sentence, sentences, etc.), a statistical language model tells us the probability of that sequence of words existing in this language.

For instance, an English language model would assign `I love dogs` a higher probability than `I dogs love`. 

**Why is this important? **

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
A. I love dogs so I got one.
B. I love dogs so Canada sucks.
```

What sort of score would a perfect, ideal language model assign to these sentences? They both seem grammatically correct, so the scores of both would probably be on the higher side. But which sentence is more _natural_? Probably the same sentence which is more sensible?

Let's read sentence `A` again. Getting a dog because you love dogs is a sensible thing to do. And at first glance `B` is a nonsensical sentence. But then again, we don't know the context of sentence `B`. Maybe the sentence is from an article about lawmakers in Canada proposing a bill that demands mandatory neutering of all dogs. In that case, `B` is a sensible sentence too.

But `A` is still more *natural* or more *sensible* than `B`, since `A` would make sense in most contexts, while `B` only makes sense in some contexts.

What we're trying to argue is that if language models are good enough, they can also be thought of as *Sensibility Models* or at least *Common Sense Models*.

Even if you do agree with everything that was just said, you would still doubt the feasibility or practicality of training a model that was actually a good Sensibility model or Common Sense Model. These are good doubts to have, but thankfully there are researchers who have done work to support this way of thinking.

> Neural language models are different than statistical language models in the sense that the task neural language models are trained to do is to predict the next word after a sequence of words or predict the missing words in a sequence. 
> 
> The word that the neural language model predicts is the word which when added to the given input sequence, gives us a sequence with the highest probability of occurring in the language. A subtle but important difference. 
> 
> Neural language models can also thusly generate language by predicting the next word again and again. For instance, given a sequence `I love dogs so I got`, a neural language model might generate the next few words as `a dog`, etc. For the rest of the document whenever a language model is mentioned we refer to it being as something that predicts the next word(s), rather than something that gives a probability score for the sequence of input words.

### Related work

- **[The Natural Language Decathlon: Multitask Learning as Question Answering](https://arxiv.org/abs/1806.08730):**

  The focus of this paper was on introducing a new benchmark for measuring the performance of NLP models. They presented MQAN model for simple question answering which capitalizes on questions with the help of a multi-pointer-generator decoder. They demonstrated how labelled data can be used to train a language model to perform multiple tasks by casting all tasks as question-answers over a context. We are using this concept to understand the question answer mechanism in order to make the model extract the correct query to perform slot filling.

  However, the MQAN model this paper used takes the questions and the context as inputs separately, while we do not want to tell the model which part of the sequence is the question and which part is the context. 

  | ![MQAN model](MQAN.png) | 
  |:--:| 
  | The MQAN model - Questions and context are passed separately. |

- **[Learning and Evaluating General Linguistic Intelligence](https://arxiv.org/abs/1901.11373):**

  This paper defines general linguistic intelligence as the ability to reuse previously acquired knowledge about a language’s lexicon, syntax, semantics, and pragmatic conventions to adapt to new tasks quickly. Using this definition, they analyze state-of-the-art natural language understanding models and perform experiments to evaluate them against these criteria through a series of experiments that assess the task-independence of the knowledge being acquired by the learning process. The results show that while the field has made impressive progress in terms of model architectures that generalize to many tasks, these models still require a lot of in-domain training examples (fine tuning, training task-specific modules), and are prone to catastrophic forgetting. Moreover, they find that far from solving general tasks (e.g., document question answering), the models are overfitting to the quirks of particular datasets (like SQuAD).

  The authors also write that they believe *"generative language models will drive progress on general linguistic intelligence"*. What they call general linguistic intelligence is conceptually not all that different from what we described as a model of Sensibility above.

- **[Language models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf):**

  This paper is where the feasibility and practicality of language models being *intelligent* is proven. The authors first construct a high-quality dataset of millions of human-curated web-pages called WebText. They then proceed to train a modified version of the original Generative Pre-trained Transformer (GPT), called Generative Pre-Trained Transformer 2 on WebText, and demonstrate how the model - without any explicit supervision - achieves state of art results on 7 out of 8 tested language modeling datasets in a zero-shot setting but still underfits WebText.

  While the authors tested the performance of GPT2 in a zero-shot setting, we intend to re-use the smallest GPT2 model released by the authors and attempt to fine-tune it by casting our labelled NLU dataset as question-answer pairs in natural language; more on this later.

- **[Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146):**

  This paper is directed towards understanding how lack of knowledge about fine tuning can be a hindrance. The researchers have presented various novel fine tuning techniques to achieve effective results like discriminative fine-tuning, target task classifier fine-tuning, etc. They have also defined a Universal Language Model Fine-Tuning(ULMFiT) method to make robust models. This method comprises of three stages, firstly, LM is trained on a general domain corpus for capturing general features of the language, then the LM is fine-tuned using discriminative fine-tuning and slanted triangular learning rates. Finally the classifier is fine-tuned on target task using gradual freezing. The ULMFiT was able to achieve state-of-the-art performance on widely used text classification tasks.   

- **[Exploring the limits of language modeling](https://arxiv.org/abs/1602.02410):**

  This research dedicatedly explored different techniques like character convolutional network, Long-Short term memory, etc to deal with challenges like corpora and vocabulary sizes and complex long term structure of language. They demonstrated how the state-of-art model performed significantly better using lesser number of parameters. They scaled RNN based language models on 1 billion word benchmark to outperform competing models including tuned N-grams. This study helped us understand the state-of-art model better and how language models can be further improved using correct softmax approximations with important sampling. We were also able to extend our understanding of the regularization concept much deeply.



## The Problem

### Introduction

We intend to use generative language models to solve Natural language Understanding (NLU) tasks, as described above. More so than solving we want to analyze how good generative language models can be at this task, and think about the best way to cast a labelled dataset into question-answer pairs to solve this task. The language model we'll use is GPT2 as described above.

### Motivation

As we've discussed above, recent work done is NLP proves that generative language models can indeed act as Sensibility Models, or act as though they're *intelligent*.

That being said, the work done does not explore how best to train a generative language model to do a certain NLP task; the three major inspirations for our work ([McCann et al. 2018](https://arxiv.org/abs/1901.11373), [Radford et al. 2019](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), and [Yogatama et al. 2019](https://arxiv.org/abs/1901.11373)) have focused on Multitask learning, and have made no special efforts in coercing language models to perform better at a certain task.

We intend to choose one task - Natural Language Understanding/Inference - and try to evaluate the performance of various ways of casting the labelled data into question-answer pairs for the generative language model to be trained on, and see which ways perform the best, and think about why it does so. More on this in the approach and summary of methods section.

### Dataset

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

### Approach and Summary of Methods

To understand the approach easily, let's consider a sample from the dataset -

*Book spot for three at Maid-Rite Sandwich Shop in Antigua and Barbuda*

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

The relevant slots that need to be filled here are `party_size_number`, and `restaurant_name`.

## Logistics

### Updated Timeline

### Updated Division of Work

## Reflections

### Differences and novelty

## References and Citation
