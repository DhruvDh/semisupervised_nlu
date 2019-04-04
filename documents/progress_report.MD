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
  - [Problem Statement](#problem-statement)
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

## Problem Statement

### Motivation

### Dataset

### Approach and Summary of Methods

## Logistics

### Updated Timeline

### Updated Division of Work

## Reflections

### Differences and novelty

## References and Citation
