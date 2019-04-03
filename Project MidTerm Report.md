<h2 id="fine-tuning-language-models-to-solve-natural-language-understanding-tasks"><strong>Fine-tuning Language models to solve Natural Language Understanding tasks</strong></h2>
<p><strong>Team Name:</strong> Regrayshun</p>
<p><strong>Team Members</strong><br>
Bhavya Chawla(801081909)<br>
Dhruv Dhamani(801084292)<br>
Himanshu Dhawale(801084142)<br>
Saloni Gupta(801080992)</p>
<p><strong>Introduction</strong><br>
In everyday life we use Google’s <a href="http://API.ai">API.ai</a>, Amazon’s Alexa, Microsoft’s <a href="http://Luis.ai">Luis.ai</a> and other artificially intelligent assistants that are trained to understand our language and process it to perform tasks. So, taking an inspiration from these assistants, we are aiming to fine tune a semi-supervised natural language model that is capable of understanding human language without any prior knowledge.</p>
<p>Omni-Supervised learning was defined as a special regime of semi-supervised learning in which the learner exploits all available labeled data plus internet-scale sources of unlabeled data in a paper by Facebook AI Research (FAIR) in the 2017 paper Data Distillation: Towards Omni-Supervised Learning . A fancy name to give to something researchers in NLP have been doing for years, word embeddings have been sourced from internet-scale data, and then applied to several tasks achieving state-of-the-art results.</p>
<p>A paper “The Natural Language Decathlon: Multitask Learning as Question Answering” has demonstrated how labelled data can be used to train a language model to perform multiple tasks by casting all tasks as question-answers over a context. Another another recent paper Language Models are Unsupervised Multitask Learners by researchers at OpenAI has shown how better quality data, and a more complex transformer based architecture results in a model that can achieve state-of-the-art results without any fine tuning whatsoever. We will be using these two papers as our basis for fine tuning the model to have a more efficient and effective language prediction model.</p>
<p>While the researchers at OpenAI made no attempts at fine tuning the GPT2 on various tasks - the whole point of the paper was that language models trained with quality data can achieve competitive results on various tasks without any fine tuning. However, we couldn’t help but be very excited about finding out how such a model would perform with fine tuning, considering that, to the best of our knowledge, there has never been any language model trained with data of this quality, and scale without the data being bastardized by any harsh pre-processing. So, that is what we plan on doing. Utilizing the smallest pretrained GPT2 model released by OpenAI, we would be fine tuning the model and evaluating its performance on either the open, crowd-sourced NLU benchmark by <a href="http://Snips.ai">Snips.ai</a>, or the NLU Evaluation Corpora (Braun et al.), whichever proves to be easier to work with. We’ll also be experimenting with the use of data augmentation in the question-context-answer format proposed by McCann et al., by paraphrasing the questions and answers, which we hypothesize will result in a model that generalizes better.</p>
<p><strong>The Problem</strong><br>
Whenever one person asks another person a question, it is important for the listener to extract the query from the sentence. Similarly for a language processing model to answer a question, the most important task is to extract the query that the user is making. In the world of Natural Language Processing, this problem is referred to as slot filling. As our language model uses no prior knowledge about the context, it aims to fill these slots with the correct answer after understanding the query asked. For example, if the user asks, ‘Which is the most famous restaurant in the city?’, the language model should be capable enough to understand the grammar of the sentence to identify the important aspects. We will be evaluating the performance based on the metrics we learned in the class, precision and recall. Precision measures how exact are the attributes extracted and recall measures the amount of existing attributes that are recovered by the model.</p>
<p><strong>Dataset</strong><br>
The dataset that we are using for our project has been sourced from a company named Snips. We are dealing with seven intents namely, AddToPlaylist, BookRestaurant, GetWeather, PlayMusic, RateBook, SearchCreativeWork and SearchScreeningEvent. Each intent comprises of more than 2000 queries that have been generated with crowdsourced method.</p>
<p><strong>Model</strong><br>
Explain the fine tuned model and how it works.</p>
<p><strong>Literature Survey</strong><br>
At least 10 examples</p>
<p><strong>Method</strong><br>
Describe how are you fine tuning it.</p>
<p><strong>Updated Plan</strong></p>
<p><strong>Reflection of Feedback</strong></p>
<p><strong>What we have achieved</strong></p>
<p><strong>Future Scope</strong></p>
<p><img src="https://lh4.googleusercontent.com/BfpEeDUIgI-IRx-5QInO0JNBpB5_eQkDwIIq4jszusSaFI6UfWVPQGo8HhGywcZUdC5avxdYAYpzMPxUdiE5EJqHl_H8RV-A5EhUMu70lHqTDk0ffD9n0OhB_8m3eL3hghyB3oMV" alt=""></p>
