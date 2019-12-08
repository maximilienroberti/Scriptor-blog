---
title: "Fastai with \U0001F917Transformers (BERT, RoBERTa, XLNet, XLM, DistilBERT)"
description: >-
  A tutorial to implement state-of-the-art NLP models with Fastai for Sentiment
  Analysis
date: 2019-12-08T20:42:58.842Z
feature_image: /images/fastai-transformers-1.png
tags:
  - NLP
  - Machine Learning
layout: post
---
> A tutorial to implement state-of-the-art NLP models with Fastai for Sentiment Analysis

In early 2018, [Jeremy Howard](undefined) (co-founder of fast.ai) and [Sebastian Ruder](undefined) introduced the [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf) (ULMFiT) method. ULMFiT was the first **Transfer Learning** method applied to NLP. As a result, besides significantly outperforming many state-of-the-art tasks, it allowed, with only 100 labeled examples, to match performances equivalent to models trained on 100× more data.

<!--more-->

{% include image_caption.html imageurl="https://cdn-images-1.medium.com/max/3038/0*HUhpxwRcyNFEXNNd" title="Apple Super" 
caption="ULMFiT requires less data than previous approaches. (Howard and Ruder, ACL 2018)" %}

```python
from sklearn.ensemble import RandomForestClassifier 
# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100,                                                     
                               bootstrap = True,                              
                                max_features = 'sqrt')
# Fit on training data
model.fit(train, train_labels)
```

![](https://cdn-images-1.medium.com/max/3038/0*HUhpxwRcyNFEXNNd)*(ULMFiT requires less data than previous approaches.)*

The first time I heard about ULMFiT was during a [fast.ai course](https://course.fast.ai/videos/?lesson=4) given by Jeremy Howard. He demonstrated how it is easy — thanks to the fastai library — to implement the complete ULMFiT method with only a few lines of codes. In his demo, he used an AWD-LSTM neural network pre-trained on Wikitext-103 and get rapidly state-of-the-art results. He also explained key techniques — also demonstrated in ULMFiT — to fine-tune models like **Discriminate Learning Rate**, **Gradual Unfreezing** or **Slanted Triangular Learning Rates**.

Since the introduction of ULMFiT, **Transfer Learning** became very popular in NLP and yet Google (BERT, Transformer-XL, XLNet), Facebook (RoBERTa, XLM) and even OpenAI (GPT, GPT-2) begin to pre-train their own model on very large corpora. This time, instead of using the AWD-LSTM neural network, they all used a more powerful architecture based on the Transformer (cf. [Attention is all you need](https://arxiv.org/abs/1706.03762)).

Although these models are powerful, fastai do not integrate all of them. Fortunately, [Hugging Face](https://huggingface.co/) 🤗 created the well know [transformers library](https://github.com/huggingface/transformers). Formerly known as pytorch-transformers or pytorch-pretrained-bert, this library brings together over 40 state-of-the-art pre-trained NLP models (BERT, GPT-2, RoBERTa, CTRL…). The implementation gives interesting additional utilities like tokenizer, optimizer or scheduler.

```
jdklmqsjdfdjslkd
```

`Hello`

[**huggingface/transformers**
_State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch 🤗 Transformers (formerly known as…_github.com](https://github.com/huggingface/transformers)

The transformers library can be self-sufficient but incorporating it within the fastai library provides simpler implementation compatible with powerful fastai tools like **Discriminate Learning Rate**, **Gradual Unfreezing** or **Slanted Triangular Learning Rates**. The point here is to allow anyone — expert or non-expert — to get easily state-of-the-art results and to “make NLP uncool again”.

It is worth noting that integrating the Hugging Face transformers library in fastai has already been demonstrated in:

* _Keita Kurita_’s article [A Tutorial to Fine-Tuning BERT with Fast AI](https://mlexplained.com/2019/05/13/a-tutorial-to-fine-tuning-bert-with-fast-ai/) which makes pytorch_pretrained_bert library compatible with fastai.
* [Dev Sharma](undefined)’s article [Using RoBERTa with Fastai for NLP](https://medium.com/analytics-vidhya/using-roberta-with-fastai-for-nlp-7ed3fed21f6c) which makes pytorch_transformers library compatible with fastai.

Although these articles are of high quality, some part of their demonstration is not anymore compatible with the last version of transformers.

## 🛠 Integrating transformers with fastai for multiclass classification

Before beginning the implementation, note that integrating transformers within fastai can be done in multiple ways. For that reason, I brought — what I think are — the most generic and flexible solutions. More precisely, I tried to make the minimum modification in both libraries while making them compatible with the maximum amount of transformer architectures. However, if you find a clever way to make this implementation, please let us know in the comment section!

A Jupiter Notebook version of this tutorial is available on this [Kaggle kernel](https://www.kaggle.com/maroberti/fastai-with-transformers-bert-roberta).

### Libraries Installation

First, you will need to install the fastai and transformers libraries. To do so, just follow the instructions [here](https://github.com/fastai/fastai/blob/master/README.md#installation) and [here](https://github.com/huggingface/transformers#installation).

For this demonstration, I used Kaggle which already has the fastai library installed. So I just installed transformers with the command :

```
pip install transformers
```

The versions of the libraries used for this demonstration are fastai 1.0.58 and transformers 2.1.1.

### 🎬 The example task

The chosen task is a multi-class text classification on [Movie Reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/overview).

The [dataset](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data) and the respective [Notebook](https://www.kaggle.com/maroberti/fastai-with-transformers-bert-roberta) of this article can be found on Kaggle.
[**Sentiment Analysis on Movie Reviews**
_Download Open Datasets on 1000s of Projects + Share Projects on One Platform. Explore Popular Topics Like Government…_www.kaggle.com](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data)

For each text movie review, the model has to predict a label for the sentiment. We evaluate the outputs of the model on classification _accuracy_. The sentiment labels are:

0 →Negative
1 →Somewhat negative
2 →Neutral
3 →Somewhat positive
4 →Positive

The data is loaded into a DataFrame using pandas.

 <iframe src="https://medium.com/media/b728f8c3d1d71536b2705995cb54c549" frameborder=0></iframe>

### Main transformers classes

In transformers, each model architecture is associated with 3 main types of classes:

* A **model class** to load/store a particular pre-train model.
* A **tokenizer class** to pre-process the data and make it compatible with a particular model.
* A **configuration class** to load/store the configuration of a particular model.

For example, if you want to use the BERT architecture for text classification, you would use [BertForSequenceClassification](https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification) for the **model class**, [BertTokenizer](https://huggingface.co/transformers/model_doc/bert.html#berttokenizer) for the **tokenizer class** and [BertConfig](https://huggingface.co/transformers/model_doc/bert.html#bertconfig) for the **configuration class**.

Later, you will see that those classes share a common class method from_pretrained(pretrained_model_name, ...). In our case, the parameter pretrained_model_name is a string with the shortcut name of a pre-trained model/tokenizer/configuration to load, e.g bert-base-uncased. We can find all the shortcut names in the transformers documentation [here](https://huggingface.co/transformers/pretrained_models.html#pretrained-models).

In order to switch easily between classes — each related to a specific model type — I created a dictionary that allows loading the correct classes by just specifying the correct model type name.

 <iframe src="https://medium.com/media/f0393118fcf01d72f2559e52952d3c7a" frameborder=0></iframe>

It is worth noting that in this case, we use the transformers library only for a _multi-class text classification_ task. For that reason, this tutorial integrates only the transformer architectures that have a model for sequence classification implemented. These model types are :

* BERT (from Google)
* XLNet (from Google/CMU)
* XLM (from Facebook)
* RoBERTa (from Facebook)
* DistilBERT (from Hugging Face)

However, if you want to go further — by implementing another type of model or NLP task — this tutorial still an excellent starter.

### Data pre-processing

To match pre-training, we have to format the model input sequence in a specific format. 
To do so, you have to first **tokenize** and then **numericalize** the texts correctly.
The difficulty here is that each pre-trained model, that we will fine-tune, requires exactly the same specific pre-process — **tokenization** & **numericalization** — than the pre-process used during the pre-train part.
Fortunately, the **tokenizer class** from \*\*\*\*transformers provides the correct pre-process tools that correspond to each pre-trained model.

In the fastai library, data pre-processing is done automatically during the creation of the DataBunch. 
As you will see in the DataBunch implementation part, the **tokenizer** and the **numericalizer** are passed in the processor argument under the following format :

```
processor = [TokenizeProcessor(tokenizer=tokenizer,…), NumericalizeProcessor(vocab=vocab,…)]
```

Let’s first analyze how we can integrate the transformers tokenizer within the TokenizeProcessor function.

**Custom tokenizer**

This part can be a little confusing because a lot of classes are wrapped in each other and with similar names.
To resume, if we look attentively at the fastai implementation, we notice that :

1. The [TokenizeProcessor object](https://docs.fast.ai/text.data.html#TokenizeProcessor) takes as tokenizer argument a Tokenizer object.
2. The [Tokenizer object](https://docs.fast.ai/text.transform.html#Tokenizer) takes as tok_func argument a BaseTokenizer object.
3. The [BaseTokenizer object](https://docs.fast.ai/text.transform.html#BaseTokenizer) implement the function tokenizer(t:str) → List\[str] that takes a text t and returns the list of its tokens.

Therefore, we can simply create a new class TransformersBaseTokenizer that inherits from BaseTokenizer and overwrite a new tokenizer function.

 <iframe src="https://medium.com/media/a198d6c1682e1f00c42cb321223f7bfa" frameborder=0></iframe>

In this implementation, be careful about 3 things:

1. As we are not using RNN, we have to limit the _sequence length_ to the model input size.
2. Most of the models require _special tokens_ placed at the beginning and end of the sequences.
3. Some models like RoBERTa require a _space_ to start the input string. For those models, the encoding methods should be called with add_prefix_space set to True.

Below, you can find the resume of each pre-process requirement for the 5 model types used in this tutorial. You can also find this information on the [Hugging Face documentation](https://huggingface.co/transformers/) in each model section.

>  BERT: \[CLS] + tokens + \[SEP] + padding
>  DistilBERT: \[CLS] + tokens + \[SEP] + padding
>  RoBERTa: \[CLS] + prefix_space + tokens + \[SEP] + padding
>  XLM: \[CLS] + tokens + \[SEP] + padding
>  XLNet: padding + \[CLS] + tokens + \[SEP]

It is worth noting that we don’t add padding in this part of the implementation. 
As we will see later, fastai manage it automatically during the creation of the DataBunch.

**Custom Numericalizer**

In fastai, [NumericalizeProcessor object](https://docs.fast.ai/text.data.html#NumericalizeProcessor) takes as vocab argument a [Vocab object](https://docs.fast.ai/text.transform.html#Vocab).
From this analyze, I suggest two ways to adapt the fastai **numericalizer**:

1. You can like described in [Dev Sharma’s article](https://medium.com/analytics-vidhya/using-roberta-with-fastai-for-nlp-7ed3fed21f6c) (Section 1. _Setting Up the Tokenizer_), retrieve the list of tokens and create a Vocab object.
2. Create a new class TransformersVocab that inherits from Vocab and overwrite numericalize and textify functions.

Even if the first solution seems to be simpler, Transformers does not provide, for all models, a straightforward way to retrieve his list of tokens. 
Therefore, I implemented the second solution, which runs for each model type. 
It consists of using the functions convert_tokens_to_ids and convert_ids_to_tokens in respectively numericalize and textify.

 <iframe src="https://medium.com/media/f622055d96c6fca0d965e2fcc2124c68" frameborder=0></iframe>

**Custom processor**

Now that we have our custom **tokenizer** and **numericalizer**, we can create the custom **processor**. Notice we are passing the include_bos = False and include_eos = False options. This is because fastai adds its own special tokens by default which interferes with the \[CLS] and \[SEP] tokens added by our custom tokenizer.

 <iframe src="https://medium.com/media/42ad6c6b8cfb2ec92d3c3fbed8f5575a" frameborder=0></iframe>

**Setting up the DataBunch**

For the DataBunch creation, you have to pay attention to set the processor argument to our new custom processor transformer_processor and manage correctly the padding.

As mentioned in the Hugging Face documentation, BERT, RoBERTa, XLM, and DistilBERT are models with absolute position embeddings, so it’s usually advised to pad the inputs on the right rather than the left. Regarding XLNET, it is a model with relative position embeddings, therefore, you can either pad the inputs on the right or on the left.

 <iframe src="https://medium.com/media/94465a3efbc8dc9e9626898bcf4a2e6d" frameborder=0></iframe>

### Custom model

As mentioned [here](https://github.com/huggingface/transformers#models-always-output-tuples), every model's forward method always outputs a tuple with various elements depending on the model and the configuration parameters. In our case, we are interested to access only to the logits. 
One way to access them is to create a custom model.

 <iframe src="https://medium.com/media/03849baa5f99f03c02192addace559b9" frameborder=0></iframe>

To make our transformers adapted to multiclass classification, before loading the pre-trained model, we need to precise the number of labels. To do so, you can modify the config instance or either modify like in [_Keita Kurita_’s article](https://mlexplained.com/2019/05/13/a-tutorial-to-fine-tuning-bert-with-fast-ai/) (Section: _Initializing the Learner_) the num_labels argument.

 <iframe src="https://medium.com/media/cdcea6fa7f2f625f64eb0bc3d56cab4d" frameborder=0></iframe>

### Learner : Custom Optimizer / Custom Metric

In pytorch-transformers, Hugging Face had implemented two specific optimizers — BertAdam and OpenAIAdam — that have been replaced by a single AdamW optimizer.
This optimizer matches Pytorch Adam optimizer Api, therefore, it becomes straightforward to integrate it within fastai.
Note that for reproducing BertAdam specific behavior, you have to set correct_bias = False.

 <iframe src="https://medium.com/media/bd28c7255eb5ee1fcb89f5b9d7350e31" frameborder=0></iframe>

### Discriminative Fine-tuning and Gradual unfreezing

To use **Discriminative Learning Rate** and **G_radual Unfreezing_**, fastai provides one tool that allows to “split” the structure model into groups. An instruction to perform that “split” is described in the fastai documentation [here](https://docs.fast.ai/basic_train.html#Discriminative-layer-training).

Unfortunately, the model architectures are too different to create a unique generic function that can “split” all the model types in a convenient way. Thereby, you will have to implement a custom “split” for each different model architecture.

For example, if we use the DistilBERT model and that we observe his architecture by making print(learner.model). We can decide to divide the model in 8 blocks :

* 1 Embedding
* 6 transformer
* 1 classifier

In this case, we can split our model in this way:

 <iframe src="https://medium.com/media/22b8ac3c3ea37ae7c1c12323bc8b93e8" frameborder=0></iframe>

Note that I didn’t found any document that has studied the influence of **Discriminative Learning Rate** and **Gradual Unfreezing** or even **Slanted Triangular Learning Rates** with transformer architectures. \*\*\*\*Therefore, using these tools does not guarantee better results. If you found any interesting documents, please let us know in the comment.

### Train

Now we can finally use all the fastai build-in features to train our model. Like the ULMFiT method, we will use **Slanted Triangular Learning Rates, Discriminate Learning Rate** and **gradually unfreeze** the model.

Therefore, we first freeze all the groups but the classifier with :

```
learner.freeze_to(-1)
```

For **Slanted Triangular Learning Rates** you have to use the function fit_one_cycle. For more information, please check the fastai documentation [here](https://docs.fast.ai/callbacks.one_cycle.html).

To use our fit_one_cycle we will need an optimum learning rate. We can find this learning rate by using a learning rate finder, which can be called by using [lr_find](https://docs.fast.ai/callbacks.lr_finder.html#callbacks.lr_finder). Our graph would look something like this:

![](https://cdn-images-1.medium.com/max/2000/1*p0EmMlM1i5AG-OQ3uJoAog.png)

We will pick a value a bit before the minimum, where the loss still improves. Here 2x10–3 seems to be a good value.

```
learner.fit_one_cycle(1,max_lr=2e-03,moms=(0.8,0.7))
```

The graph of the loss would look like this:

![](https://cdn-images-1.medium.com/max/2000/1*dJyzyjsaMaVUhea9beSkCA.png)

We then unfreeze the second group and repeat the operations until all the groups are unfrozen. If you want to use **Discriminative Learning Rate** you can use slice as follow :

 <iframe src="https://medium.com/media/deb0695c2ed4ad5f70721b38c182178d" frameborder=0></iframe>

To unfreeze all the groups, use learner.unfreeze() .

### Creating prediction

Now that we have trained the model, we want to generate predictions from the test dataset.

As specified in _Keita Kurita_’s [article](https://mlexplained.com/2019/05/13/a-tutorial-to-fine-tuning-bert-with-fast-ai/), as the function get_preds does not return elements in order by default, you will have to resort the elements into their correct order.

<script src="https://gist.github.com/maximilienroberti/c474f55774a107fd02ef7f531e3bbceb.js"></script>

In the [Kaggle example](https://www.kaggle.com/maroberti/fastai-with-transformers-bert-roberta), without playing too much with the parameters, we get a Public Score of 0.70059, which leads us to the 5th position on the leaderboard!

## 📋Conclusion

In this article, I explain how to combine the transformers library with the beloved fastai library. It aims to make you understand where to look and modify both libraries to make them work together. Likely, it allows you to use **Slanted Triangular Learning Rates**, **Discriminate Learning Rate** and even **Gradual Unfreezing**. As a result, without even tunning the parameters, you can obtain rapidly state-of-the-art results.

This year, the transformers became an essential tool for NLP. Because of that, I think that pre-trained transformers architectures will be integrated soon to future versions of fastai. Meanwhile, this tutorial is a good starter.

I hope you enjoyed this first article and found it useful. 
Thanks for reading and don’t hesitate in leaving questions or suggestions.

I will keep writing articles on NLP so stay tuned!

## 📑 References

\[1] Hugging Face, Transformers GitHub (Nov 2019), <https://github.com/huggingface/transformers>

\[2] Fast.ai, Fastai documentation (Nov 2019), <https://docs.fast.ai/text.html>

\[3] Jeremy Howard & Sebastian Ruder, Universal Language Model Fine-tuning for Text Classification (May 2018), <https://arxiv.org/abs/1801.06146>

\[4] Keita Kurita , [A Tutorial to Fine-Tuning BERT with Fast AI](https://mlexplained.com/2019/05/13/a-tutorial-to-fine-tuning-bert-with-fast-ai/) (May 2019)

\[5](undefined), [Using RoBERTa with Fastai for NLP](https://medium.com/analytics-vidhya/using-roberta-with-fastai-for-nlp-7ed3fed21f6c) (Sep 2019)
