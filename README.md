# velarization-classifier
Fine-tuning Wav2Vec2 on some data I recorded and labelled to classify suprasegemental velarization in Lishan Didan

### Introduction: 
It is crucial that neural speech models can extract relevant linguistic information for all inputs to be able to use them effectively. However, learning to extract that can be very difficult for a model when not enough training data is available. Many languages have unique linguistic features that are not shared by many languages around the world. This can pose a problem to low-resource languages, where collecting enough data is extremely difficult. Pre-trained models have been a boon for this problem - by training models on high resource languages, models can be tuned to the many linguistic properties that are shared across many languages. This project will attempt to demonstrate how to utilize such models to teach a neural classifier to distinguish velarized and plain words in Lishan Didan, a relatively rare phonological distinction in a critically low resource language.

Previous research in this area has shown that pre-trained speech models do generally have good cross-lingual transfer. One way this has been shown is for audio transcription: Wav2Vec2Phoneme demonstrated zero-shot cross-lingual transfer learning by fine-tuning a multilingually pretrained wav2vec 2.0 model to transcribe unseen languages (Xu et. al, 2021). The linguistic information learned by a model from certain languages was able to transfer over when parsing others. The specific linguistic feature that will be investigated – suprasegmental velarization in Neo-Aramaic – has been documented as a semantically distinguishing feature with an abundance of minimal pairs by Tsereteli as a general trend in Semitic languages, and by Georey Khan in his documentation of Lishan Didan (Tsereteli, 1982) (Khan, 2008). Khan’s work also provides a well-documented latin-based orthography that will be used. Lastly, working with very low amounts of data can benefit from data augmentation, as shown in a recent Google paper that documents SpecAugment. SpecAugment is a method of augmenting spectrogram data, and consists of 3 methods: time stretch (slowing down/speeding up the audio), frequency masking (removing frequencies from the spectrogram), and time masking (removing segments of time from the audio) (Park et. al, 2019). This will allow training on very low amounts of data.

### Hypothesis: 
Suprasegmental Velarization in Neo-Aramaic can be taught as a feature to
an NLP audio classification model.

More specifically, a pre-trained neural speech classifier will be able to distinguish
between velarized and non-velarized words in Lishan Didan with small amounts of data
for fine-tuning.

### Methods:
Collect data: 
- Record velarized and non-velarized words spoken by speakers of Lishan Didan (in this case I recorded them myself)
- Segregate recordings into separate audio files, labeled as velarized or non-velarized
- See Appendix A for all words used
Process data:
- Convert audio files to arrays with torch audio
- Augment the Data to 10 times its size
- Randomly applying frequency and time masking 5 different times for all files [adds x5 the data]
- Stretching the time by factors of 0.8, 0.9, 1.1, 1.2 for all files [adds x4 the data]
- Split files into train, validation, and test sets, and save to csv files.
- Fine tune facebook/wav2vec2-base from huggingface on the training/validation data
- Load the csv files into a dataset from huggingface’s dataset library
- Encode the data with a tokenizer from the model
- Train the model with the huggingface trainer
- hyperparameters: learning_rate=2e-5, batch_size=8, optimizer=AdamW, epochs=5, weight_decay=0.01, main_metric=f1
- training settings to decrease gpu usage: gradient_checkpointing=True,  fp16=True, gradient_accumulation_steps=4
- Evaluate the model on the test set 
- Make predictions on unseen data

### Materials: 
Recordings of velarized and non velarized words in Lishan Didan, with sets
of minimal pairs, and some that do not have a corresponding minimal pair (but may or may not be phonetically similar to some of the words used). Some vowels are different when words are velarized, but not all of them (/e/ & /i/ stay the same), so a mixture of both cases are included. See Appendix A for a full list of words recorded.

### Results: 
After training, the model was able to distinguish between velarized and non-velarized words in some capacity, though by no means perfectly. Here are the results for the validation and test sets after training, averaged over 4 runs:



|             |             | Accuracy    | Precision   | Recall      | F1          |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|Run 1|Validation|0.792|0.778|0.700|0.737|
||Test|0.725|0.886|0.517|0.653|
|Run 2|Validation|0.717|0.614|0.860|0.717|
||Test|0.692|0.695|0.683|0.689|
|Run 3|Validation|0.608|0.519|0.800|0.630|
||Test|0.700|0.684|0.520|0.591|
|Run 4|Validation|0.658|0.565|0.780|0.655|
||Test|0.817|0.759|0.820|0.788|
|**Average**|**Validation**|**0.694**|**0.649**|**0.785**|**0.685**|
||**Test**|**0.734**|**0.756**|**0.635**|**0.680**|

The accuracy around 70% indicates a model that is right more often than wrong. Further, it appears that there were some extraneous factors that may have been causing the model to make mistakes: a couple audios were consistently predicted incorrectly across runs.
Words containing [x q] gave the model difficulty. Since [x q] are guttural consonants, that may influence the rest of the word and the model’s ability to understand if the word is velarized. For example: qrawa, +yqərri, and +xyate were all consistently misclassified (across data augmentation methods as well).
It appears that it may be the case that frequency masking across lower frequencies may make this task more difficult, which would make sense since velarized words are generally pronounced at a lower frequency in Lishan Didan (Khan, 2008). However, there needs to be a further investigation into this, as it is difficult to tell if this truly is the case with the current methods.

The F1 score is a bit trickier to interpret, since there is no direct baseline to compare it to. In this case, it represents the average of the percentage of words predicted to be velarized that actually are (precision) and the percentage of all velarized words that were predicted as such (recall). These results indicate that the model is better than a blind guess at both of these. Moreover, in a well-balanced dataset such as this one (34/33 split), these results indicate a similar picture for F1 and accuracy, which is borne out for the most part.

### Conclusions: 
Given the low data size (75 words), that a model was able to be fine-tuned to be better than a coin toss on a relatively minor acoustic feature in an audio clip is quite impressive. Though by no means perfect, these results demonstrate that it is sometimes possible to use existing technologies and techniques to expand them to low-resource languages, when working with very small amounts of data. The results ultimately lend support to the idea that a pre-trained neural speech classifier can distinguish between velarized and non-velarized words in Lishan Didan with small amounts of data for fine-tuning. They may not show a model to be very good at this task, but they demonstrate the capability that with even a little more data, this is possible.

However, more work is needed to demonstrate unassailably that the results indicate pre-trained models are good for this task, not simply able to do it. A strong baseline needs to be established using linear regression or other naive method. This would lend credibility to the idea that pre-trained neural models are relatively very effective, with regards to the other options out there. It is also important to test his out on many different models. A very big shortcoming of this paper is the maximum GPU allotment of 4GB. With that limitation, Wav2Vec 2.0 XLSR, the multilingual version of Wav2Vec 2.0, could not be used, as it requires more memory than 4GB to train. Using the XLSR version of Wav2Vec may have allowed for more cross-lingual transfer of information, which is especially important given the low amount of tuning data used.

### References:

Khan, G. (2008). The Jewish Neo-Aramaic Dialect of Urmi. United States: Gorgias Press.

Park, D. S., Chan, W., Zhang, Y., Chiu, C.-C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). Specaugment: A simple data augmentation method for automatic speech recognition. Interspeech 2019, 2613–2617. https://doi.org/10.21437/Interspeech.2019-2680

Tsereteli, K. (1982). On One Suprasegmental Phoneme in Modern Semitic. Journal of the
American Oriental Society, 102(2), 343–346. https://doi.org/10.2307/602533

Xu, Q., Baevski, A., & Auli, M. (2021, September 23). Simple and effective zero-shot
cross-lingual phoneme recognition. arXiv.org. https://doi.org/10.48550/arXiv.2109.11680


### Appendix A. 
All data recorded, total of 75 words (37 Velarized, 38 Non-Velarized).
Also in words.csv file.

|Non-Velarized/With Pair|Translation|Velarized/With Pair|Translation|Foil/Without Pair|Translation|
|---|---|---|---|---|---|
|drele|He placed|+drele|He scattered|srele|He did evil|
|nabyawa|She would prophesy|+nabyawa|She would swell|gabyawa|She would choose|
|saxulu|They are being generous|+saxulu|They are swimming|maxulu|They are playing (instruments)|
|šrila|Untie it!|+šrila|Judge it!|štila!|Drink it!|
|šatet|You (ms.) drink|+šatet|You (ms.) lie down|+šaret|You (ms.) judge|
|xliqi|They have become knotted|+xliqi|They have been created|+xlibi|They have won|
|yqərri|I became heavy|+yqərri|I dug|ytəwli|I sat|
|aqlew|His foot|+aqlew|His intelligence|+aska|antelope|
|ara|land|+ara|Situation, relations|+awa|inhabited|
|aslan|fundamentally|+aslan|lion|+arzan|cheap|
|bali|My mind|+bali|My child|beli|My house|
|bar|after|+bar|fruit (as in child)|babr|tiger|
|dawa|camel|+dawa|quarrel|qrawa|war|
|mara|owner|+mara|shovel|+nare|shouting|
|qari|My squash|+qari|The old woman|+gari|my roof|
|qora|grave|+qora|Sour grape|qoma|Height, stature|
|sətra|hole in a rock|+sətra|crack|+sətwa|winter|
|səwya|satiated|+səwya|rigid|+rəwya|great, big|
|šala|fever|+šala|load|šama|honeycomb|
|tora|bull|+tora|Torah|tara|door|
|wada|She is doing|+wada|A specific timing|+jada|wide street|
|xale|(He) is sweet|+xale|new (pl.)|+xasa|back (human)|
|xəlta|She has eaten|+xəlta|mistake|xleta|gift|
|xyare|He is looking|+xyare|cucumbers|+xyate|tailors|
|zabt|recording|+zabt|sequestering|ziwuǧ|soulmate|
