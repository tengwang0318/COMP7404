# WHAT I CAN DO: 

run the generating data and selecting data scripts respectively.

I have modified the author's code, since there are some issues need to fix. I have already set the number of generated data very small and could finish the whole process of generating and selecting in 2 minutes with CPU but requires lots of RAM!

Running methods could be found in the [demo.ipynb](demo.ipynb)

### Example

Both samples have the same text1 and different labels, so the model generates the different sentences based on the different labels.

![](pics/10.png)

Firstly,  pip install the packages.

```
!pip install transformers=4.24.0, torch, sentencepiece, tqdm, numpy, nltk
```



## One way 

just run 

```
!git clone https://github.com/tengwang0318/COMP7404.git
!cd SuperGen
!chmod +x run_gen.sh
!./run_gen.sh SST-2

```

then you could finish generating data and selecting generated data.

## Another way

```
!git clone https://github.com/tengwang0318/COMP7404.git
!cd SuperGen
```

run 

```
python gen_train_data.py --task SST-2 --label all --num_gen 10 --max_len 30 --print_res

```

and 

```
python gen_train_data.py --task SST-2 label all --save_dir temp_gen --print_res \
                         --num_gen 10 --max_len 30 --temperature 0.2
```

## the log:
[log file](log.log)

```
Namespace(pretrain_corpus_dir='pretrain_corpus/wiki_long.txt', task='MRPC', label='all', model_type='ctrl', model_name_or_path='ctrl', temperature='0', repetition_reward=None, repetition_penalty=None, p=1.0, k=10, seed=42, no_cuda=False, fp16=False, num_gen=10, max_len=40, save_dir='temp_gen', print_res=True)
11/03/2023 00:50:15 - WARNING - __main__ - device: cpu, n_gpu: 0, 16-bits training: False
  0%|                                                                                                                                           | 0/10 [00:00<?, ?it/s]{'text1': "From 2000 until 2004 she led a Stability Pact for South Eastern Europe task force against human trafficking, and in 2004 was appointed as the Organization for Security and Co-operation in Europe's special representative for the issue.", 'text2': 'She was the representative for human trafficking, and in 2004 led a task force against trafficking as special representative of the Organization for Security and Co-operation in Europe (OSCE).', 'label': 'entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'In other words,', 'score': -0.4007759690284729}
{'text1': "From 2000 until 2004 she led a Stability Pact for South Eastern Europe task force against human trafficking, and in 2004 was appointed as the Organization for Security and Co-operation in Europe's special representative for the issue.", 'text2': 'She was the representative for human trafficking, and in 2004 led a task force against trafficking as special representative of the Organization for Security and Co-operation in Europe (OSCE).', 'label': 'entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'In other words,', 'score': -0.4007759690284729}
 10%|█████████████                                                                                                                      | 1/10 [00:16<02:29, 16.66s/it]None
 20%|██████████████████████████▏                                                                                                        | 2/10 [00:28<01:49, 13.70s/it]{'text1': "On December 8, 2019 after Norvell's departure to Florida State, Silverfield served as the interim head coach before being promoted to head coach on December 13, 2019.", 'text2': "Silverfield served as the interim head coach after Norvell's departure to Florida State, before being promoted on December 13, 2019.", 'label': 'entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'In other words,', 'score': -0.16363264620304108}
{'text1': "On December 8, 2019 after Norvell's departure to Florida State, Silverfield served as the interim head coach before being promoted to head coach on December 13, 2019.", 'text2': "Silverfield served as the interim head coach after Norvell's departure to Florida State, before being promoted on December 13, 2019.", 'label': 'entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'In other words,', 'score': -0.16363264620304108}
 30%|███████████████████████████████████████▎                                                                                           | 3/10 [00:39<01:28, 12.62s/it]None
 40%|████████████████████████████████████████████████████▍                                                                              | 4/10 [00:49<01:10, 11.68s/it]{'text1': 'In the same year, Prince Gong displeased Empress Dowager Cixi when he strongly opposed her plan to rebuild the Old Summer Palace.', 'text2': 'He opposed the plan to rebuild Old Summer Palace.', 'label': 'entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'In other words,', 'score': -0.4058457016944885}
{'text1': 'In the same year, Prince Gong displeased Empress Dowager Cixi when he strongly opposed her plan to rebuild the Old Summer Palace.', 'text2': 'He opposed the plan to rebuild Old Summer Palace.', 'label': 'entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'In other words,', 'score': -0.4058457016944885}
 50%|█████████████████████████████████████████████████████████████████▌                                                                 | 5/10 [00:59<00:55, 11.05s/it]{'text1': 'Karstadt contributed a large sum of money towards the decoration of the station and was in return rewarded with direct access from the station to the store.', 'text2': 'The station was a direct access to Karstadt and from there to the store.', 'label': 'entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'In other words,', 'score': -0.5564016699790955}
{'text1': 'Karstadt contributed a large sum of money towards the decoration of the station and was in return rewarded with direct access from the station to the store.', 'text2': 'The station was a direct access to Karstadt and from there to the store.', 'label': 'entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'In other words,', 'score': -0.5564016699790955}
 60%|██████████████████████████████████████████████████████████████████████████████▌                                                    | 6/10 [01:11<00:44, 11.21s/it]{'text1': 'While in Drill Instructor status, both male and female DIs wear a World War I campaign hat with their service and utility uniforms.', 'text2': 'Both male and female DIs wear a World War I campaign hat with their service uniforms.', 'label': 'entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'In other words,', 'score': -0.14014530181884766}
{'text1': 'While in Drill Instructor status, both male and female DIs wear a World War I campaign hat with their service and utility uniforms.', 'text2': 'Both male and female DIs wear a World War I campaign hat with their service uniforms.', 'label': 'entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'In other words,', 'score': -0.14014530181884766}
 70%|███████████████████████████████████████████████████████████████████████████████████████████▋                                       | 7/10 [01:22<00:33, 11.17s/it]{'text1': 'The film won four Golden Eagle Awards (2020) for Best Motion Picture, Best Leading Actor (Alexander Petrov), Best Supporting Actor (Ivan Yankovsky), and Best Film Editing.', 'text2': 'It won four Golden Eagle Awards for Best Film and Best Leading Actor (Alexander Petrov), Best Supporting Actress (Ivan Yankovsky), and Best Editing.', 'label': 'entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'In other words,', 'score': -0.23435038328170776}
{'text1': 'The film won four Golden Eagle Awards (2020) for Best Motion Picture, Best Leading Actor (Alexander Petrov), Best Supporting Actor (Ivan Yankovsky), and Best Film Editing.', 'text2': 'It won four Golden Eagle Awards for Best Film and Best Leading Actor (Alexander Petrov), Best Supporting Actress (Ivan Yankovsky), and Best Editing.', 'label': 'entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'In other words,', 'score': -0.23435038328170776}
 80%|████████████████████████████████████████████████████████████████████████████████████████████████████████▊                          | 8/10 [01:32<00:21, 10.94s/it]{'text1': 'Its basin contains over 3000 rivers, of which 425 are longer than and 11 are longer than ; 1011 of those rivers directly flow into the Donets.', 'text2': 'The basin of the Donets contains over 3000 rivers and streams.', 'label': 'entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'In other words,', 'score': -0.5840206146240234}
{'text1': 'Its basin contains over 3000 rivers, of which 425 are longer than and 11 are longer than ; 1011 of those rivers directly flow into the Donets.', 'text2': 'The basin of the Donets contains over 3000 rivers and streams.', 'label': 'entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'In other words,', 'score': -0.5840206146240234}
 90%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉             | 9/10 [01:43<00:10, 11.00s/it]{'text1': 'He had also gone to seek medical attention for his failing health and settle a business deal with his Dutch associate.', 'text2': 'His health had gone to a very bad and dangerous place.', 'label': 'entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'In other words,', 'score': -1.1980963945388794}
{'text1': 'He had also gone to seek medical attention for his failing health and settle a business deal with his Dutch associate.', 'text2': 'His health had gone to a very bad and dangerous place.', 'label': 'entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'In other words,', 'score': -1.1980963945388794}
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [01:53<00:00, 11.40s/it]
saved to temp_gen/mrpc_entailment_10_sorted.json
  0%|                                                                                                                                           | 0/10 [00:00<?, ?it/s{'text1': "From 2000 until 2004 she led a Stability Pact for South Eastern Europe task force against human trafficking, and in 2004 was appointed as the Organization for Security and Co-operation in Europe's special representative for the issue.", 'text2': 'From 2003 to 2005 she served as the Special Representative of the Secretary-General on Human Rights Issues in Central Africa.', 'label': 'not_entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'Furthermore,', 'score': -1.0093516111373901}
{'text1': "From 2000 until 2004 she led a Stability Pact for South Eastern Europe task force against human trafficking, and in 2004 was appointed as the Organization for Security and Co-operation in Europe's special representative for the issue.", 'text2': 'From 2003 to 2005 she served as the Special Representative of the Secretary-General on Human Rights Issues in Central Africa.', 'label': 'not_entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'Furthermore,', 'score': -1.0093516111373901}
 10%|█████████████                                                                                                                      | 1/10 [00:18<02:49, 18.82s/it{'text1': 'Bandar bin Mohammed bin Abdulrahman Al-Saud (; 1924 – 21 January 2020), was a Saudi prince as member of House of Saud, son of Muhammad bin Abdul-Rahman, nephew of Ibn Saud and cousin of current monarch Salman of Saudi Arabia.', 'text2': 'He is the grandson of Ibn Saud.', 'label': 'not_entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'Furthermore,', 'score': -0.9371277093887329}
{'text1': 'Bandar bin Mohammed bin Abdulrahman Al-Saud (; 1924 – 21 January 2020), was a Saudi prince as member of House of Saud, son of Muhammad bin Abdul-Rahman, nephew of Ibn Saud and cousin of current monarch Salman of Saudi Arabia.', 'text2': 'He is the grandson of Ibn Saud.', 'label': 'not_entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'Furthermore,', 'score': -0.9371277093887329}
 20%|██████████████████████████▏                                                                                                        | 2/10 [00:29<01:50, 13.84s/it]{'text1': "On December 8, 2019 after Norvell's departure to Florida State, Silverfield served as the interim head coach before being promoted to head coach on December 13, 2019.", 'text2': 'He was named the interim athletic director at the University of Central Florida.', 'label': 'not_entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'Furthermore,', 'score': -1.5110338926315308}
{'text1': "On December 8, 2019 after Norvell's departure to Florida State, Silverfield served as the interim head coach before being promoted to head coach on December 13, 2019.", 'text2': 'He was named the interim athletic director at the University of Central Florida.', 'label': 'not_entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'Furthermore,', 'score': -1.5110338926315308}
 30%|███████████████████████████████████████▎                                                                                           | 3/10 [00:39<01:24, 12.03s/it]{'text1': 'On the conceptual side, securing organs at optimum times does not require us to constantly redefine death and when it occurs so that persons who are alive may have their organs taken.', 'text2': 'We can use this concept of optimal time to determine how long a person should live in order to be able to donate organs.', 'label': 'not_entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'Furthermore,', 'score': -1.6909466981887817}
{'text1': 'On the conceptual side, securing organs at optimum times does not require us to constantly redefine death and when it occurs so that persons who are alive may have their organs taken.', 'text2': 'We can use this concept of optimal time to determine how long a person should live in order to be able to donate organs.', 'label': 'not_entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'Furthermore,', 'score': -1.6909466981887817}
 40%|████████████████████████████████████████████████████▍                                                                              | 4/10 [00:49<01:08, 11.36s/it]{'text1': 'In the same year, Prince Gong displeased Empress Dowager Cixi when he strongly opposed her plan to rebuild the Old Summer Palace.', 'text2': 'He was not happy with the fact that his mother Consort Wang had been made a consort and given the title of empress, which angered him.', 'label': 'not_entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'Furthermore,', 'score': -1.6959114074707031}
{'text1': 'In the same year, Prince Gong displeased Empress Dowager Cixi when he strongly opposed her plan to rebuild the Old Summer Palace.', 'text2': 'He was not happy with the fact that his mother Consort Wang had been made a consort and given the title of empress, which angered him.', 'label': 'not_entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'Furthermore,', 'score': -1.6959114074707031}
 50%|█████████████████████████████████████████████████████████████████▌                                                                 | 5/10 [00:59<00:53, 10.77s/it]{'text1': 'Karstadt contributed a large sum of money towards the decoration of the station and was in return rewarded with direct access from the station to the store.', 'text2': 'He was given the right to sell goods at the store for a fixed price.', 'label': 'not_entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'Furthermore,', 'score': -1.7875747680664062}
{'text1': 'Karstadt contributed a large sum of money towards the decoration of the station and was in return rewarded with direct access from the station to the store.', 'text2': 'He was given the right to sell goods at the store for a fixed price.', 'label': 'not_entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'Furthermore,', 'score': -1.7875747680664062}
 60%|██████████████████████████████████████████████████████████████████████████████▌                                                    | 6/10 [01:08<00:41, 10.42s/it]{'text1': 'While in Drill Instructor status, both male and female DIs wear a World War I campaign hat with their service and utility uniforms.', 'text2': 'The uniform is worn by all members of the drill team as well as the drill instructor during the annual training exercises.', 'label': 'not_entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'Furthermore,', 'score': -1.9901036024093628}
{'text1': 'While in Drill Instructor status, both male and female DIs wear a World War I campaign hat with their service and utility uniforms.', 'text2': 'The uniform is worn by all members of the drill team as well as the drill instructor during the annual training exercises.', 'label': 'not_entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'Furthermore,', 'score': -1.9901036024093628}
 70%|███████████████████████████████████████████████████████████████████████████████████████████▋                                       | 7/10 [01:18<00:30, 10.23s/it]{'text1': 'The film won four Golden Eagle Awards (2020) for Best Motion Picture, Best Leading Actor (Alexander Petrov), Best Supporting Actor (Ivan Yankovsky), and Best Film Editing.', 'text2': 'The film was nominated for three awards at the European Film Awards.', 'label': 'not_entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'Furthermore,', 'score': -1.0726913213729858}
{'text1': 'The film won four Golden Eagle Awards (2020) for Best Motion Picture, Best Leading Actor (Alexander Petrov), Best Supporting Actor (Ivan Yankovsky), and Best Film Editing.', 'text2': 'The film was nominated for three awards at the European Film Awards.', 'label': 'not_entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'Furthermore,', 'score': -1.0726913213729858}
 80%|████████████████████████████████████████████████████████████████████████████████████████████████████████▊                          | 8/10 [01:28<00:20, 10.19s/it]{'text1': 'Its basin contains over 3000 rivers, of which 425 are longer than and 11 are longer than ; 1011 of those rivers directly flow into the Donets.', 'text2': 'There are more than 100 lakes in the basin.', 'label': 'not_entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'Furthermore,', 'score': -1.372369647026062}
{'text1': 'Its basin contains over 3000 rivers, of which 425 are longer than and 11 are longer than ; 1011 of those rivers directly flow into the Donets.', 'text2': 'There are more than 100 lakes in the basin.', 'label': 'not_entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'Furthermore,', 'score': -1.372369647026062}
 90%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉             | 9/10 [01:39<00:10, 10.27s/it]{'text1': 'He had also gone to seek medical attention for his failing health and settle a business deal with his Dutch associate.', 'text2': 'He was in the process of writing a book about his experiences during the war.', 'label': 'not_entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'Furthermore,', 'score': -1.4308563470840454}
{'text1': 'He had also gone to seek medical attention for his failing health and settle a business deal with his Dutch associate.', 'text2': 'He was in the process of writing a book about his experiences during the war.', 'label': 'not_entailment', 'start_prompt': 'Wikipedia ', 'conj_prompt': 'Furthermore,', 'score': -1.4308563470840454}
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [01:49<00:00, 10.94s/it]
saved to temp_gen/mrpc_not_entailment_10_sorted.json
temp_gen
['sst-2_1_10_sorted.json', 'mrpc_entailment_10.json', 'mnli_neutral_20.json', 'mnli_contradiction_20.json', 'mrpc_not_entailment_10.json', 'mrpc_not_entailment_10_sorted.json', 'sst-2_0_10_sorted.json', 'mnli_entailment_20.json', 'mrpc_entailment_10_sorted.json', 'mnli_entailment_20_sorted.json', 'sst-2_0_10.json', 'mnli_neutral_20_sorted.json', 'sst-2_0_50.json', 'sst-2_0_50_sorted.json', 'sst-2_1_10.json', 'mnli_contradiction_20_sorted.json']
Label entailment: 8 total samples
Label not_entailment: 10 total samples
Label entailment: 8 selected samples
Label not_entailment: 10 selected samples
Total 18 samples
saved to data/MRPC/train.json

```



# WHAT I CANNOT DO:



The bidirectional PLM code has lots of error🤷 and is extremely old-fashioned, so I haven't fixed those bugs.



