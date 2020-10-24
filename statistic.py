import os, sys
import numpy as np
import torch
from sklearn.utils import shuffle
import json
import xml.etree.ElementTree as ET


def read_bing_reviews(location):
    num_sentence = 0
    sentences = []
    sentiments = []
    aspects  = []


    with open(location,'r') as review_file:
        reviews = review_file.readlines()
        for review in reviews:
            current_aspect = []
            current_sentiment = []
            if review[:2] != '##' and '##' in review and '[' in review and \
                    ('+' in review.split('##')[0] or '-' in review.split('##')[0]):
                # print(review.split('##')[0])
                sentences.append(review.split('##')[1][:-1].replace('\t',' '))

                #aspects: may be more than one
                aspect_str = review.split('##')[0]
                if ',' in aspect_str:
                    aspect_all = aspect_str.split(',')
                    for each_aspect in aspect_all:
                        # print('each_aspect.split([)[0]: ',each_aspect.split('[')[0])
                        current_aspect.append(each_aspect.split('[')[0])
                        current_sentiment.append(each_aspect.split('[')[1][0])

                elif ',' not in aspect_str:
                    # print('aspect_str.split([)[0]: ',aspect_str.split('[')[0])
                    current_aspect.append(aspect_str.split('[')[0])
                    current_sentiment.append(aspect_str.split('[')[1][0])
                num_sentence+=1
                aspects.append(current_aspect)
                sentiments.append(current_sentiment)


    return sentences,aspects,sentiments



def read_xu_semseval16(location):
    sentences = []
    sentiments = []
    aspects  = []

    with open(location,'r') as review_file:
        instance = json.load(review_file)
        for id, ins in instance['data'].items():
            if ins['term']=='NULL': continue
            aspects.append(ins['term'])
            sentiments.append(ins['polarity'])
            sentences.append(ins['sentence'])
    return sentences,aspects,sentiments



def read_xu_semseval14(location):
    sentences = []
    sentiments = []
    aspects  = []

    with open(location,'r') as review_file:
        instance = json.load(review_file)
        for id, ins in instance.items():
            if ins['term']=='NULL': continue
            aspects.append(ins['term'])
            sentiments.append(ins['polarity'])
            sentences.append(ins['sentence'])
    return sentences,aspects,sentiments




def statistic(file_name,sentences,aspects,sentiments):
    '''
    #sentences
    #postive
    #negative
    #neutral
    #aspects

    '''
    if 'XuSemEval' in file_name:
        print('#sentences: ',len(list(set(sentences))))
        print('#aspects: ',len(aspects))
        print('#postive: ',len([s for s in sentiments if s=='positive' or s == '+']))
        print('#negative: ',len([s for s in sentiments if s=='negative' or s == '-']))
        print('#neutral: ',len([s for s in sentiments if s=='neutral' or s == '=']))

        num_sentences=len(list(set(sentences)))
        num_aspects=len(aspects)
        num_positive=len([s for s in sentiments if s=='positive' or s == '+'])
        num_negative=len([s for s in sentiments if s=='negative' or s == '-'])
        num_neutral=len([s for s in sentiments if s=='neutral' or s == '='])


        return str(num_sentences)+' S./'+str(num_aspects)+' A./'+ \
               str(num_positive) + ' P./'+str(num_negative)+' N./'+str(num_neutral)+' Ne.'


    elif 'Bing' in file_name:
        sentence_nums = len(sentences)

        train_nums = int(sentence_nums *0.8)
        valid_nums = int(sentence_nums *0.1)
        test_nums = int(sentence_nums *0.1)

        sentences_=[]
        aspects_=[]
        sentiments_=[]
        print('Train')
        for i in range(len(sentences)):
            for i_i in range(len(aspects[i])):
                sentences_.append(str(sentences[i]))
                aspects_.append(str(aspects[i][i_i]))
                sentiments_.append(str(sentiments[i][i_i]))
            if i == train_nums:
                break

        print('#sentences: ',len(list(set(sentences_))))
        print('#aspects: ',len(aspects_))
        print('#postive: ',len([s for s in sentiments_ if s=='positive' or s == '+']))
        print('#negative: ',len([s for s in sentiments_ if s=='negative' or s == '-']))
        print('#neutral: ',len([s for s in sentiments_ if s=='neutral' or s == '=']))
        num_train_sentences=len(list(set(sentences_)))
        num_train_aspects=len(aspects_)
        num_train_positive=len([s for s in sentiments_ if s=='positive' or s == '+'])
        num_train_negative=len([s for s in sentiments_ if s=='negative' or s == '-'])
        num_train_neutral=len([s for s in sentiments_ if s=='neutral' or s == '='])

        sentences_=[]
        aspects_=[]
        sentiments_=[]
        print('Valid')
        for i in range(len(sentences)):
            x = i+train_nums
            for i_i in range(len(aspects[x])):
                sentences_.append(str(sentences[x]))
                aspects_.append(str(aspects[x][i_i]))
                sentiments_.append(str(sentiments[x][i_i]))
            if i == valid_nums:
                break

        print('#sentences: ',len(list(set(sentences_))))
        print('#aspects: ',len(aspects_))
        print('#postive: ',len([s for s in sentiments_ if s=='positive' or s == '+']))
        print('#negative: ',len([s for s in sentiments_ if s=='negative' or s == '-']))
        print('#neutral: ',len([s for s in sentiments_ if s=='neutral' or s == '=']))
        num_val_sentences=len(list(set(sentences_)))
        num_val_aspects=len(aspects_)
        num_val_positive=len([s for s in sentiments_ if s=='positive' or s == '+'])
        num_val_negative=len([s for s in sentiments_ if s=='negative' or s == '-'])
        num_val_neutral=len([s for s in sentiments_ if s=='neutral' or s == '='])


        sentences_=[]
        aspects_=[]
        sentiments_=[]
        print('Test')
        for i in range(len(sentences)):
            x = i+train_nums+valid_nums

            for i_i in range(len(aspects[x])):
                sentences_.append(str(sentences[x]))
                aspects_.append(str(aspects[x][i_i]))
                sentiments_.append(str(sentiments[x][i_i]))

            if x == len(sentences)-1:
                break

        print('#sentences: ',len(list(set(sentences_))))
        print('#aspects: ',len(aspects_))
        print('#postive: ',len([s for s in sentiments_ if s=='positive' or s == '+']))
        print('#negative: ',len([s for s in sentiments_ if s=='negative' or s == '-']))
        print('#neutral: ',len([s for s in sentiments_ if s=='neutral' or s == '=']))

        num_test_sentences=len(list(set(sentences_)))
        num_test_aspects=len(aspects_)
        num_test_positive=len([s for s in sentiments_ if s=='positive' or s == '+'])
        num_test_negative=len([s for s in sentiments_ if s=='negative' or s == '-'])
        num_test_neutral=len([s for s in sentiments_ if s=='neutral' or s == '='])

        return str(num_train_sentences)+' S./'+str(num_train_aspects)+' A./'+ \
               str(num_train_positive) + ' P./'+str(num_train_negative)+' N./'+str(num_train_neutral)+' Ne.'+'\t' + \
               str(num_val_sentences)+' S./'+str(num_val_aspects)+' A./'+ \
               str(num_val_positive) + ' P./'+str(num_val_negative)+' N./'+str(num_val_neutral)+' Ne.'+'\t' + \
               str(num_test_sentences)+' S./'+str(num_test_aspects)+' A./'+ \
               str(num_test_positive) + ' P./'+str(num_test_negative)+' N./'+str(num_test_neutral)+' Ne.'+'\t'


if __name__ == "__main__":

    with open('statistic','w') as f_sta:
        domains = ['Speaker','Router','Computer']
        for domain in domains:
            print('Read Bing3domains ' + domain)
            sentences,aspects,sentiments = read_bing_reviews('./data/Review3Domains/'+domain+'.txt')
            sta = statistic('Bing3domains',sentences,aspects,sentiments)
            f_sta.writelines(sta+'\n')

        domains = ['Nokia6610','NikonCoolpix4300','CreativeLabsNomadJukeboxZenXtra40GB','CanonG3','ApexAD2600Progressive']
        for domain in domains:
            print('Read Bing5domains ' + domain)
            sentences,aspects,sentiments = read_bing_reviews('./data/Review5Domains/'+domain+'.txt')
            sta = statistic('Bing5domains',sentences,aspects,sentiments)
            f_sta.writelines(sta+'\n')

        domains = ['CanonPowerShotSD500','CanonS100','DiaperChamp','HitachiRouter','ipod', \
                   'LinksysRouter','MicroMP3','Nokia6600','Norton']
        for domain in domains:
            print('Read Bing9domains ' + domain)
            sentences,aspects,sentiments = read_bing_reviews('./data/Review9Domains/'+domain+'.txt')
            sta = statistic('Bing9domains', sentences,aspects,sentiments)
            f_sta.writelines(sta+'\n')

        # read from Xu
        years = ['14','16']
        for year in years:
            if year == '16':
                domains = ['rest',]
            elif year == '14':
                domains = ['rest','laptop']

            for domain in domains:
                sta=''
                for type in ['train','valid','test']:
                    print('Read XuSemEval ' + year + ' ' + domain + ' ' + type)
                    if year == '16':
                        sentences,aspects,sentiments = read_xu_semseval16('./data/XuSemEval/'+year+'/'+domain+'/'+type+'.json')
                    elif year == '14':
                        sentences,aspects,sentiments = read_xu_semseval14('./data/XuSemEval/'+year+'/'+domain+'/'+type+'.json')
                    sta += statistic('XuSemEval',sentences,aspects,sentiments) + '\t'


                f_sta.writelines(sta+'\n')


