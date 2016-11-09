# _*_ coding:utf-8 _*_
import sys
from cStringIO  import StringIO
from xml.etree  import ElementTree as ET
from htmlentitydefs import name2codepoint
import os
import re
import itertools
import numpy as np
import random


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = str(string.encode("utf8"))
    string = re.sub(r"[^A-Za-z0-9,.!?]", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = " ".join(string.split())
    string = string.strip().lower()
    return string

def parse(file1):
    path = '../wdw_script/who_did_what/Strict/'

    parser = ET.XMLParser()
    parser.parser.UseForeignDTD(True)
    parser.entity.update((x, unichr(i)) for x, i in name2codepoint.iteritems())
    etree = ET.ElementTree()
    for filename in os.listdir(path):
        if filename == file1:
            data = open(path+filename, 'r')
            root = etree.parse(data, parser=parser)
            break
    #question:question, answer:choice=True, document
    entity_lists = []
    choices = [' a> ',' b> ',' c> ',' d> ',' e> ']
    
    for mc in root:
        entity_lists.append({"C":"", "Q":"","A":""})
        Choices = []
        for child in mc:
            if child.tag == 'question':
                ind = 0
                for grandchildren in child:
                    t = grandchildren.text
                    if grandchildren.tag=='blank':
                        entity_lists[-1]["Q"] += ' who '
                        continue
                    elif not t:
                        continue
                    t = clean_str(t)+ ' '
                    entity_lists[-1]["Q"] += t
            if child.tag == 'contextart':
                t = clean_str(child.text)#document
                if t[-1]!='.':
                    entity_lists[-1]["C"] += t+' . '
                else:
                    entity_lists[-1]["C"] += t+' '
            elif child.tag == 'choice':
                t = clean_str(child.text)
                # try:
                choice = choices[ind]
                if child.attrib['correct'] =='true':
                    entity_lists[-1]["A"] = ind
                # entity_lists[-1]["C"] += choice+t+' '
                Choices.append(choice+t+' ')
                # except:
                #     print ind
                #     if child.attrib['correct'] =='true':
                #         index = random.randint(0,4)
                #         entity_lists[-1]["A"] = index
                #         Choices[index] = choices[index] + t + ' '
                ind += 1

        entity_lists[-1]["C"] += "".join(Choices)
    # print entity_lists[0]                
    return entity_lists


def get_babi_raw(file1, file2):
    return parse(file1), parse(file2)           
def load_glove(dim):
    word2vec = {}
    
    print "==> loading glove"
    with open("../wdw_script/glove.6B/glove.6B." + str(dim) + "d.txt") as f:
        for line in f:    
            l = line.split()
            word2vec[l[0]] = map(float, l[1:])
            
    print "==> glove is loaded"
    
    return word2vec


def create_vector(word, word2vec, word_vector_size, silent=False):
    # if the word is missing from Glove, create some fake vector and store in glove!
    vector = np.random.uniform(0.0, 1.0, (word_vector_size,))
    word2vec[word] = vector
    if (not silent):
        print "utils.py::create_vector => %s is missing" % word
    return vector


def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=False):
    if not word in word2vec:
        create_vector(word, word2vec, word_vector_size, silent)
    if not word in vocab: 
        next_index = len(vocab)
        vocab[word] = next_index
        ivocab[next_index] = word
    
    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet")


def get_norm(x):
    x = np.array(x)
    return np.sum(x * x)