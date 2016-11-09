# _*_ coding:utf-8 _*_
import sys
from cStringIO  import StringIO
from xml.etree  import ElementTree as ET
from htmlentitydefs import name2codepoint
import os
import re
import itertools


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = str(string.encode("utf8"))
    # string = re.sub(r"\`\`"," ", string)
    # string = re.sub(r"\'\'"," ", string)
    string = re.sub(r"[^A-Za-z0-9,.!?]", " ", string)
    # string = re.sub(r"[^A-Za-z0-9,!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = " ".join(string.split())
    return string.strip().lower()


def parse(file1, file2):
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
	ind = 1
	entity_lists = []
	plus = []
	correct = []
	for mc in root:
		entity_lists.append({})
		plus.append({})
		correct.append({})
		memory = None
		for child in mc:
			if child.tag == 'question':
				for grandchildren in child:
					if grandchildren.tag=='blank':
						t = grandchildren.text
						t = clean_str(t)
						plus[-1][t] = ' @entity'+str(ind) + ' '	
						ind += 1
						memory = plus[-1][t]
						break
			elif child.tag == 'choice':
				t = clean_str(child.text)
				if child.attrib['correct'] =='true':
					correct[-1][t] = memory
				else:
					entity_lists[-1][t] = ' @entity'+str(ind) + ' '
					ind += 1


	i = -1
	f = open(path+file2,'w')
	for mc in root:
		i += 1
		dic = entity_lists[i]
		pl = plus[i]
		co = correct[i]
		for child in mc:
			if child.tag=='question':
				question = ''
				for grandchildren in child:
					# print grandchildren.text
					if grandchildren.tag == 'blank':
						t = '@placeholder '
					else:
						t = grandchildren.text
						if not t:
							continue
						t = clean_str(t)+ ' '
					question += t
			if child.tag == 'choice':
				if child.attrib['correct'] =='true':
					t = clean_str(child.text)
					answer = co[t]
			if child.tag == 'contextart':
				document = clean_str(child.text)
		flag = True
		for key, value in co.items():
			if key in document:
				flag = False
				document = document.replace(key, value)
		for key, value in pl.items():
			if key in document:
				flag = False
				document = document.replace(key, value)
		if flag:
			continue
		for key, value in dic.items():
			document = document.replace(key, value)
		f.write(question+'\n')
		f.write(answer+'\n')
		f.write(document+'\n\n')

reads = ['test.xml','train.xml','val.xml']
writes = ['test.txt','train.txt','val.txt']
for file1, file2 in zip(reads, writes):
	parse(file1, file2)
# print clean_str("James `` Duke '' Aiona")


