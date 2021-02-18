# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import re
from nltk.stem import *
from nir.utils.args_utils import ArgsParser, load_config, merge_config

#
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

FLAGS = ArgsParser().parse_args()
config = load_config(FLAGS.config)
merge_config(FLAGS.opt)

# retrieve paras
doStem = config['Preprocess']['Topic']['is_stemming']
inputFile = topic_path = config['Preprocess']['Topic']['topic_name']
outFilepath = config['Preprocess']['Topic']['output_name']

cleanTextRegex = re.compile('[^a-zA-Z]')
count = 0
stemmer = PorterStemmer()


with open(outFilepath, 'w') as outputFile:
    with open(sys.argv[1], 'r') as inputFile:
        currentId = ''
        for inLine in inputFile.readlines():
            if inLine.startswith('<num> Number:'):
                currentId = inLine.replace('<num> Number:', '').strip()
            if inLine.startswith('<title>'):
                text = inLine.replace('<title>', '').strip()
                # clean text
                text = cleanTextRegex.sub(' ', text).lower()
                # remove multiple whitespaces
                text = text.replace('    ',' ').replace('   ',' ').replace('  ',' ')

                wordList = []
                for w in text.split(' '):
                    if w:
                        if doStem:
                            cleaned = stemmer.stem(w.strip())
                        else:
                            cleaned = w.strip()
                        wordList.append(cleaned)
                outputText = ' '.join(wordList)

                # write single line output
                outputFile.write(currentId)
                outputFile.write(' ')
                outputFile.write(outputText.strip())
                outputFile.write('\n')
                count = count + 1

print('Completed all ', count, ' topics')
print('Saved in: ', outFilepath)
