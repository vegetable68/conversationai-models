"""

Cleans and splits the toxicity data from Figshare: 

https://figshare.com/articles/Wikipedia_Talk_Labels_Toxicity/4563973

------------------------------------------------------------------------

Copyright 2018, Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import os

DEFAULT_DATA_DIR = 'data/'

def process_figshare(input_data_dir = DEFAULT_DATA_DIR, output_data_dir = DEFAULT_DATA_DIR):
    """
    Cleans and splits the toxicity data from Figshare.
    
    Args:
          * input_data_dir (string): if provided, the directory where the Figshare tsvs are stored
          * output_data_dir (string): if provided, the directory where the output splits should be written
    """
    already_exist = True
    for split in ['train', 'test', 'dev']:
        if not os.path.isfile(os.path.join(output_data_dir, 'wiki_%s.csv' % split)):
            already_exist = False
            
    if already_exist:
        print('Processed files already exist.')
        return

    print('Processing files...', end = '')
    toxicity_annotated_comments = pd.read_csv(os.path.join(input_data_dir, 'toxicity_annotated_comments.tsv'), sep = '\t')
    toxicity_annotations = pd.read_csv(os.path.join(input_data_dir, 'toxicity_annotations.tsv'), sep = '\t')

    annotations_gped = toxicity_annotations.groupby('rev_id', as_index=False).agg({'toxicity': 'mean'})
    all_data = pd.merge(annotations_gped, toxicity_annotated_comments, on = 'rev_id')

    all_data['comment'] = all_data['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
    all_data['comment'] = all_data['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

    all_data['is_toxic'] = all_data['toxicity'] > 0.5

    # split into train, valid, test
    wiki_splits = {}
    for split in ['train', 'test', 'dev']:
        wiki_splits[split] = all_data.query('split == @split')

    for split in wiki_splits:
        wiki_splits[split].to_csv(os.path.join(output_data_dir, 'wiki_%s.csv' % split), index=False)
    print('Done!')

# TODO(nthain): Add input and output dirs as flags.
if __name__ == "__main__":
    process_figshare()