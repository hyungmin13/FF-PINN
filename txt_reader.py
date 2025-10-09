#%%
import os
import pickle
import ast
import pickle
import re
from constants import *
def parse_tree_structured_txt(txt_path):
    tree_dict = {}

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(r'^(\w+):\s*(\{.*\})$', line)
        if match:
            key, value_str = match.groups()
            try:
                value = ast.literal_eval(value_str)
            except Exception as e:
                value = value_str  
            tree_dict[key] = value
        else:
            b = line.split(': ')
            key = b[0]
            value = b[1]
            tree_dict[key] = value
    return tree_dict


if __name__ == "__main__":
    cur_dir = os.getcwd()
    input_txt = os.path.dirname(cur_dir) + '/results/summaries/test_txt/test.txt'
    data = parse_tree_structured_txt(input_txt)

    c = Constants(**data)

    with open(os.path.dirname(cur_dir) + '/results/summaries/test_txt/constants_test_txt.pickle', 'rb') as f:
        pkl = pickle.load(f)
    print(pkl)

