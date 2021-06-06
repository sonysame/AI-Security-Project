import os
from tqdm import tqdm
from collections import defaultdict
import csv
import pandas as pd
import numpy as np
import pickle

def main():
	mode="antipatch"
	with open("2gram_feature_selected.txt", 'r') as reader:
		data=reader.read()
		lines=data.split(",\n")[:-1]
		print(lines[0])
	total_opcode = []
	for i in lines:
		tmp = i.split(')')[0].replace('(', '').replace('"', '').replace(' ', '')
		total_opcode.append(tmp)
	print(total_opcode[0])
	ori_path=os.getcwd()
	#path_file=ori_path
	path_file= [ori_path+("/" +mode+ "_result"),ori_path+("/original_result")]
	total_file=[os.listdir(path_file[0]),os.listdir(path_file[1])]

	_2gram_dataset=np.zeros((len(total_file[0])+len(total_file[1]),len(total_opcode)+1))
	for category_idx in range(len(total_file)):
		for i in tqdm(range(len(total_file[category_idx])), mininterval=1):
			target_file = total_file[category_idx][i]
			f = open(path_file[category_idx] + "/" + target_file)
			opcode = []
			w=0
			while True:
				line = f.readline()
				if not line: break
				if "Instruction" in line:
					opcode.append(line.strip().split("Instruction : ")[1])
			f.close()
			_2gram = {}
			for j in total_opcode:
				_2gram[j] = 0

			for j in range(len(opcode) - 1):
				key="'" + opcode[j] + "','" + opcode[j + 1] + "'"
				if key in _2gram:
					_2gram[key]+=1
					w+=1
			#_2gram_dataset[0]=total_opcode.append("category")
			tmp=np.round(np.array(list(_2gram.values()))/w,4).tolist()
			tmp.append(category_idx)
			_2gram_dataset[82*category_idx+i]=np.array(tmp)
	column_data=total_opcode
	column_data.append("category")
	_2gram_selected_df=pd.DataFrame(_2gram_dataset, columns=column_data)
	_2gram_selected_df.to_pickle('2gram_selected.pkl')
	print(_2gram_dataset)
if __name__=="__main__":
	#main(mode=["antipatch", "antidump", "antivm", "original"])
	main()