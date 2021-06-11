import os
from tqdm import tqdm
from collections import defaultdict
import csv
import pandas as pd
import numpy as np
import pickle
import random
import time
def main():
	start_time=time.time()
	_2gram_feature_opcode = []
	with open("_2gram_antipatch_multiple_feature_selected", 'r') as reader:
		data=reader.read()
		lines1=data.split(",\n")[:-1]
	with open("_2gram_antidump_multiple_feature_selected", 'r') as reader:
		data=reader.read()
		lines2=data.split(",\n")[:-1]
	with open("_2gram_antivm_multiple_feature_selected", 'r') as reader:
		data=reader.read()
		lines3=data.split(",\n")[:-1]

	for i in range(9):
		opcode_tmp = []
		for j in lines1[i*(len(lines1)//10):(i+1)*(len(lines1)//10)]:
			tmp = j.replace('"', '').replace(' ', '').replace("]",'').replace("[",'')
			if tmp not in _2gram_feature_opcode:
				opcode_tmp.append(tmp)
		for j in lines2[i*(len(lines2)//10):(i+1)*(len(lines2)//10)]:
			tmp = j.replace('"', '').replace(' ', '').replace("]",'').replace("[",'')
			if tmp not in _2gram_feature_opcode:
				opcode_tmp.append(tmp)
		for j in lines3[i * (len(lines3) // 10):(i + 1) * (len(lines3) // 10)]:
			tmp = j.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
			if tmp not in _2gram_feature_opcode:
				opcode_tmp.append(tmp)
		opcode_tmp=list(set(opcode_tmp))
		random.shuffle(opcode_tmp)
		_2gram_feature_opcode+=opcode_tmp

	opcode_tmp = []
	for j in lines1[9 * (len(lines1) // 10):]:
		tmp = j.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
		if tmp not in _2gram_feature_opcode:
			opcode_tmp.append(tmp)
	for j in lines2[9 * (len(lines2) // 10):]:
		tmp = j.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
		if tmp not in _2gram_feature_opcode:
			opcode_tmp.append(tmp)
	for j in lines3[9 * (len(lines3) // 10):]:
		tmp = j.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
		if tmp not in _2gram_feature_opcode:
			opcode_tmp.append(tmp)
	opcode_tmp = list(set(opcode_tmp))
	random.shuffle(opcode_tmp)
	_2gram_feature_opcode += opcode_tmp

	print(time.time() - start_time)

	start_time = time.time()
	_3gram_feature_opcode = []
	with open("_3gram_antipatch_multiple_feature_selected", 'r') as reader:
		data = reader.read()
		lines1 = data.split(",\n")[:-1]
	with open("_3gram_antidump_multiple_feature_selected", 'r') as reader:
		data = reader.read()
		lines2 = data.split(",\n")[:-1]
	with open("_3gram_antivm_multiple_feature_selected", 'r') as reader:
		data = reader.read()
		lines3 = data.split(",\n")[:-1]

	for i in range(9):
		opcode_tmp = []
		for j in lines1[i * (len(lines1) // 10):(i + 1) * (len(lines1) // 10)]:
			tmp = j.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
			if tmp not in _3gram_feature_opcode:
				opcode_tmp.append(tmp)
		for j in lines2[i * (len(lines2) // 10):(i + 1) * (len(lines2) // 10)]:
			tmp = j.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
			if tmp not in _3gram_feature_opcode:
				opcode_tmp.append(tmp)
		for j in lines3[i * (len(lines3) // 10):(i + 1) * (len(lines3) // 10)]:
			tmp = j.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
			if tmp not in _3gram_feature_opcode:
				opcode_tmp.append(tmp)
		opcode_tmp = list(set(opcode_tmp))
		random.shuffle(opcode_tmp)
		_3gram_feature_opcode += opcode_tmp

	opcode_tmp = []
	for j in lines1[9 * (len(lines1) // 10):]:
		tmp = j.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
		if tmp not in _3gram_feature_opcode:
			opcode_tmp.append(tmp)
	for j in lines2[9 * (len(lines2) // 10):]:
		tmp = j.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
		if tmp not in _3gram_feature_opcode:
			opcode_tmp.append(tmp)
	for j in lines3[9 * (len(lines3) // 10):]:
		tmp = j.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
		if tmp not in _3gram_feature_opcode:
			opcode_tmp.append(tmp)
	opcode_tmp = list(set(opcode_tmp))
	random.shuffle(opcode_tmp)
	_3gram_feature_opcode += opcode_tmp

	print(time.time() - start_time)

	start_time = time.time()
	_4gram_feature_opcode = []
	with open("_4gram_antipatch_multiple_feature_selected", 'r') as reader:
		data = reader.read()
		lines1 = data.split(",\n")[:-1]
	with open("_4gram_antidump_multiple_feature_selected", 'r') as reader:
		data = reader.read()
		lines2 = data.split(",\n")[:-1]
	with open("_4gram_antivm_multiple_feature_selected", 'r') as reader:
		data = reader.read()
		lines3 = data.split(",\n")[:-1]

	for i in range(9):
		opcode_tmp = []
		for j in lines1[i * (len(lines1) // 10):(i + 1) * (len(lines1) // 10)]:
			tmp = j.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
			if tmp not in _4gram_feature_opcode:
				opcode_tmp.append(tmp)
		for j in lines2[i * (len(lines2) // 10):(i + 1) * (len(lines2) // 10)]:
			tmp = j.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
			if tmp not in _4gram_feature_opcode:
				opcode_tmp.append(tmp)
		for j in lines3[i * (len(lines3) // 10):(i + 1) * (len(lines3) // 10)]:
			tmp = j.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
			if tmp not in _4gram_feature_opcode:
				opcode_tmp.append(tmp)
		opcode_tmp = list(set(opcode_tmp))
		random.shuffle(opcode_tmp)
		_4gram_feature_opcode += opcode_tmp

	opcode_tmp = []
	for j in lines1[9 * (len(lines1) // 10):]:
		tmp = j.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
		if tmp not in _4gram_feature_opcode:
			opcode_tmp.append(tmp)
	for j in lines2[9 * (len(lines2) // 10):]:
		tmp = j.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
		if tmp not in _4gram_feature_opcode:
			opcode_tmp.append(tmp)
	for j in lines3[9 * (len(lines3) // 10):]:
		tmp = j.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
		if tmp not in _4gram_feature_opcode:
			opcode_tmp.append(tmp)
	opcode_tmp = list(set(opcode_tmp))
	random.shuffle(opcode_tmp)
	_4gram_feature_opcode += opcode_tmp
	print(time.time() - start_time)


	print(len(_2gram_feature_opcode))
	print(len(_3gram_feature_opcode))
	print(len(_4gram_feature_opcode))

	ori_path=os.getcwd()
	#path_file=ori_path

	path_file= [ori_path+("/antipatch_result"),ori_path+("/antidump_result"),ori_path+("/antivm_result"),ori_path+("/ori_original_result")]
	total_file=[os.listdir(path_file[0])[:62],os.listdir(path_file[1])[:62],os.listdir(path_file[2])[:62],os.listdir(path_file[3])[:62]]

	start_time = time.time()
	_2gram_dataset=np.zeros((len(total_file[0])+len(total_file[1])+len(total_file[2])+len(total_file[3]),len(_2gram_feature_opcode)+1))
	for category_idx in range(len(total_file)):
		for i in tqdm(range(len(total_file[category_idx])), mininterval=1):
			target_file = total_file[category_idx][i]
			f = open(path_file[category_idx] + "/" + target_file)
			opcode = []
			w=0.
			while True:
				line = f.readline()
				if not line: break
				if "Instruction" in line:
					opcode.append(line.strip().split("Instruction : ")[1])
			f.close()
			_2gram = {}

			for j in _2gram_feature_opcode:
				_2gram[j] = 0

			for j in range(len(opcode) - 1):
				key="'" + opcode[j] + "','" + opcode[j + 1] + "'"
				if key in _2gram:
					_2gram[key]+=1
					w+=1
			tmp=np.round(np.array(list(_2gram.values()))/w,4).tolist()
			tmp.append(category_idx)
			_2gram_dataset[62*category_idx+i]=np.array(tmp)
	column_data=_2gram_feature_opcode+["category"]
	_2gram_selected_df=pd.DataFrame(_2gram_dataset, columns=column_data)
	_2gram_selected_df.to_pickle('2gram_selected.pkl')

	print(time.time() - start_time)

	start_time = time.time()
	_3gram_dataset = np.zeros((len(total_file[0]) + len(total_file[1]) + len(total_file[2]) + len(total_file[3]),
							   len(_3gram_feature_opcode) + 1))
	for category_idx in range(len(total_file)):
		for i in tqdm(range(len(total_file[category_idx])), mininterval=1):
			target_file = total_file[category_idx][i]
			f = open(path_file[category_idx] + "/" + target_file)
			opcode = []
			w = 0.
			while True:
				line = f.readline()
				if not line: break
				if "Instruction" in line:
					opcode.append(line.strip().split("Instruction : ")[1])
			f.close()
			_3gram = {}
			for j in _3gram_feature_opcode:
				_3gram[j] = 0

			for j in range(len(opcode) - 2):
				key = "'" + opcode[j] + "','" + opcode[j + 1] + "','" + opcode[j + 2] + "'"
				if key in _3gram:
					_3gram[key] += 1
					w += 1
			tmp = np.round(np.array(list(_3gram.values())) / w, 4).tolist()
			tmp.append(category_idx)
			_3gram_dataset[62 * category_idx + i] = np.array(tmp)
	column_data = _3gram_feature_opcode + ["category"]
	_3gram_selected_df = pd.DataFrame(_3gram_dataset, columns=column_data)
	_3gram_selected_df.to_pickle('3gram_selected.pkl')
	print(time.time() - start_time)

	start_time = time.time()
	_4gram_dataset = np.zeros((len(total_file[0]) + len(total_file[1]) + len(total_file[2]) + len(total_file[3]),
							   len(_4gram_feature_opcode) + 1))
	for category_idx in range(len(total_file)):
		for i in tqdm(range(len(total_file[category_idx])), mininterval=1):
			target_file = total_file[category_idx][i]
			f = open(path_file[category_idx] + "/" + target_file)
			opcode = []
			w = 0.
			while True:
				line = f.readline()
				if not line: break
				if "Instruction" in line:
					opcode.append(line.strip().split("Instruction : ")[1])
			f.close()
			_4gram = {}
			for j in _4gram_feature_opcode:
				_4gram[j] = 0

			for j in range(len(opcode) - 3):
				key = "'" + opcode[j] + "','" + opcode[j + 1]+"','" + opcode[j + 2]+"','" + opcode[j + 3] + "'"
				if key in _4gram:
					_4gram[key] += 1
					w += 1
			tmp = np.round(np.array(list(_4gram.values())) / w, 4).tolist()
			tmp.append(category_idx)
			_4gram_dataset[62 * category_idx + i] = np.array(tmp)
	column_data = _4gram_feature_opcode+["category"]
	_4gram_selected_df = pd.DataFrame(_4gram_dataset, columns=column_data)
	_4gram_selected_df.to_pickle('4gram_selected.pkl')
	print(time.time()-start_time)

	total_file_test = [os.listdir(path_file[0])[62:], os.listdir(path_file[1])[62:], os.listdir(path_file[2])[62:],
					   os.listdir(path_file[3])[62:]]

	start_time = time.time()
	_2gram_dataset_test = np.zeros((len(total_file_test[0]) + len(total_file_test[1]) + len(total_file_test[2]) + len(
		total_file_test[3]), len(_2gram_feature_opcode) + 1))
	for category_idx in range(len(total_file_test)):
		for i in tqdm(range(len(total_file_test[category_idx])), mininterval=1):
			target_file = total_file_test[category_idx][i]
			f = open(path_file[category_idx] + "/" + target_file)
			opcode = []
			w = 0.
			while True:
				line = f.readline()
				if not line: break
				if "Instruction" in line:
					opcode.append(line.strip().split("Instruction : ")[1])
			f.close()
			_2gram = {}
			for j in _2gram_feature_opcode:
				_2gram[j] = 0

			for j in range(len(opcode) - 1):
				key = "'" + opcode[j] + "','" + opcode[j + 1] + "'"
				if key in _2gram:
					_2gram[key] += 1
					w += 1
			tmp = np.round(np.array(list(_2gram.values())) / w, 4).tolist()
			tmp.append(category_idx)
			_2gram_dataset_test[20 * category_idx + i] = np.array(tmp)
	column_data = _2gram_feature_opcode+["category"]
	_2gram_selected_test_df = pd.DataFrame(_2gram_dataset_test, columns=column_data)
	_2gram_selected_test_df.to_pickle('2gram_selected_test.pkl')

	print(time.time() - start_time)

	start_time = time.time()

	_3gram_dataset_test = np.zeros(
		(len(total_file_test[0]) + len(total_file_test[1]) + len(total_file_test[2]) + len(total_file_test[3]),
		 len(_3gram_feature_opcode) + 1))
	for category_idx in range(len(total_file_test)):
		for i in tqdm(range(len(total_file_test[category_idx])), mininterval=1):
			target_file = total_file_test[category_idx][i]
			f = open(path_file[category_idx] + "/" + target_file)
			opcode = []
			w = 0.
			while True:
				line = f.readline()
				if not line: break
				if "Instruction" in line:
					opcode.append(line.strip().split("Instruction : ")[1])
			f.close()
			_3gram = {}
			for j in _3gram_feature_opcode:
				_3gram[j] = 0

			for j in range(len(opcode) - 2):
				key = "'" + opcode[j] + "','" + opcode[j + 1] + "','" + opcode[j + 2]+ "'"
				if key in _3gram:
					_3gram[key] += 1
					w += 1
			tmp = np.round(np.array(list(_3gram.values())) / w, 4).tolist()
			tmp.append(category_idx)
			_3gram_dataset_test[20 * category_idx + i] = np.array(tmp)
	column_data = _3gram_feature_opcode+["category"]
	_3gram_selected_test_df = pd.DataFrame(_3gram_dataset_test, columns=column_data)
	_3gram_selected_test_df.to_pickle('3gram_selected_test.pkl')
	print(time.time() - start_time)

	start_time = time.time()
	_4gram_dataset_test = np.zeros(
		(len(total_file_test[0]) + len(total_file_test[1]) + len(total_file_test[2]) + len(total_file_test[3]),
		 len(_4gram_feature_opcode) + 1))
	for category_idx in range(len(total_file_test)):
		for i in tqdm(range(len(total_file_test[category_idx])), mininterval=1):
			target_file = total_file_test[category_idx][i]
			f = open(path_file[category_idx] + "/" + target_file)
			opcode = []
			w = 0.
			while True:
				line = f.readline()
				if not line: break
				if "Instruction" in line:
					opcode.append(line.strip().split("Instruction : ")[1])
			f.close()
			_4gram = {}
			for j in _4gram_feature_opcode:
				_4gram[j] = 0

			for j in range(len(opcode) - 3):
				key = "'" + opcode[j] + "','" + opcode[j + 1] + "','" + opcode[j + 2]+"','" + opcode[j + 3]+"'"
				if key in _4gram:
					_4gram[key] += 1
					w += 1
			tmp = np.round(np.array(list(_4gram.values())) / w, 4).tolist()
			tmp.append(category_idx)
			_4gram_dataset_test[20 * category_idx + i] = np.array(tmp)
	column_data = _4gram_feature_opcode+["category"]
	_4gram_selected_test_df = pd.DataFrame(_4gram_dataset_test, columns=column_data)
	_4gram_selected_test_df.to_pickle('4gram_selected_test.pkl')

	print(time.time() - start_time)

if __name__=="__main__":
	main()