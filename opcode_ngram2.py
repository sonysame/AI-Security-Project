import os
from tqdm import tqdm
from collections import defaultdict
import csv
import pandas as pd
import numpy as np
import pickle

def main():
	with open("2gram.csv", 'r') as reader:
		data=reader.read()
		lines=data.split("\n")[1:-1]
		print(lines[0])
	total_opcode=[]
	mode=['antipatch', 'antidump', 'antivm', 'original']
	for i in lines:
		tmp=i.split(')')[0].replace('(','').replace('"','').replace(' ','')
		total_opcode.append(tmp)
	print(total_opcode[0])
	total_2gram={}
	for i in total_opcode:
		total_2gram[i]=[0,0,0,0,0,0,0,0]
	#print(total_2gram)
	print(len(total_2gram))
	total_file = []

	ori_path = os.getcwd()
	path_file = ori_path

	for mode_index in range(len(mode)):
		path_file = ori_path
		path_file+=("/"+mode[mode_index]+"_result")
		print(path_file)
		total_file.append(os.listdir(path_file))
		for i in tqdm(range(len(total_file[mode_index])), mininterval=1):
			target_file=total_file[mode_index][i]
			f=open(path_file+"/"+target_file)
			opcode=[]
			while True:
				line=f.readline()
				if not line: break
				if "Instruction" in line:
					opcode.append(line.strip().split("Instruction : ")[1])
			f.close()
			_2gram={}
			for i in total_opcode:
				_2gram[i] = False

			for j in range(len(opcode)-1):
				total_2gram["'"+opcode[j]+"','"+opcode[j+1]+"'"][2*mode_index]+=1
				if _2gram["'" + opcode[j] + "','" + opcode[j + 1] + "'"]==False:
					_2gram["'" + opcode[j] + "','" + opcode[j + 1] + "'"] = True

			for j in _2gram:
				if _2gram[j]==True:
					total_2gram[j][2*mode_index+1]+=1

	#print(total_2gram)
	key1List = total_2gram.keys()
	antipatch_tf = np.array(list(total_2gram.values()))[:, 0]
	antipatch_df = np.array(list(total_2gram.values()))[:, 1]
	antidump_tf= np.array(list(total_2gram.values()))[:, 2]
	antidump_df = np.array(list(total_2gram.values()))[:, 3]
	antivm_tf = np.array(list(total_2gram.values()))[:, 4]
	antivm_df = np.array(list(total_2gram.values()))[:, 5]
	original_tf = np.array(list(total_2gram.values()))[:, 6]
	original_df = np.array(list(total_2gram.values()))[:, 7]
	original_calc_tf=np.round(original_tf/np.sum(original_tf),4)
	antipatch_calc_tf=np.round(antipatch_tf/np.sum(antipatch_tf),4)
	antipatch_calc_idf = np.round(164 / np.log2(antipatch_df+original_df+0.1),4)
	antipatch_tf_idf=np.multiply(antipatch_calc_tf, antipatch_calc_idf)
	data=np.stack((antipatch_tf, antipatch_df, original_tf, original_df, original_calc_tf, antipatch_calc_tf,antipatch_tf_idf), axis=1)
	data_df = pd.DataFrame(data, columns=["antipatch_TF", "antipatch_DF", "original_TF", "original_DF", "original_calc_TF", "antipatch_calc_TF", "antipatch_calc_TF_IDF"],
						   index=total_opcode)
	data_df.to_pickle('2gram.pkl')
	rows1 = zip(key1List, antipatch_tf.tolist(), antipatch_df.tolist(), antidump_tf.tolist(), antidump_df.tolist(), antivm_tf.tolist(), antivm_df.tolist(), original_tf.tolist(), original_df.tolist(),
				original_calc_tf.tolist(),antipatch_calc_tf.tolist(), antipatch_calc_idf.tolist(), antipatch_tf_idf.tolist())
	with open('2gram_result.csv', 'w', encoding='utf-8') as f:
		w1 = csv.writer(f)
		w1.writerow(["2-gram", "antipatch_TF","antipatch_DF", "antidump_TF", "antidump_DF", "antivm_TF", "antivm_DF", "original_TF", "original_DF",
					 "original_calc_TF","antipatch_calc_TF","antipatch_calc_IDF","antipatch_calc_TF_IDF"])
		for row in rows1:
			w1.writerow(row)
def main2(mode):
	ori_path=os.getcwd()
	path_file=ori_path
	#mode=["original","antipatch", "antidump", "antivm"]
	total_file=[]

	total_2gram=defaultdict(lambda: 0)
	total_3gram=defaultdict(lambda: 0)
	total_4gram=defaultdict(lambda: 0)
	#data = pd.read_csv(r"2gram_result.csv")  # CSV 파일 불러오기
	#mode_2gram=pd.DataFrame(data, columns=['2gram', 'frequency'])  # 특정컬럼선택

	mode_2gram=[]
	mode_3gram=[]
	mode_4gram=[]

	for mode_index in range(len(mode)):
		path_file = ori_path
		path_file+=("/"+mode[mode_index]+"_result")
		print(path_file)
		total_file.append(os.listdir(path_file))
		mode_2gram.append(defaultdict(lambda: 0))
		mode_3gram.append(defaultdict(lambda: 0))
		mode_4gram.append(defaultdict(lambda: 0))
		index=0
		for i in tqdm(range(len(total_file[mode_index])), mininterval=1):
			target_file=total_file[mode_index][i]
			f=open(path_file+"/"+target_file)
			opcode=[]
			while True:
				line=f.readline()
				if not line: break
				if "Instruction" in line:
					opcode.append(line.strip().split("Instruction : ")[1])
			f.close()

			_2gram=defaultdict(lambda: 0)
			_3gram=defaultdict(lambda: 0)
			_4gram=defaultdict(lambda: 0)

			for i in range(len(opcode)-3):
				_2gram[(opcode[i], opcode[i+1])]+=1
				mode_2gram[mode_index][(opcode[i], opcode[i+1])]+=1
				total_2gram[(opcode[i], opcode[i + 1])] += 1
				_3gram[(opcode[i], opcode[i+1], opcode[i+2])]+=1
				mode_3gram[mode_index][(opcode[i], opcode[i+1], opcode[i+2])]+=1
				total_3gram[(opcode[i], opcode[i + 1], opcode[i + 2])] += 1
				_4gram[(opcode[i], opcode[i+1], opcode[i+2], opcode[i+3])]+=1
				mode_4gram[mode_index][(opcode[i], opcode[i+1], opcode[i+2], opcode[i+3])]+=1
				total_4gram[(opcode[i], opcode[i + 1], opcode[i + 2], opcode[i + 3])] += 1

			_2gram[(opcode[len(opcode)-3], opcode[len(opcode)-2])]+=1
			mode_2gram[mode_index][(opcode[len(opcode)-3], opcode[len(opcode)-2])]+=1
			total_2gram[(opcode[len(opcode) - 3], opcode[len(opcode) - 2])] += 1
			_2gram[(opcode[len(opcode)-2], opcode[len(opcode)-1])]+=1
			mode_2gram[mode_index][(opcode[len(opcode)-2], opcode[len(opcode)-1])]+=1
			total_2gram[(opcode[len(opcode) - 2], opcode[len(opcode) - 1])] += 1

			_3gram[(opcode[len(opcode)-3], opcode[len(opcode)-2], opcode[len(opcode)-1])]+=1
			mode_3gram[mode_index][(opcode[len(opcode)-3], opcode[len(opcode)-2], opcode[len(opcode)-1])]+=1
			total_3gram[(opcode[len(opcode) - 3], opcode[len(opcode) - 2], opcode[len(opcode) - 1])] += 1

			#print(index, len(opcode), len(_2gram), len(_3gram), len(_4gram))

	print(len(total_2gram))
	print(len(total_3gram))
	print(len(total_4gram))

	print(len(mode_2gram))
	print(len(mode_3gram))
	print(len(mode_4gram))

	key1List=total_2gram.keys()
	value1List=total_2gram.values()
	rows1=zip(key1List, value1List)

	key2List = total_3gram.keys()
	value2List = total_3gram.values()
	rows2 = zip(key2List, value2List)

	key3List=total_4gram.keys()
	value3List=total_4gram.values()
	rows3=zip(key3List, value3List)

	with open('new_2gram_result.csv', 'w', encoding='utf-8') as f:
		w1=csv.writer(f)
		w1.writerow(["2-gram","frequency"])
		for row in rows1:
			w1.writerow(row)

	with open('new_3gram_result.csv', 'w', encoding='utf-8') as f:
		w2=csv.writer(f)
		w2.writerow(["3-gram","frequency"])
		for row in rows2:
			w2.writerow(row)

	with open('new_4gram_result.csv', 'w', encoding='utf-8') as f:
		w3=csv.writer(f)
		w3.writerow(["4-gram","frequency"])
		for row in rows3:
			w3.writerow(row)


if __name__=="__main__":
	#main(mode=["antipatch", "antidump", "antivm", "original"])
	main()