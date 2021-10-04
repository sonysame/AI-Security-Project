import os
from tqdm import tqdm
from collections import defaultdict
import csv
import time
def main():
	ori_path=os.getcwd()
	total_2gram=defaultdict(lambda: 0)
	total_3gram=defaultdict(lambda: 0)
	total_4gram=defaultdict(lambda: 0)
	mode = ["antipatch", "antidump", "antivm", "ori_original"]

	for mode_index in range(len(mode)):
		path_file = ori_path
		path_file+=("/"+mode[mode_index]+"_result")
		print(path_file)
		total_file=os.listdir(path_file)
		start_time=time.time()
		for i in tqdm(range(62), mininterval=1):
			target_file=total_file[i]
			f=open(path_file+"/"+target_file)
			opcode=[]
			while True:
				line=f.readline()
				if not line: break
				if "Instruction" in line:
					opcode.append(line.strip().split("Instruction : ")[1])
			f.close()

			for j in range(len(opcode)-3):
				total_2gram[(opcode[j], opcode[j + 1])] += 1
				total_3gram[(opcode[j], opcode[j + 1], opcode[j + 2])] += 1
				total_4gram[(opcode[j], opcode[j + 1], opcode[j + 2], opcode[j + 3])] += 1

			total_2gram[(opcode[len(opcode) - 3], opcode[len(opcode) - 2])] += 1
			total_2gram[(opcode[len(opcode) - 2], opcode[len(opcode) - 1])] += 1
			total_3gram[(opcode[len(opcode) - 3], opcode[len(opcode) - 2], opcode[len(opcode) - 1])] += 1
		print(time.time()-start_time)

	key2List=total_2gram.keys()
	value2List=total_2gram.values()
	rows2=zip(key2List, value2List)

	key3List = total_3gram.keys()
	value3List = total_3gram.values()
	rows3 = zip(key3List, value3List)

	key4List=total_4gram.keys()
	value4List=total_4gram.values()
	rows4=zip(key4List, value4List)

	with open('2gram_opcode.csv', 'w', encoding='utf-8') as f:
		w2=csv.writer(f)
		w2.writerow(["2-gram","frequency"])
		for row in rows2:
			w2.writerow(row)

	with open('3gram_opcode.csv', 'w', encoding='utf-8') as f:
		w3=csv.writer(f)
		w3.writerow(["3-gram","frequency"])
		for row in rows3:
			w3.writerow(row)

	with open('4gram_opcode.csv', 'w', encoding='utf-8') as f:
		w4=csv.writer(f)
		w4.writerow(["4-gram","frequency"])
		for row in rows4:
			w4.writerow(row)


if __name__=="__main__":
	main()
	
