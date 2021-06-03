import os
from tqdm import tqdm
from collections import defaultdict

ori_path=os.getcwd()
path_file=ori_path
mode="antipatch"
path_file+=("\\"+mode+"_result")
print(path_file)
total_file=os.listdir(path_file)
total_2gram=defaultdict(lambda: 0)
total_3gram=defaultdict(lambda: 0)
total_4gram=defaultdict(lambda: 0)
index=0
for i in tqdm(range(len(total_file)), mininterval=1): 
	target_file=total_file[i]
	f=open(path_file+"\\"+target_file)
	opcode=[]
	while True:
		line=f.readline()
		if not line: break
		opcode.append(line.strip().split("Instruction : ")[1])
	f.close()

	_2gram=defaultdict(lambda: 0)
	_3gram=defaultdict(lambda: 0) 
	_4gram=defaultdict(lambda: 0)  
	
	for i in range(len(opcode)-3):
		_2gram[(opcode[i], opcode[i+1])]+=1
		total_2gram[(opcode[i], opcode[i+1])]+=1
		_3gram[(opcode[i], opcode[i+1], opcode[i+2])]+=1
		total_3gram[(opcode[i], opcode[i+1], opcode[i+2])]+=1
		_4gram[(opcode[i], opcode[i+1], opcode[i+2], opcode[i+3])]+=1
		total_4gram[(opcode[i], opcode[i+1], opcode[i+2], opcode[i+3])]+=1
	
	_2gram[(opcode[len(opcode)-3], opcode[len(opcode)-2])]+=1
	total_2gram[(opcode[len(opcode)-3], opcode[len(opcode)-2])]+=1
	_2gram[(opcode[len(opcode)-2], opcode[len(opcode)-1])]+=1
	total_2gram[(opcode[len(opcode)-2], opcode[len(opcode)-1])]+=1
	
	_3gram[(opcode[len(opcode)-3], opcode[len(opcode)-2], opcode[len(opcode)-1])]+=1
	total_3gram[(opcode[len(opcode)-3], opcode[len(opcode)-2], opcode[len(opcode)-1])]+=1

	index+=1
	#print(index, len(opcode), len(_2gram), len(_3gram), len(_4gram))

print(len(total_2gram))
print(len(total_3gram))
print(len(total_4gram))