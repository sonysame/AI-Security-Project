import os
from tqdm import tqdm
ori_path=os.getcwd()
pin_path=ori_path+("\\pin\\pin\\pin.exe")
mode="antivm"
target_path=ori_path+"\\"+mode
target_file=os.listdir(target_path)
for i in target_file:
    if not i.endswith(".exe"):
        target_file.remove(i)

for i in tqdm(range(len(target_file)), mininterval=1):
    command=pin_path+" -t opcode.dll -o "+mode+"_result\\result_"+target_file[i].split(".")[0]+".txt -- "+ target_path+"\\"+target_file[i]
    os.system(command)