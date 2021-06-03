import os
from tqdm import tqdm
ori_path=os.getcwd()
path_file=ori_path
path_file+=("\\Original Program")
total_file=os.listdir(path_file)
for i in total_file:
    if not i.endswith(".exe"):
        total_file.remove(i)
print(len(total_file))
path_themida=ori_path+"\\Themida\\Themida.exe"
print(path_themida)
#print(os.path.basename(path))
mode="antidump"
for i in tqdm(range(len(total_file)), mininterval=1):
    command=path_themida+" /protect "+path_themida.split("Themida.exe")[0]+mode+".tmd"+" /inputfile "+"\""+path_file+"\\"+total_file[i]+"\""+" /outputfile "+"\""+path_file+"\\"+mode+"\\"+total_file[i].split(".exe")[0]+"_"+mode+".exe"+"\""
    os.system(command)