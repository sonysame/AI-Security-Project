import os
from tqdm import tqdm
import csv
import pandas as pd
import numpy as np
import time
def main():

    total_2gram_opcode=[]
    with open("2gram_opcode.csv", 'r') as reader:
        data=reader.read()
        lines=data.split("\n")[1:-1]
    for i in lines:
        tmp=i.split(')')[0].replace('(','').replace('"','').replace(' ','')
        total_2gram_opcode.append(tmp)
    total_2gram={}
    for i in total_2gram_opcode:
        total_2gram[i]=[0,0,0,0,0,0,0,0]

    total_3gram_opcode = []
    with open("3gram_opcode.csv", 'r') as reader:
        data = reader.read()
        lines = data.split("\n")[1:-1]
    for i in lines:
        tmp = i.split(')')[0].replace('(', '').replace('"', '').replace(' ', '')
        total_3gram_opcode.append(tmp)
    total_3gram = {}
    for i in total_3gram_opcode:
        total_3gram[i] = [0, 0, 0, 0, 0, 0, 0, 0]

    total_4gram_opcode = []
    with open("4gram_opcode.csv", 'r') as reader:
        data = reader.read()
        lines = data.split("\n")[1:-1]
    for i in lines:
        tmp = i.split(')')[0].replace('(', '').replace('"', '').replace(' ', '')
        total_4gram_opcode.append(tmp)
    total_4gram = {}
    for i in total_4gram_opcode:
        total_4gram[i] = [0, 0, 0, 0, 0, 0, 0, 0]

    ori_path = os.getcwd()

    mode = ['antipatch', 'antidump', 'antivm', 'ori_original']

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
            _2gram={}
            _3gram={}
            _4gram={}
            for j in total_2gram_opcode:
                _2gram[j] = False
            for j in total_3gram_opcode:
                _3gram[j] = False
            for j in total_4gram_opcode:
                _4gram[j] = False

            for j in range(len(opcode)-3):
                total_2gram["'"+opcode[j]+"','"+opcode[j+1]+"'"][2*mode_index]+=1
                if _2gram["'" + opcode[j] + "','" + opcode[j + 1] + "'"]==False:
                    _2gram["'" + opcode[j] + "','" + opcode[j + 1] + "'"] = True

                total_3gram["'" + opcode[j] + "','" + opcode[j + 1] + "','" + opcode[j + 2] + "'"][2 * mode_index] += 1
                if _3gram["'" + opcode[j] + "','" + opcode[j + 1] + "','" + opcode[j + 2] + "'"] == False:
                    _3gram["'" + opcode[j] + "','" + opcode[j + 1] + "','" + opcode[j + 2] + "'"] = True

                total_4gram["'" + opcode[j] + "','" + opcode[j + 1] + "','" + opcode[j + 2] + "','" + opcode[j + 3] + "'"][2 * mode_index] += 1
                if _4gram["'" + opcode[j] + "','" + opcode[j + 1] + "','" + opcode[j + 2] + "','" + opcode[j + 3] + "'"] == False:
                    _4gram["'" + opcode[j] + "','" + opcode[j + 1] + "','" + opcode[j + 2] + "','" + opcode[j + 3] + "'"] = True

            total_2gram["'" + opcode[len(opcode)-3] + "','" + opcode[len(opcode)-2] + "'"][2 * mode_index] += 1
            if _2gram["'" + opcode[len(opcode)-3] + "','" + opcode[len(opcode)-2] + "'"] == False:
                _2gram["'" + opcode[len(opcode)-3] + "','" + opcode[len(opcode)-2] + "'"] = True

            total_2gram["'" + opcode[len(opcode) - 2] + "','" + opcode[len(opcode) - 1] + "'"][2 * mode_index] += 1
            if _2gram["'" + opcode[len(opcode) - 2] + "','" + opcode[len(opcode) - 1] + "'"] == False:
                _2gram["'" + opcode[len(opcode) - 2] + "','" + opcode[len(opcode) - 1] + "'"] = True

            total_3gram["'" + opcode[len(opcode)-3] + "','" + opcode[len(opcode)-2] + "','" + opcode[len(opcode)-1] + "'"][2 * mode_index] += 1
            if _3gram["'" + opcode[len(opcode)-3] + "','" + opcode[len(opcode)-2] + "','" + opcode[len(opcode)-1] + "'"] == False:
                _3gram["'" + opcode[len(opcode)-3] + "','" + opcode[len(opcode)-2] + "','" + opcode[len(opcode)-1] + "'"] = True

            for j in _2gram:
                if _2gram[j]==True:
                    total_2gram[j][2*mode_index+1]+=1

            for j in _3gram:
                if _3gram[j]==True:
                    total_3gram[j][2*mode_index+1]+=1

            for j in _4gram:
                if _4gram[j]==True:
                    total_4gram[j][2*mode_index+1]+=1
        print(time.time()-start_time)
    key2List = total_2gram.keys()
    _2antipatch_tf = np.array(list(total_2gram.values()))[:, 0]
    _2antipatch_df = np.array(list(total_2gram.values()))[:, 1]
    _2antidump_tf= np.array(list(total_2gram.values()))[:, 2]
    _2antidump_df = np.array(list(total_2gram.values()))[:, 3]
    _2antivm_tf = np.array(list(total_2gram.values()))[:, 4]
    _2antivm_df = np.array(list(total_2gram.values()))[:, 5]
    _2original_tf = np.array(list(total_2gram.values()))[:, 6]
    _2original_df = np.array(list(total_2gram.values()))[:, 7]

    key3List = total_3gram.keys()
    _3antipatch_tf = np.array(list(total_3gram.values()))[:, 0]
    _3antipatch_df = np.array(list(total_3gram.values()))[:, 1]
    _3antidump_tf = np.array(list(total_3gram.values()))[:, 2]
    _3antidump_df = np.array(list(total_3gram.values()))[:, 3]
    _3antivm_tf = np.array(list(total_3gram.values()))[:, 4]
    _3antivm_df = np.array(list(total_3gram.values()))[:, 5]
    _3original_tf = np.array(list(total_3gram.values()))[:, 6]
    _3original_df = np.array(list(total_3gram.values()))[:, 7]

    key4List = total_4gram.keys()
    _4antipatch_tf = np.array(list(total_4gram.values()))[:, 0]
    _4antipatch_df = np.array(list(total_4gram.values()))[:, 1]
    _4antidump_tf = np.array(list(total_4gram.values()))[:, 2]
    _4antidump_df = np.array(list(total_4gram.values()))[:, 3]
    _4antivm_tf = np.array(list(total_4gram.values()))[:, 4]
    _4antivm_df = np.array(list(total_4gram.values()))[:, 5]
    _4original_tf = np.array(list(total_4gram.values()))[:, 6]
    _4original_df = np.array(list(total_4gram.values()))[:, 7]

    _2antipatch_calc_tf = np.round(_2antipatch_tf / np.sum(_2antipatch_tf), 4)
    _2_binary_antipatch_calc_idf = np.round(124 / (_2antipatch_df + _2original_df + 0.1), 4)
    _2_binary_antipatch_tf_idf=np.round(np.multiply(_2antipatch_tf, _2_binary_antipatch_calc_idf),4)
    _2_multiple_antipatch_calc_idf = np.round(
        248 / (_2antipatch_df + _2antidump_df+_2antivm_df+_2original_df + 0.1), 4)
    _2_multiple_antipatch_tf_idf = np.round(np.multiply(_2antipatch_tf, _2_multiple_antipatch_calc_idf),4)

    _2antidump_calc_tf = np.round(_2antidump_tf / np.sum(_2antidump_tf), 4)
    _2_binary_antidump_calc_idf = np.round(124 / (_2antidump_df + _2original_df + 0.1), 4)
    _2_binary_antidump_tf_idf = np.round(np.multiply(_2antidump_tf, _2_binary_antidump_calc_idf),4)
    _2_multiple_antidump_calc_idf = np.round(
        248 / (_2antipatch_df + _2antidump_df + _2antivm_df + _2original_df + 0.1), 4)
    _2_multiple_antidump_tf_idf = np.round(np.multiply(_2antidump_tf, _2_multiple_antidump_calc_idf),4)

    _2antivm_calc_tf = np.round(_2antivm_tf / np.sum(_2antivm_tf), 4)
    _2_binary_antivm_calc_idf = np.round(124 / (_2antivm_df + _2original_df + 0.1), 4)
    _2_binary_antivm_tf_idf = np.round(np.multiply(_2antivm_tf, _2_binary_antivm_calc_idf),4)
    _2_multiple_antivm_calc_idf = np.round(
        248 / (_2antipatch_df + _2antidump_df + _2antivm_df + _2original_df + 0.1), 4)
    _2_multiple_antivm_tf_idf = np.round(np.multiply(_2antivm_tf, _2_multiple_antivm_calc_idf),4)

    _3antipatch_calc_tf = np.round(_3antipatch_tf / np.sum(_3antipatch_tf), 4)
    _3_binary_antipatch_calc_idf = np.round(124 / (_3antipatch_df + _3original_df + 0.1), 4)
    _3_binary_antipatch_tf_idf = np.round(np.multiply(_3antipatch_tf, _3_binary_antipatch_calc_idf),4)
    _3_multiple_antipatch_calc_idf = np.round(
        248 / (_3antipatch_df + _3antidump_df + _3antivm_df + _3original_df + 0.1), 4)
    _3_multiple_antipatch_tf_idf = np.round(np.multiply(_3antipatch_tf, _3_multiple_antipatch_calc_idf),4)

    _3antidump_calc_tf = np.round(_3antidump_tf / np.sum(_3antidump_tf), 4)
    _3_binary_antidump_calc_idf = np.round(124 / (_3antidump_df + _3original_df + 0.1), 4)
    _3_binary_antidump_tf_idf = np.round(np.multiply(_3antidump_tf, _3_binary_antidump_calc_idf),4)
    _3_multiple_antidump_calc_idf = np.round(
        248 / (_3antipatch_df + _3antidump_df + _3antivm_df + _3original_df + 0.1), 4)
    _3_multiple_antidump_tf_idf = np.round(np.multiply(_3antidump_tf, _3_multiple_antidump_calc_idf),4)

    _3antivm_calc_tf = np.round(_3antivm_tf / np.sum(_3antivm_tf), 4)
    _3_binary_antivm_calc_idf = np.round(124 / (_3antivm_df + _3original_df + 0.1), 4)
    _3_binary_antivm_tf_idf = np.round(np.multiply(_3antivm_tf, _3_binary_antivm_calc_idf),4)
    _3_multiple_antivm_calc_idf = np.round(
        248 / (_3antipatch_df + _3antidump_df + _3antivm_df + _3original_df + 0.1), 4)
    _3_multiple_antivm_tf_idf = np.round(np.multiply(_3antivm_tf, _3_multiple_antivm_calc_idf),4)

    _4antipatch_calc_tf = np.round(_4antipatch_tf / np.sum(_4antipatch_tf), 4)
    _4_binary_antipatch_calc_idf = np.round(124 / (_4antipatch_df + _4original_df + 0.1), 4)
    _4_binary_antipatch_tf_idf = np.round(np.multiply(_4antipatch_tf, _4_binary_antipatch_calc_idf),4)
    _4_multiple_antipatch_calc_idf = np.round(
        248 / (_4antipatch_df + _4antidump_df + _4antivm_df + _4original_df + 0.1), 4)
    _4_multiple_antipatch_tf_idf = np.round(np.multiply(_4antipatch_tf, _4_multiple_antipatch_calc_idf),4)

    _4antidump_calc_tf = np.round(_4antidump_tf / np.sum(_4antidump_tf), 4)
    _4_binary_antidump_calc_idf = np.round(124 / (_4antidump_df + _4original_df + 0.1), 4)
    _4_binary_antidump_tf_idf = np.round(np.multiply(_4antidump_tf, _4_binary_antidump_calc_idf),4)
    _4_multiple_antidump_calc_idf = np.round(
        248 / (_4antipatch_df + _4antidump_df + _4antivm_df + _4original_df + 0.1), 4)
    _4_multiple_antidump_tf_idf = np.round(np.multiply(_4antidump_tf, _4_multiple_antidump_calc_idf),4)

    _4antivm_calc_tf = np.round(_4antivm_tf / np.sum(_4antivm_tf), 4)
    _4_binary_antivm_calc_idf = np.round(124 / (_4antivm_df + _4original_df + 0.1), 4)
    _4_binary_antivm_tf_idf = np.round(np.multiply(_4antivm_tf, _4_binary_antivm_calc_idf),4)
    _4_multiple_antivm_calc_idf = np.round(
        248 / (_4antipatch_df + _4antidump_df + _4antivm_df + _4original_df + 0.1), 4)
    _4_multiple_antivm_tf_idf = np.round(np.multiply(_4antivm_tf, _4_multiple_antivm_calc_idf),4)


    _2original_calc_tf=np.round(_2original_tf/np.sum(_2original_tf),4)
    _3original_calc_tf = np.round(_3original_tf / np.sum(_3original_tf), 4)
    _4original_calc_tf = np.round(_4original_tf / np.sum(_4original_tf), 4)


    # 2gram-antipatch
    data = np.stack((_2antipatch_calc_tf, _2original_calc_tf, _2antipatch_df, _2original_df, _2_binary_antipatch_tf_idf,
                     _2_multiple_antipatch_tf_idf), axis=1)
    data_df = pd.DataFrame(data, columns=["2gram_antipatch_TF", "2gram_original_TF", "2gram_antipatch_DF",
                                          "2gram_original_DF", "2gram_binary_antipatch_TF-IDF",
                                          "2gram_multiple_antipatch_TF-IDF"],
                           index=total_2gram_opcode)
    data_df.to_pickle('2gram_antipatch.pkl')
    rows1 = zip(key2List, _2antipatch_calc_tf.tolist(), _2original_calc_tf.tolist(), _2antipatch_df.tolist(),
                _2original_df.tolist(), _2_binary_antipatch_tf_idf.tolist(), _2_multiple_antipatch_tf_idf.tolist())
    with open('2gram_antipatch_result.csv', 'w', encoding='utf-8') as f:
        w1 = csv.writer(f)
        w1.writerow(["2gram_opcode", "2gram_antipatch_TF", "2gram_original_TF", "2gram_antipatch_DF", "2gram_original_DF",
                     "2gram_binary_antipatch_TF-IDF", "2gram_multiple_antipatch_TF-IDF"])
        for row in rows1:
            w1.writerow(row)

    #2gram-antidump
    data=np.stack((_2antidump_calc_tf, _2original_calc_tf, _2antidump_df, _2original_df, _2_binary_antidump_tf_idf, _2_multiple_antidump_tf_idf), axis=1)
    data_df = pd.DataFrame(data, columns=["2gram_antidump_TF", "2gram_original_TF", "2gram_antidump_DF", "2gram_original_DF", "2gram_binary_antidump_TF-IDF", "2gram_multiple_antidump_TF-IDF"],
                           index=total_2gram_opcode)
    data_df.to_pickle('2gram_antidump.pkl')
    rows2 = zip(key2List, _2antidump_calc_tf.tolist(), _2original_calc_tf.tolist(), _2antidump_df.tolist(), _2original_df.tolist(), _2_binary_antidump_tf_idf.tolist(), _2_multiple_antidump_tf_idf.tolist())
    with open('2gram_antidump_result.csv', 'w', encoding='utf-8') as f:
        w2 = csv.writer(f)
        w2.writerow(["2gram_opcode", "2gram_antidump_TF", "2gram_original_TF", "2gram_antidump_DF", "2gram_original_DF", "2gram_binary_antidump_TF-IDF", "2gram_multiple_antidump_TF-IDF"])
        for row in rows2:
            w2.writerow(row)

    #2gram-antivm
    data=np.stack((_2antivm_calc_tf, _2original_calc_tf, _2antivm_df, _2original_df, _2_binary_antivm_tf_idf, _2_multiple_antivm_tf_idf), axis=1)
    data_df = pd.DataFrame(data, columns=["2gram_antivm_TF", "2gram_original_TF", "2gram_antivm_DF", "2gram_original_DF", "2gram_binary_antivm_TF-IDF", "2gram_multiple_antivm_TF-IDF"],
                           index=total_2gram_opcode)
    data_df.to_pickle('2gram_antivm.pkl')
    rows3 = zip(key2List, _2antivm_calc_tf.tolist(), _2original_calc_tf.tolist(), _2antivm_df.tolist(), _2original_df.tolist(), _2_binary_antivm_tf_idf.tolist(), _2_multiple_antivm_tf_idf.tolist())
    with open('2gram_antivm_result.csv', 'w', encoding='utf-8') as f:
        w3 = csv.writer(f)
        w3.writerow(["2gram_opcode", "2gram_antivm_TF", "2gram_original_TF", "2gram_antivm_DF", "2gram_original_DF", "2gram_binary_antivm_TF-IDF", "2gram_multiple_antivm_TF-IDF"])
        for row in rows3:
            w3.writerow(row)


    # 3gram-antipatch
    data = np.stack((_3antipatch_calc_tf, _3original_calc_tf, _3antipatch_df, _3original_df, _3_binary_antipatch_tf_idf,
                     _3_multiple_antipatch_tf_idf), axis=1)
    data_df = pd.DataFrame(data, columns=["3gram_antipatch_TF", "3gram_original_TF", "3gram_antipatch_DF",
                                          "3gram_original_DF", "3gram_binary_antipatch_TF-IDF",
                                          "3gram_multiple_antipatch_TF-IDF"],
                           index=total_3gram_opcode)
    data_df.to_pickle('3gram_antipatch.pkl')
    rows4 = zip(key3List, _3antipatch_calc_tf.tolist(), _3original_calc_tf.tolist(), _3antipatch_df.tolist(),
                _3original_df.tolist(), _3_binary_antipatch_tf_idf.tolist(), _3_multiple_antipatch_tf_idf.tolist())
    with open('3gram_antipatch_result.csv', 'w', encoding='utf-8') as f:
        w4 = csv.writer(f)
        w4.writerow(["3gram_opcode", "3gram_antipatch_TF", "3gram_original_TF", "3gram_antipatch_DF", "3gram_original_DF",
                     "3gram_binary_antipatch_TF-IDF", "3gram_multiple_antipatch_TF-IDF"])
        for row in rows4:
            w4.writerow(row)

    #3gram-antidump
    data=np.stack((_3antidump_calc_tf, _3original_calc_tf, _3antidump_df, _3original_df, _3_binary_antidump_tf_idf, _3_multiple_antidump_tf_idf), axis=1)
    data_df = pd.DataFrame(data, columns=["3gram_antidump_TF", "3gram_original_TF", "3gram_antidump_DF", "3gram_original_DF", "3gram_binary_antidump_TF-IDF", "3gram_multiple_antidump_TF-IDF"],
                           index=total_3gram_opcode)
    data_df.to_pickle('3gram_antidump.pkl')
    rows5 = zip(key3List, _3antidump_calc_tf.tolist(), _3original_calc_tf.tolist(), _3antidump_df.tolist(), _3original_df.tolist(), _3_binary_antidump_tf_idf.tolist(), _3_multiple_antidump_tf_idf.tolist())
    with open('3gram_antidump_result.csv', 'w', encoding='utf-8') as f:
        w5 = csv.writer(f)
        w5.writerow(["3gram_opcode", "3gram_antidump_TF", "3gram_original_TF", "3gram_antidump_DF", "3gram_original_DF", "3gram_binary_antidump_TF-IDF", "3gram_multiple_antidump_TF-IDF"])
        for row in rows5:
            w5.writerow(row)

    #3gram-antivm
    data=np.stack((_3antivm_calc_tf, _3original_calc_tf, _3antivm_df, _3original_df, _3_binary_antivm_tf_idf, _3_multiple_antivm_tf_idf), axis=1)
    data_df = pd.DataFrame(data, columns=["3gram_antivm_TF", "3gram_original_TF", "3gram_antivm_DF", "3gram_original_DF", "3gram_binary_antivm_TF-IDF", "3gram_multiple_antivm_TF-IDF"],
                           index=total_3gram_opcode)
    data_df.to_pickle('3gram_antivm.pkl')
    rows6 = zip(key3List, _3antivm_calc_tf.tolist(), _3original_calc_tf.tolist(), _3antivm_df.tolist(), _3original_df.tolist(), _3_binary_antivm_tf_idf.tolist(), _3_multiple_antivm_tf_idf.tolist())
    with open('3gram_antivm_result.csv', 'w', encoding='utf-8') as f:
        w6 = csv.writer(f)
        w6.writerow(["3gram_opcode", "3gram_antivm_TF", "3gram_original_TF", "3gram_antivm_DF", "3gram_original_DF", "3gram_binary_antivm_TF-IDF", "3gram_multiple_antivm_TF-IDF"])
        for row in rows6:
            w6.writerow(row)

    # 4gram-antipatch
    data = np.stack((_4antipatch_calc_tf, _4original_calc_tf, _4antipatch_df, _4original_df, _4_binary_antipatch_tf_idf,
                     _4_multiple_antipatch_tf_idf), axis=1)
    data_df = pd.DataFrame(data, columns=["4gram_antipatch_TF", "4gram_original_TF", "4gram_antipatch_DF",
                                          "4gram_original_DF", "4gram_binary_antipatch_TF-IDF",
                                          "4gram_multiple_antipatch_TF-IDF"],
                           index=total_4gram_opcode)
    data_df.to_pickle('4gram_antipatch.pkl')
    rows7 = zip(key4List, _4antipatch_calc_tf.tolist(), _4original_calc_tf.tolist(), _4antipatch_df.tolist(),
                _4original_df.tolist(), _4_binary_antipatch_tf_idf.tolist(), _4_multiple_antipatch_tf_idf.tolist())
    with open('4gram_antipatch_result.csv', 'w', encoding='utf-8') as f:
        w7 = csv.writer(f)
        w7.writerow(["4gram_opcode", "4gram_antipatch_TF", "4gram_original_TF", "4gram_antipatch_DF", "4gram_original_DF",
                     "4gram_binary_antipatch_TF-IDF", "4gram_multiple_antipatch_TF-IDF"])
        for row in rows7:
            w7.writerow(row)

    #4gram-antidump
    data=np.stack((_4antidump_calc_tf, _4original_calc_tf, _4antidump_df, _4original_df, _4_binary_antidump_tf_idf, _4_multiple_antidump_tf_idf), axis=1)
    data_df = pd.DataFrame(data, columns=["4gram_antidump_TF", "4gram_original_TF", "4gram_antidump_DF", "4gram_original_DF", "4gram_binary_antidump_TF-IDF", "4gram_multiple_antidump_TF-IDF"],
                           index=total_4gram_opcode)
    data_df.to_pickle('4gram_antidump.pkl')
    rows8 = zip(key4List, _4antidump_calc_tf.tolist(), _4original_calc_tf.tolist(), _4antidump_df.tolist(), _4original_df.tolist(), _4_binary_antidump_tf_idf.tolist(), _4_multiple_antidump_tf_idf.tolist())
    with open('4gram_antidump_result.csv', 'w', encoding='utf-8') as f:
        w8 = csv.writer(f)
        w8.writerow(["4gram_opcode", "4gram_antidump_TF", "4gram_original_TF", "4gram_antidump_DF", "4gram_original_DF", "4gram_binary_antidump_TF-IDF", "4gram_multiple_antidump_TF-IDF"])
        for row in rows8:
            w8.writerow(row)

    #4gram-antivm
    data=np.stack((_4antivm_calc_tf, _4original_calc_tf, _4antivm_df, _4original_df, _4_binary_antivm_tf_idf, _4_multiple_antivm_tf_idf), axis=1)
    data_df = pd.DataFrame(data, columns=["4gram_antivm_TF", "4gram_original_TF", "4gram_antivm_DF", "4gram_original_DF", "4gram_binary_antivm_TF-IDF", "4gram_multiple_antivm_TF-IDF"],
                           index=total_4gram_opcode)
    data_df.to_pickle('4gram_antivm.pkl')
    rows9 = zip(key4List, _4antivm_calc_tf.tolist(), _4original_calc_tf.tolist(), _4antivm_df.tolist(), _4original_df.tolist(), _4_binary_antivm_tf_idf.tolist(), _4_multiple_antivm_tf_idf.tolist())
    with open('4gram_antivm_result.csv', 'w', encoding='utf-8') as f:
        w9 = csv.writer(f)
        w9.writerow(["4gram_opcode", "4gram_antivm_TF", "4gram_original_TF", "4gram_antivm_DF", "4gram_original_DF", "4gram_binary_antivm_TF-IDF", "4gram_multiple_antivm_TF-IDF"])
        for row in rows9:
            w9.writerow(row)
    """
    # 2gram-total
    data = np.stack((_2antipatch_df, _2antidump_df,_2antivm_df,_2original_df),axis=1)
    data_df = pd.DataFrame(data, columns=["2gram_antipatch_DF", "2gram_antidump_DF", "2gram_antivm_DF","2gram_original_DF"],
                           index=total_2gram_opcode)
    data_df.to_pickle('2gram_total.pkl')

    # 2gram-total
    data = np.stack((_2antipatch_calc_tf, _2antidump_calc_tf, _2antivm_calc_tf, _2original_calc_tf), axis=1)
    data_tf = pd.DataFrame(data,
                           columns=["2gram_antipatch_calc_tf", "2gram_antidump_calc_tf", "2gram_antivm_calc_tf", "2gram_original_calc_tf"],
                           index=total_2gram_opcode)
    data_tf.to_pickle('2gram_total_tf.pkl')

    # 3gram-total
    data = np.stack((_3antipatch_df, _3antidump_df, _3antivm_df, _3original_df), axis=1)
    data_df = pd.DataFrame(data,
                           columns=["3gram_antipatch_DF", "3gram_antidump_DF", "3gram_antivm_DF", "3gram_original_DF"],
                           index=total_3gram_opcode)
    data_df.to_pickle('3gram_total.pkl')

    # 4gram-total
    data = np.stack((_4antipatch_df, _4antidump_df, _4antivm_df, _4original_df), axis=1)
    data_df = pd.DataFrame(data,
                           columns=["4gram_antipatch_DF", "4gram_antidump_DF", "4gram_antivm_DF", "4gram_original_DF"],
                           index=total_4gram_opcode)
    data_df.to_pickle('4gram_total.pkl')
    """
if __name__=="__main__":
    main()