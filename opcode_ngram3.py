import os
from tqdm import tqdm
from collections import defaultdict
import csv
import pandas as pd

_2gram_antipatch_df = pd.read_pickle("./2gram_antipatch.pkl")

_2gram_antipatch_df.info()

cond1=(_2gram_antipatch_df['2gram_antipatch_DF']>30)
cond2=(_2gram_antipatch_df['2gram_original_DF']>30)
cond3=(_2gram_antipatch_df['2gram_antipatch_DF']==0)
cond4=(_2gram_antipatch_df['2gram_original_DF']<30)
cond5=(_2gram_antipatch_df['2gram_antipatch_DF']<30)
cond6=(_2gram_antipatch_df['2gram_original_DF']==0)
_2gram_antipatch_df=_2gram_antipatch_df[(~(cond1&cond2))&(~(cond3&cond4))&(~(cond5&cond6))]
_2gram_antipatch_binary_sorted=_2gram_antipatch_df.sort_values(by=['2gram_binary_antipatch_TF-IDF'], ascending=False)
_2gram_antipatch_multiple_sorted=_2gram_antipatch_df.sort_values(by=['2gram_multiple_antipatch_TF-IDF'], ascending=False)

f=open("./_2gram_antipatch_binary_feature_selected",'w')
f.write(str(_2gram_antipatch_binary_sorted.index.tolist()).replace(" ","\n"))
f.close
f=open("./_2gram_antipatch_multiple_feature_selected",'w')
f.write(str(_2gram_antipatch_multiple_sorted.index.tolist()).replace(" ","\n"))
f.close

_2gram_antidump_df = pd.read_pickle("./2gram_antidump.pkl")

_2gram_antidump_df.info()

cond1=(_2gram_antidump_df['2gram_antidump_DF']>30)
cond2=(_2gram_antidump_df['2gram_original_DF']>30)
cond3=(_2gram_antidump_df['2gram_antidump_DF']==0)
cond4=(_2gram_antidump_df['2gram_original_DF']<30)
cond5=(_2gram_antidump_df['2gram_antidump_DF']<30)
cond6=(_2gram_antidump_df['2gram_original_DF']==0)
_2gram_antidump_df=_2gram_antidump_df[(~(cond1&cond2))&(~(cond3&cond4))&(~(cond5&cond6))]
_2gram_antidump_binary_sorted=_2gram_antidump_df.sort_values(by=['2gram_binary_antidump_TF-IDF'], ascending=False)
_2gram_antidump_multiple_sorted=_2gram_antidump_df.sort_values(by=['2gram_multiple_antidump_TF-IDF'], ascending=False)

f=open("./_2gram_antidump_binary_feature_selected",'w')
f.write(str(_2gram_antidump_binary_sorted.index.tolist()).replace(" ","\n"))
f.close

f=open("./_2gram_antidump_multiple_feature_selected",'w')
f.write(str(_2gram_antidump_multiple_sorted.index.tolist()).replace(" ","\n"))
f.close

_2gram_antivm_df = pd.read_pickle("./2gram_antivm.pkl")

_2gram_antivm_df.info()

cond1=(_2gram_antivm_df['2gram_antivm_DF']>30)
cond2=(_2gram_antivm_df['2gram_original_DF']>30)
cond3=(_2gram_antivm_df['2gram_antivm_DF']==0)
cond4=(_2gram_antivm_df['2gram_original_DF']<30)
cond5=(_2gram_antivm_df['2gram_antivm_DF']<30)
cond6=(_2gram_antivm_df['2gram_original_DF']==0)
_2gram_antivm_df=_2gram_antivm_df[(~(cond1&cond2))&(~(cond3&cond4))&(~(cond5&cond6))]
_2gram_antivm_binary_sorted=_2gram_antivm_df.sort_values(by=['2gram_binary_antivm_TF-IDF'], ascending=False)
_2gram_antivm_multiple_sorted=_2gram_antivm_df.sort_values(by=['2gram_multiple_antivm_TF-IDF'], ascending=False)

f=open("./_2gram_antivm_binary_feature_selected",'w')
f.write(str(_2gram_antivm_binary_sorted.index.tolist()).replace(" ","\n"))
f.close

f=open("./_2gram_antivm_multiple_feature_selected",'w')
f.write(str(_2gram_antivm_multiple_sorted.index.tolist()).replace(" ","\n"))
f.close

_3gram_antipatch_df = pd.read_pickle("./3gram_antipatch.pkl")

_3gram_antipatch_df.info()

cond1=(_3gram_antipatch_df['3gram_antipatch_DF']>30)
cond2=(_3gram_antipatch_df['3gram_original_DF']>30)
cond3=(_3gram_antipatch_df['3gram_antipatch_DF']==0)
cond4=(_3gram_antipatch_df['3gram_original_DF']<30)
cond5=(_3gram_antipatch_df['3gram_antipatch_DF']<30)
cond6=(_3gram_antipatch_df['3gram_original_DF']==0)
_3gram_antipatch_df=_3gram_antipatch_df[(~(cond1&cond2))&(~(cond3&cond4))&(~(cond5&cond6))]
_3gram_antipatch_binary_sorted=_3gram_antipatch_df.sort_values(by=['3gram_binary_antipatch_TF-IDF'], ascending=False)
_3gram_antipatch_multiple_sorted=_3gram_antipatch_df.sort_values(by=['3gram_multiple_antipatch_TF-IDF'], ascending=False)

f=open("./_3gram_antipatch_binary_feature_selected",'w')
f.write(str(_3gram_antipatch_binary_sorted.index.tolist()).replace(" ","\n"))
f.close

f=open("./_3gram_antipatch_multiple_feature_selected",'w')
f.write(str(_3gram_antipatch_multiple_sorted.index.tolist()).replace(" ","\n"))
f.close

_3gram_antidump_df = pd.read_pickle("./3gram_antidump.pkl")

_3gram_antidump_df.info()

cond1=(_3gram_antidump_df['3gram_antidump_DF']>30)
cond2=(_3gram_antidump_df['3gram_original_DF']>30)
cond3=(_3gram_antidump_df['3gram_antidump_DF']==0)
cond4=(_3gram_antidump_df['3gram_original_DF']<30)
cond5=(_3gram_antidump_df['3gram_antidump_DF']<30)
cond6=(_3gram_antidump_df['3gram_original_DF']==0)
_3gram_antidump_df=_3gram_antidump_df[(~(cond1&cond2))&(~(cond3&cond4))&(~(cond5&cond6))]
_3gram_antidump_binary_sorted=_3gram_antidump_df.sort_values(by=['3gram_binary_antidump_TF-IDF'], ascending=False)
_3gram_antidump_multiple_sorted=_3gram_antidump_df.sort_values(by=['3gram_multiple_antidump_TF-IDF'], ascending=False)

f=open("./_3gram_antidump_binary_feature_selected",'w')
f.write(str(_3gram_antidump_binary_sorted.index.tolist()).replace(" ","\n"))
f.close

f=open("./_3gram_antidump_multiple_feature_selected",'w')
f.write(str(_3gram_antidump_multiple_sorted.index.tolist()).replace(" ","\n"))
f.close

_3gram_antivm_df = pd.read_pickle("./3gram_antivm.pkl")

_3gram_antivm_df.info()

cond1=(_3gram_antivm_df['3gram_antivm_DF']>30)
cond2=(_3gram_antivm_df['3gram_original_DF']>30)
cond3=(_3gram_antivm_df['3gram_antivm_DF']==0)
cond4=(_3gram_antivm_df['3gram_original_DF']<30)
cond5=(_3gram_antivm_df['3gram_antivm_DF']<30)
cond6=(_3gram_antivm_df['3gram_original_DF']==0)
_3gram_antivm_df=_3gram_antivm_df[(~(cond1&cond2))&(~(cond3&cond4))&(~(cond5&cond6))]
_3gram_antivm_binary_sorted=_3gram_antivm_df.sort_values(by=['3gram_binary_antivm_TF-IDF'], ascending=False)
_3gram_antivm_multiple_sorted=_3gram_antivm_df.sort_values(by=['3gram_multiple_antivm_TF-IDF'], ascending=False)

f=open("./_3gram_antivm_binary_feature_selected",'w')
f.write(str(_3gram_antivm_binary_sorted.index.tolist()).replace(" ","\n"))
f.close

f=open("./_3gram_antivm_multiple_feature_selected",'w')
f.write(str(_3gram_antivm_multiple_sorted.index.tolist()).replace(" ","\n"))
f.close

_4gram_antipatch_df = pd.read_pickle("./4gram_antipatch.pkl")

_4gram_antipatch_df.info()

cond1=(_4gram_antipatch_df['4gram_antipatch_DF']>30)
cond2=(_4gram_antipatch_df['4gram_original_DF']>30)
cond3=(_4gram_antipatch_df['4gram_antipatch_DF']==0)
cond4=(_4gram_antipatch_df['4gram_original_DF']<30)
cond5=(_4gram_antipatch_df['4gram_antipatch_DF']<30)
cond6=(_4gram_antipatch_df['4gram_original_DF']==0)
_4gram_antipatch_df=_4gram_antipatch_df[(~(cond1&cond2))&(~(cond3&cond4))&(~(cond5&cond6))]
_4gram_antipatch_binary_sorted=_4gram_antipatch_df.sort_values(by=['4gram_binary_antipatch_TF-IDF'], ascending=False)
_4gram_antipatch_multiple_sorted=_4gram_antipatch_df.sort_values(by=['4gram_multiple_antipatch_TF-IDF'], ascending=False)

f=open("./_4gram_antipatch_binary_feature_selected",'w')
f.write(str(_4gram_antipatch_binary_sorted.index.tolist()).replace(" ","\n"))
f.close

f=open("./_4gram_antipatch_multiple_feature_selected",'w')
f.write(str(_4gram_antipatch_multiple_sorted.index.tolist()).replace(" ","\n"))
f.close

_4gram_antidump_df = pd.read_pickle("./4gram_antidump.pkl")

_4gram_antidump_df.info()

cond1=(_4gram_antidump_df['4gram_antidump_DF']>30)
cond2=(_4gram_antidump_df['4gram_original_DF']>30)
cond3=(_4gram_antidump_df['4gram_antidump_DF']==0)
cond4=(_4gram_antidump_df['4gram_original_DF']<30)
cond5=(_4gram_antidump_df['4gram_antidump_DF']<30)
cond6=(_4gram_antidump_df['4gram_original_DF']==0)
_4gram_antidump_df=_4gram_antidump_df[(~(cond1&cond2))&(~(cond3&cond4))&(~(cond5&cond6))]
_4gram_antidump_binary_sorted=_4gram_antidump_df.sort_values(by=['4gram_binary_antidump_TF-IDF'], ascending=False)
_4gram_antidump_multiple_sorted=_4gram_antidump_df.sort_values(by=['4gram_multiple_antidump_TF-IDF'], ascending=False)

f=open("./_4gram_antidump_binary_feature_selected",'w')
f.write(str(_4gram_antidump_binary_sorted.index.tolist()).replace(" ","\n"))
f.close

f=open("./_4gram_antidump_multiple_feature_selected",'w')
f.write(str(_4gram_antidump_multiple_sorted.index.tolist()).replace(" ","\n"))
f.close

_4gram_antivm_df = pd.read_pickle("./4gram_antivm.pkl")

_4gram_antivm_df.info()

cond1=(_4gram_antivm_df['4gram_antivm_DF']>30)
cond2=(_4gram_antivm_df['4gram_original_DF']>30)
cond3=(_4gram_antivm_df['4gram_antivm_DF']==0)
cond4=(_4gram_antivm_df['4gram_original_DF']<30)
cond5=(_4gram_antivm_df['4gram_antivm_DF']<30)
cond6=(_4gram_antivm_df['4gram_original_DF']==0)
_4gram_antivm_df=_4gram_antivm_df[(~(cond1&cond2))&(~(cond3&cond4))&(~(cond5&cond6))]
_4gram_antivm_binary_sorted=_4gram_antivm_df.sort_values(by=['4gram_binary_antivm_TF-IDF'], ascending=False)
_4gram_antivm_multiple_sorted=_4gram_antivm_df.sort_values(by=['4gram_multiple_antivm_TF-IDF'], ascending=False)

f=open("./_4gram_antivm_binary_feature_selected",'w')
f.write(str(_4gram_antivm_binary_sorted.index.tolist()).replace(" ","\n"))
f.close

f=open("./_4gram_antivm_multiple_feature_selected",'w')
f.write(str(_4gram_antivm_multiple_sorted.index.tolist()).replace(" ","\n"))
f.close

print(_2gram_antipatch_df.shape, _2gram_antidump_df.shape, _2gram_antivm_df.shape)
print(_3gram_antipatch_df.shape, _3gram_antidump_df.shape, _3gram_antivm_df.shape)
print(_4gram_antipatch_df.shape, _4gram_antidump_df.shape, _4gram_antivm_df.shape)
