import pandas as pd
def readFile(filepath, filepath1):
    df = pd.read_parquet(filepath)
    df1 = pd.read_parquet(filepath1)
    print(df['part_px'])
    print(df1['thrust'])
    # df1['part_px'][0] = df['part_px'][0]
    # print(df1['part_px'][0])
    del df
    del df1
# readFile('out.csv')
readFile('../data/Bmeson/test.parquet','../data/Bmeson/train.parquet')