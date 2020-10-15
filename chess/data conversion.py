import pandas as pd

df = pd.read_csv("agaricus-lepiota.data", header=None)
# df[0] = df[0].replace("p",2)
# df[0] = df[0].replace("e",1)
#
# print(df[0])
# file = open("mushroomCom.csv", "w")
# for item in range(len(df[0])):
#     string = str(item) + ", " + str(df[0].iloc[item]) + "\n"
#     file.write(string)
# file.close()

df.drop(columns=0, inplace=True)
df.drop(columns=11, inplace=True)
print(df)
