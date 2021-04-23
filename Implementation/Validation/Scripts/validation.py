import pandas as pd
import sklearn.metrics as sk
from sklearn.metrics import precision_recall_fscore_support as score
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


original_paths = {
    1: "./Models/data/1.csv",
    2: "./Models/data/1.csv",
    3: "./Models/data/1.csv",
    4: "./Models/data/5.csv",
    5: "./Models/data/5.csv",
    6: "./Models/data/5.csv",
    7: "./Models/data/1.csv",
    8: "./Models/data/1.csv",
    9: "./Models/data/1.csv",
    10: "./Models/data/6.csv",
    11: "./Models/data/6.csv",
    12: "./Models/data/6.csv",
    13: "./Models/data/3.csv",
    14: "./Models/data/3.csv",
    15: "./Models/data/3.csv",
    16: "./Models/data/1.csv",
    17: "./Models/data/1.csv",
    18: "./Models/data/1.csv",
    19: "./Models/data/5.csv",
    20: "./Models/data/5.csv",
    21: "./Models/data/5.csv",
    22: "./Models/data/3.csv",
    23: "./Models/data/3.csv",
    24: "./Models/data/3.csv",
    25: "./Models/data/2.csv",
    26: "./Models/data/2.csv",
    27: "./Models/data/2.csv",
    28: "./Models/data/2.csv",
    29: "./Models/data/2.csv",
    30: "./Models/data/2.csv",
    31: "./Models/data/1.csv",
    32: "./Models/data/1.csv",
    33: "./Models/data/1.csv",
    34: "./Models/data/6.csv",
    35: "./Models/data/6.csv",
    36: "./Models/data/6.csv",
    37: "./Models/data/4.csv",
    38: "./Models/data/4.csv",
    39: "./Models/data/4.csv",
    40: "./Models/data/4.csv",
    41: "./Models/data/4.csv",
    42: "./Models/data/4.csv",
    43: "./Models/data/6.csv",
    44: "./Models/data/6.csv",
    45: "./Models/data/6.csv",
    46: "./Models/data/6.csv",
    47: "./Models/data/6.csv",
    48: "./Models/data/6.csv",
    49: "./Models/data/3.csv",
    50: "./Models/data/3.csv",
    51: "./Models/data/3.csv",
    52: "./Models/data/4.csv",
    53: "./Models/data/4.csv",
    54: "./Models/data/4.csv",
    55: "./Models/data/3.csv",
    56: "./Models/data/3.csv",
    57: "./Models/data/3.csv",
    58: "./Models/data/2.csv",
    59: "./Models/data/2.csv",
    60: "./Models/data/2.csv",
    61: "./Models/data/1.csv",
    62: "./Models/data/1.csv",
    63: "./Models/data/1.csv",
    64: "./Models/data/4.csv",
    65: "./Models/data/4.csv",
    66: "./Models/data/4.csv",
    67: "./Models/data/6.csv",
    68: "./Models/data/6.csv",
    69: "./Models/data/6.csv",
    70: "./Models/data/2.csv",
    71: "./Models/data/2.csv",
    72: "./Models/data/2.csv",
    73: "./Models/data/2.csv",
    74: "./Models/data/2.csv",
    75: "./Models/data/2.csv",
    76: "./Models/data/5.csv",
    77: "./Models/data/5.csv",
    78: "./Models/data/5.csv",
    79: "./Models/data/4.csv",
    80: "./Models/data/4.csv",
    81: "./Models/data/4.csv",
    82: "./Models/data/5.csv",
    83: "./Models/data/5.csv",
    84: "./Models/data/5.csv",
    85: "./Models/data/3.csv",
    86: "./Models/data/3.csv",
    87: "./Models/data/3.csv",
    88: "./Models/data/5.csv",
    89: "./Models/data/5.csv",
    90: "./Models/data/5.csv"
}

fscores_baseline = []
fscores_collaborative = []
output = ""
fscores_collaborative2 = []

times_local = []
times_collaborative = []
times_collaborative2 = []


tabularA = ""
tabularB = ""

mode = 1
for i in range(1,91):

    original_path = original_paths[i]
    original_dataset = pd.read_csv(original_path, sep =',')
    original_dataset = original_dataset[['SubClass']]

    result_path = f"./Models/new_results/{i}_result.csv"
    result_dataset = pd.read_csv(result_path, sep =',', names=['Row', 'Prediction'])
    result_dataset.set_index('Row', inplace=True)


    #The encoder will encode the different subclasses as following:
    classes = {
        "Flooding": 0,
        "Fuzzing": 1,
        "Normal": 2,
        "Replay": 3,
        "Spoofing": 4
    }

    original_dataset['SubClass'] = original_dataset['SubClass'].map(classes)

    dataset = original_dataset.join(result_dataset, how='left')
    dataset = dataset.fillna(classes['Normal'])
    dataset = dataset.astype('int32')

    actual_values = dataset['SubClass'].to_numpy()
    predicted_values = dataset['Prediction'].to_numpy()

    #f1 = sk.f1_score(actual_values, predicted_values, average='macro')
    #print(f1)

    #np.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})
    precision, recall, fscore, support = score(actual_values, predicted_values)

    """
    print(f"Simulation run {i}:")
    print('Columns: | Flooding | Fuzzing | Normal | Replay | Spoofing |')
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    print(f"fscore average: {sum(fscore) / len(fscore)} \n \n")
    """

    average_fscore = sum(fscore) / len(fscore)
    if(mode == 2):
        fscores_collaborative.append(average_fscore)
    elif(mode==3):
        fscores_collaborative2.append(average_fscore)    
    elif(mode==1):
        fscores_baseline.append(average_fscore)


    metadata_list = []
    with open(f"./Models/new_results/{i}_metadata.txt", "r") as file:
        for line in file:
            s = line.split(":")
            metadata_list.append(str(s[1]).rstrip("\n"))

    e = " "
    tabularA+=(f"Simulation run {i},{e},{e},{e},{e},{e}\n")
    tabularA+=(f"{e},Flooding,Fuzzing,Normal,Replay,Spoofing\n")
    tabularA+=(f"Precision,{precision[0]},{precision[1]},{precision[2]},{precision[3]},{precision[4]}\n")
    tabularA+=(f"Recall,{recall[0]},{recall[1]},{recall[2]},{recall[3]},{recall[4]}\n")
    tabularA+=(f"F1-score,{fscore[0]},{fscore[1]},{fscore[2]},{fscore[3]},{fscore[4]}\n")
    tabularA+=(f"F1-score average,{average_fscore},{e},{e},{e},{e}\n")
    tabularA+=(f"Average local time(s),{metadata_list[0]},{e},{e},{e},{e}\n")
    tabularA+=(f"Average consultation time(s),{metadata_list[1]},{e},{e},{e},{e}\n")
    tabularA+=(f"Total number of CAN-frames,{metadata_list[2]},{e},{e},{e},{e}\n")
    tabularA+=(f"Total number of security events,{metadata_list[3]},{e},{e},{e},{e}\n")
    tabularA+=(f"Consultation messages sent,{metadata_list[4]},{e},{e},{e},{e}\n")
    tabularA+=(f"{e},{e},{e},{e},{e},{e}\n")

    output += f"Simulation run {i}: \nColumns: | Flooding | Fuzzing | Normal | Replay | Spoofing |\nprecision: {precision}'\nrecall: {recall}\nfscore: {fscore}\nsupport: {support}\nfscore average: {sum(fscore) / len(fscore)} \n \n"

    times_local.append(float(metadata_list[0]))
    if(mode == 2):
        times_collaborative.append(float(metadata_list[1]))
    elif(mode == 3):
        times_collaborative2.append(float(metadata_list[1]))
    

    mode = mode+1
    if(mode == 4):
        mode = 1


output+="Confidence intervals for the different times:\n"
output+=f"Average local times: {stats.t.interval(0.95, len(times_local)-1, loc=np.mean(times_local), scale=stats.sem(times_local))}     MEAN: {sum(times_local)/len(times_local)}\n"
output+=f"Average consultation times: {stats.t.interval(0.95, len(times_collaborative)-1, loc=np.mean(times_collaborative), scale=stats.sem(times_collaborative))}     MEAN: {sum(times_collaborative)/len(times_collaborative)}\n"
output+=f"Average consultation 2 times: {stats.t.interval(0.95, len(times_collaborative2)-1, loc=np.mean(times_collaborative2), scale=stats.sem(times_collaborative2))}     MEAN: {sum(times_collaborative2)/len(times_collaborative2)}\n"
output += "\n \n fscores summarized: \n|      Baseline      |    Collaborative   |    Collaborative2   |\n"
for i in range(0,30):
    output += f"| {fscores_baseline[i]} | {fscores_collaborative[i]} | {fscores_collaborative2[i]} \n"


print(fscores_baseline)
print(fscores_collaborative)


output += "\n \nOk, now lets do a couple of Shapiro-Wilks tests to see if these samples comes from a normal distribution (Large p-value over 0.05 --> Data comes from normal distribution):\n"
output += f"Baseline: {stats.shapiro(fscores_baseline)} \n"
output += f"Collaborative: {stats.shapiro(fscores_collaborative)} \n"

print(stats.shapiro(fscores_baseline))
print(stats.shapiro(fscores_collaborative))
print(stats.shapiro(fscores_collaborative2))



#Does not appear to be normally distributed, so we will have to use the Wilcoxon signed-rank test.?
plt.hist(fscores_baseline,density=1, bins=30)
plt.show()
plt.hist(fscores_collaborative,density=1, bins=30)
plt.show()
plt.hist(fscores_collaborative2,density=1, bins=30)
plt.show()


tabularB += "Baseline,,Collaborative 1,,Collaborative 2,\n"
tabularB += "Simulation run,F1-score,Simulation run,F1-score,Simulation run,F1-score\n"
j = 0
for i in range(0,len(fscores_baseline)):
    tabularB += f"{j+1},{round(fscores_baseline[i],6)},{j+2},{round(fscores_collaborative[i],6)},{j+3},{round(fscores_collaborative2[i],6)}\n"
    j = j+3

#Then start the save itself
with open("./Models/validation.txt", "w") as file:
        file.write(output)

with open("./Models/tabularA.csv", "w") as file:
    file.write(tabularA)

with open("./Models/tabularB.csv", "w") as file:
    file.write(tabularB)

with open("./Models/f1-scores.csv", "w") as file:
    for value in fscores_baseline:
        file.write(f"1,{value}\n")
    for value in fscores_collaborative:
        file.write(f"2,{value}\n")
    for value in fscores_collaborative2:
        file.write(f"3,{value}\n")

with open("./Models/Anova.csv", "w") as file:
    file.write("triplet_nr,Mode,F1-score\n")
    for i in range(0,30):
        file.write(f"{i+1},Baseline,{fscores_baseline[i]}\n")
        file.write(f"{i+1},Collaborative_1,{fscores_collaborative[i]}\n")
        file.write(f"{i+1},Collaborative_2,{fscores_collaborative2[i]}\n")