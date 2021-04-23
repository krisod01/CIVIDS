
# slumpa 2 siffror mellan 1-6 (baseline) -> spara i datastruktur
# slumpa 3:e siffra mellan 1-6 (collaborative) ->  
# slumpa 4:e siffra mellan 1-6 (collaborative nivå 2)

import random

rng = random.SystemRandom()
rng.randint(1, 6)

# constraints : baseline != collaborative1 != collaborative 2 != network

#baseline1 == baseline in collaborative
#collaborative1 == collaborative 1

baseline = []
collaborative1 = []
collaborative2 = []
i = 1
j = 1
while(i<31):
    networktraffic = rng.randint(1, 6)
    model1 = rng.randint(1,6)

    if networktraffic == model1:
        continue
    if([networktraffic,model1] in baseline):
        print("already existed in baseline, disgarding")
        continue
    baseline.append([networktraffic,model1])
    while(True):
        model2 = rng.randint(1,6)
        if model2 == networktraffic or model2 == model1:
            continue
        if([networktraffic,model1,model2] in collaborative1):
            print("already existed in collaborative1, disgarding")
            continue
        else:
            collaborative1.append([networktraffic,model1,model2])
            while(True):
                model3 = rng.randint(1,6)
                if model3 == model2 or model3 == model1 or model3 == networktraffic:
                    continue
                if([networktraffic,model1,model2,model3] in collaborative2):
                    print("already existed in collaborative2, disgarding")
                    continue
                else:
                    collaborative2.append([networktraffic,model1,model2,model3])
                    break
        break
    i = i + 1
    

for x in collaborative2:
    print("Networktraffic: {}, Model1: {}, Model2: {}, Model3: {}".format(x[0],x[1],x[2],x[3]))

i = 1
header = "Simulation run, Mode, Network traffic, Requesting device model, Partner1 model, Partner2 model\n"

with open("./Models/simulationruns.csv", "w") as file:
    file.write(header)
    for x in collaborative2:
        baselinerow = f"{i},Baseline,{x[0]}, {x[1]}, N/A, N/A\n"
        file.write(baselinerow)
        i = i + 1
        collaborative1row = f"{i},Collaborative1,{x[0]}, {x[1]}, {x[2]}, N/A\n"
        file.write(collaborative1row)
        i = i + 1
        collaborative2row = f"{i},Collaborative2,{x[0]}, {x[1]}, {x[2]}, {x[3]}\n"
        file.write(collaborative2row)
        i= i + 1