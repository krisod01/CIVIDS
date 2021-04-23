# multi-class classification with Keras
import pandas

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np

output = ""


output_print = []

def hex2int(hex_string):
    """
    Converts a hexadecimal string to an integer value
    If this fails (e.g. if input is none or invalid hex string),
    return 0 (since we want to do zero-padding on the dataset,
    as they did in CAN-ADF)
    """

    try:
        return int(str(hex_string),16)
    except:
        return 0


def create_model():
    #Define the model and its various layers
    model = tf.keras.models.Sequential()
    #model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(8, input_dim=11, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax))

    #Set the optimizer, loss-function and metrics
    model.compile(optimizer ='adam',
                loss='sparse_categorical_crossentropy',
                metrics=[
                    'accuracy'
                    #tf.keras.metrics.FalsePositives()
                    #Use more metrics here! Keras has built-in metrics for TP-rate etc. 
                    ])
    return model


#    _____        _          _                 _ _             
#   |  __ \      | |        | |               | (_)            
#   | |  | | __ _| |_ __ _  | | ___   __ _  __| |_ _ __   __ _ 
#   | |  | |/ _` | __/ _` | | |/ _ \ / _` |/ _` | | '_ \ / _` |
#   | |__| | (_| | || (_| | | | (_) | (_| | (_| | | | | | (_| |
#   |_____/ \__,_|\__\__,_| |_|\___/ \__,_|\__,_|_|_| |_|\__, |
#                                                         __/ |
#                                                        |___/ 

for number in range(1,7):
    dataframe = pandas.read_csv(f"Models/data/{number}.csv", sep=',')


    #Create the relevant time columns
    dataframe['relative_time'] = dataframe['Timestamp']-dataframe['Timestamp'].iloc[0]
    dataframe['delta_time'] = dataframe['relative_time'].diff().fillna(dataframe['relative_time'])



    #    _____        _                                                           _             
    #   |  __ \      | |                                                         (_)            
    #   | |  | | __ _| |_ __ _   _ __  _ __ ___ _ __  _ __ ___   ___ ___  ___ ___ _ _ __   __ _ 
    #   | |  | |/ _` | __/ _` | | '_ \| '__/ _ \ '_ \| '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
    #   | |__| | (_| | || (_| | | |_) | | |  __/ |_) | | | (_) | (_|  __/\__ \__ \ | | | | (_| |
    #   |_____/ \__,_|\__\__,_| | .__/|_|  \___| .__/|_|  \___/ \___\___||___/___/_|_| |_|\__, |
    #                           | |            | |                                         __/ |
    #                           |_|            |_|                                        |___/ 



    #Split the data payload into 8 separate intput fields and convert from hexadecimal 
    #Save this into a new dataframe (which we will ultimately use as our input)
    newdata = dataframe.Data.str.split(expand=True)
    #If there are less than 8 fields, do zero-padding
    newdata = newdata.applymap(hex2int)

    #Add the other relevant input fields
    newdata['delta_time'] = dataframe['delta_time']
    newdata['Arbitration_ID'] = dataframe['Arbitration_ID'].apply(hex2int)
    newdata['DLC'] = dataframe['DLC']

    #Retrieve input values as an ndarray
    dataset = newdata.values
    X = dataset.astype(float)

    #Retrieve the target values as an ndarray
    Y = dataframe['SubClass'].values

    #Encode class values as integers 
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    #The encoder will encode the different subclasses as following: 
    # {
    #   Flooding: 0,
    #   Fuzzing: 1,
    #   Normal: 2,
    #   Replay: 3,
    #   Spoofing: 4 
    # }

    #Since our dataset is so imbalanced (Normal messages represent roughly 90% of the dataset)
    # we will apply different weights to the different classes
    # We determine these weights based on the frequency of the class in the training dataset
    weights = [0, 0, 0, 0, 0]
    for value in encoded_Y:
        weights[value]+=1

    i=0
    for weight in weights:
        weights[i] = len(encoded_Y) / weight
        i+=1

    print(weights)


    print("Classes:")
    classes = encoder.inverse_transform([0,1,2,3,4])
    class_weights = {
        0: weights[0],
        1: weights[1],
        2: weights[2],
        3: 0,
        4: weights[4]
    }
    print(classes)
    output_print.append(classes)


    # convert integers to dummy variables (i.e. one hot encoded)
    #dummy_y = np_utils.to_categorical(encoded_Y)

    X_train, X_test, y_train, y_test = train_test_split(X, encoded_Y, test_size=0.33, shuffle=True)
    #print(y_test)


    #Look into how we should normalize our dataset 
    #(should be possible to remember the mean and std and just apply that when predicting induvidual data-points)

    #X_train = tf.keras.utils.normalize(X_train, axis=1)
    #X_test = tf.keras.utils.normalize(X_test, axis=1)



    #    __  __           _      _                       _   _             
    #   |  \/  |         | |    | |                     | | (_)            
    #   | \  / | ___   __| | ___| |   ___ _ __ ___  __ _| |_ _  ___  _ __  
    #   | |\/| |/ _ \ / _` |/ _ \ |  / __| '__/ _ \/ _` | __| |/ _ \| '_ \ 
    #   | |  | | (_) | (_| |  __/ | | (__| | |  __/ (_| | |_| | (_) | | | |
    #   |_|  |_|\___/ \__,_|\___|_|  \___|_|  \___|\__,_|\__|_|\___/|_| |_|
    #                                                                      
                                                                        

    #Create the model
    model = create_model()
    print(model.summary())


    #    __  __           _      _   _             _       _             
    #   |  \/  |         | |    | | | |           (_)     (_)            
    #   | \  / | ___   __| | ___| | | |_ _ __ __ _ _ _ __  _ _ __   __ _ 
    #   | |\/| |/ _ \ / _` |/ _ \ | | __| '__/ _` | | '_ \| | '_ \ / _` |
    #   | |  | | (_) | (_| |  __/ | | |_| | | (_| | | | | | | | | | (_| |
    #   |_|  |_|\___/ \__,_|\___|_|  \__|_|  \__,_|_|_| |_|_|_| |_|\__, |
    #                                                               __/ |
    #                                                              |___/ 

    #Train the model using the training datasets
    model.fit(X_train,y_train,epochs=20, class_weight=class_weights)

    #Validate the model using the testing datasets
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(f"Loss: {val_loss} Accuracy: {val_acc}")


    #     _____             _               _   _                                _      _ 
    #    / ____|           (_)             | | | |                              | |    | |
    #   | (___   __ ___   ___ _ __   __ _  | |_| |__   ___   _ __ ___   ___   __| | ___| |
    #    \___ \ / _` \ \ / / | '_ \ / _` | | __| '_ \ / _ \ | '_ ` _ \ / _ \ / _` |/ _ \ |
    #    ____) | (_| |\ V /| | | | | (_| | | |_| | | |  __/ | | | | | | (_) | (_| |  __/ |
    #   |_____/ \__,_| \_/ |_|_| |_|\__, |  \__|_| |_|\___| |_| |_| |_|\___/ \__,_|\___|_|
    #                                __/ |                                                
    #                               |___/                                                 



    model.save(f"Models/models/model{number}")





    #     ____  _   _                     _          __  __ 
    #    / __ \| | | |                   | |        / _|/ _|
    #   | |  | | |_| |__   ___ _ __   ___| |_ _   _| |_| |_ 
    #   | |  | | __| '_ \ / _ \ '__| / __| __| | | |  _|  _|
    #   | |__| | |_| | | |  __/ |    \__ \ |_| |_| | | | |  
    #    \____/ \__|_| |_|\___|_|    |___/\__|\__,_|_| |_|  
    #                                                       
    #                                                       

    #Just a bit of testing code, should be removed eventually!
    predictions1 = model.predict([X_test])
    predictions = np.argmax(predictions1, axis=1)
    y_actual = encoder.inverse_transform(y_test)
    predictions = encoder.inverse_transform(predictions)
    i = 0
    n_intrusions = 0
    n_intrusions_detected = 0
    n_fp = 0
    n_tn = 0

    nr_of_detected_flood = 0
    nr_of_detected_spoof = 0
    nr_of_detected_replay = 0
    nr_of_detected_fuzzing = 0

    nr_of_floodingattacks = 0
    nr_of_spoofingattacks = 0
    nr_of_replayattacks = 0
    nr_of_fuzzingattacks = 0

    nr_predictions = 0

    for prediction in predictions:
        #print("Row: {} --> Prediction: {} Actual: {}".format(i, prediction, y_actual[i]))
        if(y_actual[i] != "Normal"):
            n_intrusions+=1
            if(prediction == y_actual[i]):
                n_intrusions_detected+=1
            else:
                n_fp+=1

            if(y_actual[i] == 'Flooding'):
                nr_of_floodingattacks += 1
                if(prediction == y_actual[i]):
                    nr_of_detected_flood +=1
            
            if(y_actual[i] == 'Spoofing'):
                nr_of_spoofingattacks += 1
                if(prediction == y_actual[i]):
                    nr_of_detected_spoof +=1
            
            if(y_actual[i] == 'Replay'):
                nr_of_replayattacks += 1
                if(prediction == y_actual[i]):
                    nr_of_detected_replay +=1
            
            if(y_actual[i] == 'Fuzzing'):
                nr_of_fuzzingattacks += 1
                if(prediction == y_actual[i]):
                    nr_of_detected_fuzzing +=1

        else:
            if(prediction != "Normal"):
                n_fp +=1
            

        if(prediction == "Normal" and prediction == y_actual[i]):
            n_tn +=1
        
        if(prediction != "Normal"):
            nr_predictions+=1
        
        
        i+=1

    output+=f"Model {number}: \n"
    output+=f"Validation accuracy: {val_acc} Validation loss: {val_loss} \n"
    output+=f"Number of correctly detected intrusions: {n_intrusions_detected} Total number of intrusions: {n_intrusions} Quota: {n_intrusions_detected/n_intrusions} \n"
    output += f"Number of predicted intrusions: {nr_predictions}  Number of false positives: {n_fp} Number of true negatives: {n_tn} \n"
    false_positive_rate = n_fp / (n_fp + n_tn)
    output += f"False positive rate: {false_positive_rate} \n"
    output += f"Number of correctly detected flooding attacks: {nr_of_detected_flood} Total number of flooding attacks: {nr_of_floodingattacks} Quota: {nr_of_detected_flood/nr_of_floodingattacks} \n"
    output += f"Number of correctly detected spoofing attacks: {nr_of_detected_spoof} Total number of spoofing attacks: {nr_of_spoofingattacks} Quota: {nr_of_detected_spoof/nr_of_spoofingattacks} \n"
    output += f"Number of correctly detected replay attacks: {nr_of_detected_replay} Total number of replay attacks: {nr_of_replayattacks} Quota: {nr_of_detected_replay/nr_of_replayattacks} \n"
    output += f"Number of correctly detected fuzzing attacks: {nr_of_detected_fuzzing} Total number of fuzzing attacks: {nr_of_fuzzingattacks} Quota: {nr_of_detected_fuzzing/nr_of_fuzzingattacks} \n \n"

n=1
for c in output_print:
    print(f"Model {n}")
    print(c)
    n+=1
    print("\n")

print("Saving results!")
with open('Models/trainingresults.txt', 'w') as file:
    file.write(output)

