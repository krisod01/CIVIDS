import can
import tensorflow as tf
import numpy as np
import sys
import time
import json
import isotp
import logging
import traceback
import pandas as pd
import time
from queue import Queue
import threading
from pathlib import Path


#    ___       _    _    _                 
#   / __| ___ | |_ | |_ (_) _ _   __ _  ___
#   \__ \/ -_)|  _||  _|| || ' \ / _` |(_-<
#   |___/\___| \__| \__||_||_||_|\__, |/__/
#                                |___/     

simulation_runs = {
    1: ['../data/1.csv', '../models/model3.tflite'],
    2: ['../data/1.csv', '../models/model3.tflite'],
    3: ['../data/1.csv', '../models/model3.tflite'],
    4: ['../data/5.csv', '../models/model2.tflite'],
    5: ['../data/5.csv', '../models/model2.tflite'],
    6: ['../data/5.csv', '../models/model2.tflite'],
    7: ['../data/1.csv', '../models/model4.tflite'],
    8: ['../data/1.csv', '../models/model4.tflite'],
    9: ['../data/1.csv', '../models/model4.tflite'],
    10: ['../data/6.csv', '../models/model5.tflite'],
    11: ['../data/6.csv', '../models/model5.tflite'],
    12: ['../data/6.csv', '../models/model5.tflite'],
    13: ['../data/3.csv', '../models/model2.tflite'],
    14: ['../data/3.csv', '../models/model2.tflite'],
    15: ['../data/3.csv', '../models/model2.tflite'],
    16: ['../data/1.csv', '../models/model6.tflite'],
    17: ['../data/1.csv', '../models/model6.tflite'],
    18: ['../data/1.csv', '../models/model6.tflite'],
    19: ['../data/5.csv', '../models/model1.tflite'],
    20: ['../data/5.csv', '../models/model1.tflite'],
    21: ['../data/5.csv', '../models/model1.tflite'],
    22: ['../data/3.csv', '../models/model1.tflite'],
    23: ['../data/3.csv', '../models/model1.tflite'],
    24: ['../data/3.csv', '../models/model1.tflite'],
    25: ['../data/2.csv', '../models/model1.tflite'],
    26: ['../data/2.csv', '../models/model1.tflite'],
    27: ['../data/2.csv', '../models/model1.tflite'],
    28: ['../data/2.csv', '../models/model6.tflite'],
    29: ['../data/2.csv', '../models/model6.tflite'],
    30: ['../data/2.csv', '../models/model6.tflite'],


    31: ['../data/1.csv', '../models/model5.tflite'],
    32: ['../data/1.csv', '../models/model5.tflite'],
    33: ['../data/1.csv', '../models/model5.tflite'],
    34: ['../data/6.csv', '../models/model2.tflite'],
    35: ['../data/6.csv', '../models/model2.tflite'],
    36: ['../data/6.csv', '../models/model2.tflite'],
    37: ['../data/4.csv', '../models/model1.tflite'],
    38: ['../data/4.csv', '../models/model1.tflite'],
    39: ['../data/4.csv', '../models/model1.tflite'],
    40: ['../data/4.csv', '../models/model2.tflite'],
    41: ['../data/4.csv', '../models/model2.tflite'],
    42: ['../data/4.csv', '../models/model2.tflite'],
    43: ['../data/6.csv', '../models/model4.tflite'],
    44: ['../data/6.csv', '../models/model4.tflite'],
    45: ['../data/6.csv', '../models/model4.tflite'],
    46: ['../data/6.csv', '../models/model1.tflite'],
    47: ['../data/6.csv', '../models/model1.tflite'],
    48: ['../data/6.csv', '../models/model1.tflite'],
    49: ['../data/3.csv', '../models/model5.tflite'],
    50: ['../data/3.csv', '../models/model5.tflite'],
    51: ['../data/3.csv', '../models/model5.tflite'],
    52: ['../data/4.csv', '../models/model5.tflite'],
    53: ['../data/4.csv', '../models/model5.tflite'],
    54: ['../data/4.csv', '../models/model5.tflite'],
    55: ['../data/3.csv', '../models/model6.tflite'],
    56: ['../data/3.csv', '../models/model6.tflite'],
    57: ['../data/3.csv', '../models/model6.tflite'],
    58: ['../data/2.csv', '../models/model4.tflite'],
    59: ['../data/2.csv', '../models/model4.tflite'],
    60: ['../data/2.csv', '../models/model4.tflite'],

    61: ['../data/1.csv', '../models/model2.tflite'],
    62: ['../data/1.csv', '../models/model2.tflite'],
    63: ['../data/1.csv', '../models/model2.tflite'],
    64: ['../data/4.csv', '../models/model3.tflite'],
    65: ['../data/4.csv', '../models/model3.tflite'],
    66: ['../data/4.csv', '../models/model3.tflite'],
    67: ['../data/6.csv', '../models/model3.tflite'],
    68: ['../data/6.csv', '../models/model3.tflite'],
    69: ['../data/6.csv', '../models/model3.tflite'],
    70: ['../data/2.csv', '../models/model3.tflite'],
    71: ['../data/2.csv', '../models/model3.tflite'],
    72: ['../data/2.csv', '../models/model3.tflite'],
    73: ['../data/2.csv', '../models/model5.tflite'],
    74: ['../data/2.csv', '../models/model5.tflite'],
    75: ['../data/2.csv', '../models/model5.tflite'],
    76: ['../data/5.csv', '../models/model4.tflite'],
    77: ['../data/5.csv', '../models/model4.tflite'],
    78: ['../data/5.csv', '../models/model4.tflite'],
    79: ['../data/4.csv', '../models/model6.tflite'],
    80: ['../data/4.csv', '../models/model6.tflite'],
    81: ['../data/4.csv', '../models/model6.tflite'],
    82: ['../data/5.csv', '../models/model3.tflite'],
    83: ['../data/5.csv', '../models/model3.tflite'],
    84: ['../data/5.csv', '../models/model3.tflite'],
    85: ['../data/3.csv', '../models/model4.tflite'],
    86: ['../data/3.csv', '../models/model4.tflite'],
    87: ['../data/3.csv', '../models/model4.tflite'],
    88: ['../data/5.csv', '../models/model6.tflite'],
    89: ['../data/5.csv', '../models/model6.tflite'],
    90: ['../data/5.csv', '../models/model6.tflite']
}

#CHANGE THIS::
simulation_run = 1




queue = Queue()
total_number_of_rows = 0

class CanFeeder():
    #1. Load data

    def __init__(self):
        #df = pd.read_csv('../Car_Hacking_Challenge_Dataset/0_Preliminary/0_Training/Pre_train_D_1.csv', sep =',')
        simulation_data_path = simulation_runs[simulation_run][0]
        df = pd.read_csv(simulation_data_path, sep =',')
        df['relative_time'] = df['Timestamp'] - df['Timestamp'].iloc[0]
        df['delta_time'] = df['relative_time'].diff().fillna(df['relative_time'])

        print(f"CanFeeder: Starting simulation run: {simulation_run}")
        print(f"CanFeeder: Pandas dataframe length: {len(df)}")

        #2. Iterate through and send data
        print('CanFeeder: Starting to send messages!')
        counter = 0
        for index, row in df.iterrows():
            counter+=1

            try:
                #Sleep until it is time to send the next message
                time.sleep(row['delta_time'])
                
                data = []
                for number in row['Data'].split():
                    data.append(int(number, 16))
                
                message ={
                    "Timestamp":row['Timestamp'],
                    "arbitration_id" : int(row['Arbitration_ID'], 16),
                    "dlc" : int(row['DLC']),
                    "data" : data,
                    "is_extended_id":True,
                    "row":index}
                queue.put(message)
                
                
            except Exception as e:
                print("CanFeeder: Sending message on row {} failed!".format(index))
                traceback.print_exc()
                print(e)
            
        print("CanFeeder: Finished sending all of the messages!")
        queue.put("Done")
        global total_number_of_rows
        total_number_of_rows = counter
        print(f"CanFeeder: Sent {counter} messages in total") 




class SecuritySensor():

    def __init__(self, model_path): 
        self.bus = can.ThreadSafeBus(bustype='socketcan', channel='vcan0')
        addr = isotp.Address(isotp.AddressingMode.Normal_11bits, rxid=0x456, txid=0x123)
        self.local_stack = isotp.CanStack(self.bus, address=addr, error_handler=self.my_error_handler)


        #Our ML model
        # Load the TFLite model and allocate tensors.
        self.model = tf.lite.Interpreter(model_path=model_path)
        self.model.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()


        #The encoder will encode the different subclasses as following:
        self.classes = {
           0: "Flooding",
           1: "Fuzzing",
           2: "Normal",
           3: "Replay",
           4: "Spoofing"
        }

        #Tracks the current row, for validation purposes
        self.row = 0
        #The model expects to receive the data in the following order: 
        #datafields 1-8, timedelta, arbitration_id, dlc 

        #Check what an appropriate initial timestamp would be
        self.latest_timestamp = 0
        print(f"SECURITY SENSOR: Loaded simulation run {simulation_run}") 
        print("SECURITY SENSOR: Ready to receive messages:...")
        self.receive_messages()


    def receive_messages(self):
        #try:
            while(True):
                message = queue.get()
                if(message == "Done"):
                    self.save_and_quit()
                    break
                else:
                    start_time = time.time()
                    row, model_input = self.process_message(message)
                    self.row = row
                    self.detect_anomaly(model_input, start_time)

    def process_message(self, message):
        #Compare and update the timestamps
        delta_time = message['Timestamp'] - self.latest_timestamp 
        self.latest_timestamp = message['Timestamp']

        #Handle the data payload and DLC fields
        message_data = list(message['data'])
        message_dlc = message['dlc']
        message_data = self.pad_data(message_data, message_dlc)

        #Retrieve the arbitration id
        message_id = message['arbitration_id']
        
        #Create the model input array
        model_input = message_data
        model_input.append(delta_time)
        model_input.append(message_id)
        model_input.append(message_dlc)
        model_input = np.array([model_input], dtype=np.float32)

        message_row = message['row']
        return message_row, model_input


    def detect_anomaly(self, model_input, start_time):
        #For checking the prediction time
        #start_time = time.time()

        #Array with 5 probabilities between 0 and 1, one for each class
        self.model.set_tensor(self.input_details[0]['index'], model_input)
        self.model.invoke()
        predictions = self.model.get_tensor(self.output_details[0]['index'])[0]

        #The index (i.e class) with the highest probability
        prediction = np.argmax(predictions)

        prediction_certainty = predictions[prediction]

        # 2 = Normal in our dataaset
        if(prediction!=2):
            self.report_anomaly(model_input,prediction, prediction_certainty, start_time)              
            #self.output.append(f"Row {self.row}: Predicted: {self.classes[prediction]} with {100*prediction_certainty}% certainty\n")

        if self.row % 400000 == 0:
            print(f'SECURITY SENSOR: At row {self.row}')

        #print("--- %s seconds ---" % (time.time() - start_time))

    def report_anomaly(self,model_input, prediction, prediction_certainty, start_time):
            #print(f"Detected an anomaly on row {self.row}! Forwarding to IDSM")
            ids_alert = {
                'type': 'Alert',
                'row':self.row,
                'prediction':int(prediction),
                'certainty':float(prediction_certainty),
                'model_input': model_input.tolist(),
                'start_time':start_time
            }
            json_string = json.dumps(ids_alert)
            b = bytearray()
            b.extend(map(ord, json_string))

            #start = time.time()
            self.local_stack.send(b)

            while self.local_stack.transmitting():
                self.local_stack.process()
                #time.sleep(self.local_stack.sleep_time())
                #time.sleep(0.01)
            #print(f"SECURITY SENSOR: Send message time: {time.time()-start}")


    def pad_data(self, message_data, message_dlc):
        #Zero padding
        for i in range(len(message_data), 8):
            message_data.append(0)
        return message_data
    
    def my_error_handler(self, error):
        logging.warning('IsoTp error happened : %s - %s' % (error.__class__.__name__, str(error)))

    
    def save_and_quit(self):
        print("Shutdown requested / Done with messages")
        self.send_exit_message()
        #sys.exit(0)
    
    def send_exit_message(self):
        #Give the IDSM time to wrap up
        time.sleep(15)
        b = bytearray()
        b.extend(map(ord, str(total_number_of_rows)))

        self.local_stack.send(b)

        while self.local_stack.transmitting():
            self.local_stack.process()
            time.sleep(self.local_stack.sleep_time())

def start_sensor():
    model_path = simulation_runs[simulation_run][1]
    sensor = SecuritySensor(model_path)

def start_feeder():
    feeder = CanFeeder()

if __name__ == '__main__':

    #for i in range(1,4):
    #Only baseline:
    #for i in range (1, 91, 3):
    for i in range(63,64,1):
        simulation_run = i

        thread_sensor = threading.Thread(target=start_sensor)
        thread_sensor.daemon = True
        thread_sensor.start()
        time.sleep(10)

        #ready = input("Start the CAN-Feeder? y/n  ")
        #if(ready == "y"):
        start_feeder()
        try:
            thread_sensor.join()
        except KeyboardInterrupt:
            print("Shutdown requested...")
            break
    
        while(True):
            time.sleep(60)
            if Path(f"../Results2/{i}_result.csv").is_file():
                print("Result file found, moving on to the next simulation")
                break
            else:
                print ("Result file does not exist yet, waiting for 10 minutes")
                time.sleep(60)
        #ADD code to wait until result file has been created
