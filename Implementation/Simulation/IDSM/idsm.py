import can
import isotp
import threading
import logging
import tensorflow as tf
import time
import numpy as np
import json 


simulation_runs = {
    1: [1, '../models/model3.tflite'],
    2: [2, '../models/model3.tflite'],
    3: [3, '../models/model3.tflite'],
    4: [1, '../models/model2.tflite'],
    5: [2, '../models/model2.tflite'],
    6: [3, '../models/model2.tflite'], 
    7: [1, '../models/model4.tflite'],
    8: [2, '../models/model4.tflite'],
    9: [3, '../models/model4.tflite'],
    10: [1, '../models/model5.tflite'],
    11: [2, '../models/model5.tflite'],
    12: [3, '../models/model5.tflite'],
    13: [1, '../models/model2.tflite'],
    14: [2, '../models/model2.tflite'],
    15: [3, '../models/model2.tflite'],
    16: [1, '../models/model6.tflite'],
    17: [2, '../models/model6.tflite'],
    18: [3, '../models/model6.tflite'],
    19: [1, '../models/model1.tflite'],
    20: [2, '../models/model1.tflite'],
    21: [3, '../models/model1.tflite'],
    22: [1, '../models/model1.tflite'],
    23: [2, '../models/model1.tflite'],
    24: [3, '../models/model1.tflite'],
    25: [1, '../models/model1.tflite'],
    26: [2, '../models/model1.tflite'],
    27: [3, '../models/model1.tflite'],
    28: [1, '../models/model6.tflite'],
    29: [2, '../models/model6.tflite'],
    30: [3, '../models/model6.tflite'],

    31: [1, '../models/model5.tflite'],
    32: [2, '../models/model5.tflite'],
    33: [3, '../models/model5.tflite'],
    34: [1, '../models/model2.tflite'],
    35: [2, '../models/model2.tflite'],
    36: [3, '../models/model2.tflite'], 
    37: [1, '../models/model1.tflite'],
    38: [2, '../models/model1.tflite'],
    39: [3, '../models/model1.tflite'],
    40: [1, '../models/model2.tflite'],
    41: [2, '../models/model2.tflite'],
    42: [3, '../models/model2.tflite'],
    43: [1, '../models/model4.tflite'],
    44: [2, '../models/model4.tflite'],
    45: [3, '../models/model4.tflite'],
    46: [1, '../models/model1.tflite'],
    47: [2, '../models/model1.tflite'],
    48: [3, '../models/model1.tflite'],
    49: [1, '../models/model5.tflite'],
    50: [2, '../models/model5.tflite'],
    51: [3, '../models/model5.tflite'],
    52: [1, '../models/model5.tflite'],
    53: [2, '../models/model5.tflite'],
    54: [3, '../models/model5.tflite'],
    55: [1, '../models/model6.tflite'],
    56: [2, '../models/model6.tflite'],
    57: [3, '../models/model6.tflite'],
    58: [1, '../models/model4.tflite'],
    59: [2, '../models/model4.tflite'],
    60: [3, '../models/model4.tflite'],

    61: [1, '../models/model2.tflite'],
    62: [2, '../models/model2.tflite'],
    63: [3, '../models/model2.tflite'],
    64: [1, '../models/model3.tflite'],
    65: [2, '../models/model3.tflite'],
    66: [3, '../models/model3.tflite'], 
    67: [1, '../models/model3.tflite'],
    68: [2, '../models/model3.tflite'],
    69: [3, '../models/model3.tflite'],
    70: [1, '../models/model3.tflite'],
    71: [2, '../models/model3.tflite'],
    72: [3, '../models/model3.tflite'],
    73: [1, '../models/model5.tflite'],
    74: [2, '../models/model5.tflite'],
    75: [3, '../models/model5.tflite'],
    76: [1, '../models/model4.tflite'],
    77: [2, '../models/model4.tflite'],
    78: [3, '../models/model4.tflite'],
    79: [1, '../models/model6.tflite'],
    80: [2, '../models/model6.tflite'],
    81: [3, '../models/model6.tflite'],
    82: [1, '../models/model3.tflite'],
    83: [2, '../models/model3.tflite'],
    84: [3, '../models/model3.tflite'],
    85: [1, '../models/model4.tflite'],
    86: [2, '../models/model4.tflite'],
    87: [3, '../models/model4.tflite'],
    88: [1, '../models/model6.tflite'],
    89: [2, '../models/model6.tflite'],
    90: [3, '../models/model6.tflite']
}

class IDSM:

    def __init__(self, sim_id):

        #    ___       _    _    _                 
        #   / __| ___ | |_ | |_ (_) _ _   __ _  ___
        #   \__ \/ -_)|  _||  _|| || ' \ / _` |(_-<
        #   |___/\___| \__| \__||_||_||_|\__, |/__/
        #                                |___/     

        self.simulation_run = sim_id

        #Sets collaborative vs baseline mode
        #self.collabarative = simulation_runs[self.simulation_run][0]
        self.collabarative = simulation_runs[self.simulation_run][0]

        model_path = simulation_runs[self.simulation_run][1]

        #Threshold to be used
        self.threshold = 0.85 

        self.addr = isotp.Address(isotp.AddressingMode.Normal_11bits, rxid=0x123, txid=0x456)
        #self.addr = isotp.Address(isotp.AddressingMode.Normal_11bits, rxid=0x456, txid=0x123)

        self.send_ip = 'ws://192.168.1.221:54701/'
        self.receive_ip = 'ws://192.168.1.39:54701/'
        #self.receive_ip_2 = 'ws://192.168.1.223:54701/'
        self.receive_ip_2 = 'ws://192.168.1.93:54701/'

        #    ___           _ 
        #   | __| _ _   __| |
        #   | _| | ' \ / _` |
        #   |___||_||_|\__,_|
        #                    


        #Tracker of the sent consultation messages
        self.consultation_messages_sent = 0
        self.consultation_responses_received = 0

        # Load the TFLite model
        self.model = tf.lite.Interpreter(model_path=model_path)        
        self.model.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

        #Registered alerts
        self.alerts = dict()

        self.exit_requested = False

        self.init_local_stack()

        if(self.collabarative == 2):
            self.init_remote_stacks()
        elif(self.collabarative == 3):
            self.init_remote_stacks_2()

        self.start_times_consultation = dict()
        self.times_consultation = dict()
        self.times_local = dict()
        self.total_number_of_rows = 0

        print(f"IDSM: Loaded simulation run {self.simulation_run}")

    def init_local_stack(self):
        self.local_bus = can.ThreadSafeBus(bustype='socketcan', channel='vcan0')
        self.local_stack = isotp.CanStack(self.local_bus, address=self.addr, error_handler=self.my_error_handler)

    def init_remote_stacks(self):
        #self.remote_bus_send = can.Bus('ws://192.168.1.39:54701/',
        self.remote_bus_send = can.ThreadSafeBus(self.send_ip,  
            bustype='remote',
            bitrate=500000,
            receive_own_messages=False)
        self.remote_stack_send = isotp.CanStack(self.remote_bus_send, address=self.addr, error_handler=self.my_error_handler_remote_send)

        # self.remote_bus_receive = can.Bus('ws://192.168.1.221:54701/',
        self.remote_bus_receive = can.ThreadSafeBus(self.receive_ip, 
            bustype='remote',
            bitrate=500000,
            receive_own_messages=False)
        self.remote_stack_receive = isotp.CanStack(self.remote_bus_receive, address=self.addr, error_handler=self.my_error_handler_remote_receive)
        

    def init_remote_stacks_2(self):
        self.remote_bus_send = can.ThreadSafeBus(self.send_ip,  
            bustype='remote',
            bitrate=500000,
            receive_own_messages=False)
        self.remote_stack_send = isotp.CanStack(self.remote_bus_send, address=self.addr, error_handler=self.my_error_handler_remote_send)

        # self.remote_bus_receive = can.Bus('ws://192.168.1.221:54701/',
        self.remote_bus_receive = can.ThreadSafeBus(self.receive_ip, 
            bustype='remote',
            bitrate=500000,
            receive_own_messages=False)
        self.remote_stack_receive = isotp.CanStack(self.remote_bus_receive, address=self.addr, error_handler=self.my_error_handler_remote_receive)

        self.remote_bus_receive_2 = can.ThreadSafeBus(self.receive_ip_2, 
            bustype='remote',
            bitrate=500000,
            receive_own_messages=False)
        self.remote_stack_receive_2 = isotp.CanStack(self.remote_bus_receive_2, address=self.addr, error_handler=self.my_error_handler_remote_receive)


    def my_error_handler_remote_send(self, error):
        #print("REMOTE BUS SEND ERROR:")
        #logging.warning('IsoTp error happened : %s - %s' % (error.__class__.__name__, str(error)))
        pass
    def my_error_handler_remote_receive(self, error):
        print("REMOTE BUS RECEIVE ERROR:")
        logging.warning('IsoTp error happened : %s - %s' % (error.__class__.__name__, str(error)))
    def my_error_handler(self, error):
        print("LOCAL BUS ERROR:")
        logging.warning('IsoTp error happened : %s - %s' % (error.__class__.__name__, str(error)))


    #If an ids_alert is received: get the message and run the "process alert message method"
    def receive_ids_alert(self):
        print("Started the alert thread!")
        while self.exit_requested == False:
            if self.local_stack.available():
                #start_time = time.time()
                payload = self.local_stack.recv()
                payload_string = payload.decode("utf-8")

                if(payload_string.isnumeric()):
                    self.total_number_of_rows = payload_string
                    print("'Done' message received, shutting down")
                    break
                    
                else:
                    ids_alert = json.loads(payload_string)
                    self.process_alert_message(ids_alert)
            #time.sleep(self.local_stack.sleep_time()) # Variable sleep time based on state machine state
            #time.sleep(0.01) # Variable sleep time based on state mac
            self.local_stack.process()

        #Shutdown when done (when "Done" message is received)        
        self.shutdown()

    def test(self):
        print("Test thread started")
        i = 0
        while(True):
            if(i%100 == 0):
                model_input = [0,0,0,0,0,0,0,0,0.001,10,8]
                start_time = time.time()
                ids_alert = {
                    'type': 'Alert',
                    'row':22,
                    'prediction':1,
                    'certainty': 0.55,
                    'model_input': [model_input]
                }
                self.process_alert_message(ids_alert, start_time)
            else:
                model_input = [0,0,0,0,0,0,0,0,0.001,10,8]
                start_time = time.time()
                ids_alert = {
                    'type': 'Alert',
                    'row':22,
                    'prediction':1,
                    'certainty': 0.55,
                    'model_input': [model_input]
                }
                self.process_alert_message(ids_alert, start_time)
            i+=1
            time.sleep(0.0005)

    #If an alert message received from security sensor, check if alert is less than threshold. if so send consultation message to other nodes. 
    #To-do: Logg all alertmessages + logg all consultation messages we receive later (maybe in another method) + needs to be logged in a way that is sorted, so that we later can validate more easily
    def process_alert_message(self, alert_message):
        start_time = float(alert_message['start_time'])
        #print(f"Received an alert: Row {alert_message['row']} Prediction {alert_message['prediction']}")
        if(self.collabarative == 1):
            self.alerts[alert_message['row']] = [alert_message["prediction"], alert_message["certainty"]]
            self.times_local[alert_message['row']] = time.time() - start_time

        else:
            if(alert_message['certainty'] < self.threshold):
                #print(f"IDSM: Not sure enough on row {alert_message['row']}, certanty = {alert_message['certainty']} Sending consultation request")
                self.consultation_messages_sent +=1

                if(self.consultation_messages_sent % 8000 == 0):
                    print(f"IDSM: {self.consultation_messages_sent} consultation messages sent!")

                self.alerts[alert_message['row']] = [alert_message["prediction"], alert_message["certainty"]]
                consultation_msg = {
                    'type':'Request',
                    'row': alert_message['row'],
                    'input': alert_message['model_input']
                }
                
                self.start_times_consultation[alert_message['row']] = start_time
                if(not self.send_message(json.dumps(consultation_msg))):
                    print("IDSM: Sending consultation message failed, falling back to the local message")
                    self.alerts[alert_message['row']] = [alert_message["prediction"], alert_message["certainty"]]
                    self.times_local[alert_message['row']] = time.time() - start_time

                    del self.start_times_consultation[alert_message['row']]
            else:
                self.alerts[alert_message['row']] = [alert_message["prediction"], alert_message["certainty"]]
                self.times_local[alert_message['row']] = time.time() - start_time

        

    #Receives consultation message and runs method for processing consultation message
    def receive_consultation_messages(self):
        print("Started the consultation thread!")
        while(self.exit_requested == False):
            try:
                if self.remote_stack_receive.available():
                    payload = self.remote_stack_receive.recv()
                    payload_string = payload.decode("utf-8")
                    consultation_message = json.loads(payload_string)
                    if(consultation_message['type'] == 'Request'):
                        self.process_consultation_message(consultation_message)
                    else:
                        self.process_consultation_response(consultation_message)
            except:
                print("Remote bus went down. Trying to reconnect")
                self.init_remote_stacks()
    
            time.sleep(self.remote_stack_receive.sleep_time())
            self.remote_stack_receive.process()

    #Receives consultation message and runs method for processing consultation message
    def receive_consultation_messages_2(self):
        print("Started the consultation thread 2!")
        while(self.exit_requested == False):
            try:
                if self.remote_stack_receive_2.available():
                    payload = self.remote_stack_receive_2.recv()
                    payload_string = payload.decode("utf-8")
                    consultation_message = json.loads(payload_string)
                    if(consultation_message['type'] == 'Request'):
                        self.process_consultation_message(consultation_message)
                    else:
                        self.process_consultation_response(consultation_message)
            except:
                print("Remote bus went down. Trying to reconnect")
                self.init_remote_stacks_2()
    
            time.sleep(self.remote_stack_receive_2.sleep_time())
            self.remote_stack_receive_2.process()

    #take the model input in consultation message as input and run predictions on the input. Take generated prediction and use send_message function to send a consultation response back to the requestee
    #To-do: fix the logging and print shit
    def process_consultation_message(self, consultation_message):
        #print(f"IDSM: Received a request for a consultation, for row {consultation_message['row']}")
        model_input = consultation_message['input']
        model_input = np.array(model_input, dtype=np.float32)

        self.model.set_tensor(self.input_details[0]['index'], model_input)
        self.model.invoke()
        predictions = self.model.get_tensor(self.output_details[0]['index'])[0]

        #The index (i.e class) with the highest probability
        prediction = np.argmax(predictions)
        prediction_certainty = predictions[prediction]
        returnmsg = {
            'type': 'Response', 
            'row': consultation_message['row'],
            'prediction': int(prediction),
            'certainty': float(prediction_certainty)
        }
        self.send_message(json.dumps(returnmsg))


    #Not done! Need to use some more advanced logic here! 
    def process_consultation_response(self, consultation_message):
        row = consultation_message['row']
        #print(f"IDSM: Received a response for the consultation message for row {row}")
        self.consultation_responses_received +=1

        if(self.consultation_responses_received % 8000 == 0):
            print(f"IDSM: {self.consultation_responses_received} consultation responses received!")

        #Remove the cons. queue entry when the response arrives (remember to add timeouts to this!)
        if(consultation_message['certainty'] > self.alerts[row][1]):
            self.alerts[row] = [consultation_message['prediction'], consultation_message['certainty']]


        self.times_consultation[row] = time.time() - self.start_times_consultation[row]


    #Send a message using the isotp remote canstack
    def send_message(self, msgstring):
        b = bytearray()
        b.extend(map(ord, msgstring))
        try:
            while(True):
                if(not self.remote_stack_send.transmitting()):
                    self.remote_stack_send.send(b)
                    while self.remote_stack_send.transmitting():
                        self.remote_stack_send.process()
                        #time.sleep(0.001)
                        time.sleep(self.remote_stack_send.sleep_time())
                    break
                else:
                    time.sleep(0.001)
                    #time.sleep(self.remote_stack_send.sleep_time())
                    self.remote_stack_send.process()
            return True
        except:
            print("Something went wrong when trying to send the message \n Trying to reconnect to remote bus")
            self.init_remote_stacks()
            return False

    def start(self):
        self.exit_requested = False
        self.start_time = time.time()

        if(not self.collabarative == 1):
            thread_consultation = threading.Thread(target=self.receive_consultation_messages)
            thread_consultation.daemon = True
            thread_consultation.start()
        if(self.collabarative == 3):
            thread_consultation_2 = threading.Thread(target=self.receive_consultation_messages_2)
            thread_consultation_2.daemon = True
            thread_consultation_2.start()

        thread_alert = threading.Thread(target=self.receive_ids_alert)
        thread_alert.daemon = True
        thread_alert.start()

        #Wait for the other threads to finish
        if(self.collabarative == 2):
            thread_consultation.join()
        if(self.collabarative == 3):
            thread_consultation.join()
            thread_consultation_2.join()

        thread_alert.join()

    def stop(self):
        self.exit_requested = True

    def save(self):
        print("Shutdown requested...saving")
        
        #Then start the save itself
        with open(f"../Results2/{self.simulation_run}_result.csv", "w") as file:
            for key, value in self.alerts.items():
                line = f"{key},{value[0]} \n"
                file.write(line)
        print("Save completed, exiting!")

    def save_metadata(self):
        print("Saving metadata")
        
        #Then start the save itself
        with open(f"../Results2/{self.simulation_run}_metadata.txt", "w") as file:

            if(len(self.times_local) > 0):
                average_local_time = sum(self.times_local.values()) / len(self.times_local)
                file.write("Average local time(seconds):{0:.17f}\n".format(average_local_time))
            else:
                file.write("Average local time:N/A \n")

            if(len(self.times_consultation) > 0):
                average_consultation_time = sum(self.times_consultation.values()) / len(self.times_consultation)
                file.write(f"Average consultation time(seconds):{average_consultation_time}\n")
            else:
                file.write(f"Average consultation time:N/A \n")

            file.write(f"Total number of lines:{self.total_number_of_rows} \n")
            file.write(f"Total number of lines sent to the IDSM:{len(self.alerts)} \n")
            file.write(f"Total number of consultation messages:{self.consultation_messages_sent} \n")
            simulation_time = (time.time() - self.start_time) / 60
            file.write(f"Total simulation time(minutes):{simulation_time} \n")

        print("Save completed!")


    def shutdown(self):
        print("Shutting down")
        print(f"Consultation messages sent: {self.consultation_messages_sent} Consultation responses received: {self.consultation_responses_received}")
        self.save()
        self.save_metadata()
        self.stop()
        if(self.collabarative == 2):
            self.send_message(str(self.total_number_of_rows))
            self.remote_bus_send.shutdown()
            self.remote_bus_receive.shutdown()
        if(self.collabarative == 3):
            self.send_message(str(self.total_number_of_rows))
            self.remote_bus_send.shutdown()
            self.remote_bus_receive.shutdown()   
            self.remote_bus_receive_2.shutdown() 

        self.local_bus.shutdown()
        print("Bye!")


if __name__ == '__main__':
    """
    for key in simulation_runs:
            #Only do baseline for now
            if simulation_runs[key][0]==3:
                idsm = IDSM(key)
                try:
                    idsm.start()
                except KeyboardInterrupt:
                    idsm.shutdown()
                    break
    """
    
    idsm = IDSM(63)
    try:
        idsm.start()
    except KeyboardInterrupt:
        idsm.shutdown()
        #break