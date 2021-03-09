import time
import json
from Run_image import *
from flask import Flask, request, jsonify
from matrix import projection_mat
app = Flask(__name__)

yolo, model, averages, angle_bins = initialize_model()

@app.route('/', methods=['GET', 'POST'])
def add_message():
    content = request.form
    file = request.files['file']
    time_stamp = str(time.time()).replace('.','')

    save_path = "../arcore_cpu_images/"+time_stamp

    file.save(save_path+".jpg")
    with open(save_path+".txt","w+") as f:
    	f.write(content['projection']+"\n"+content['viewmat']+"\n"+content['posem'])
    
    detections = object_detection_3d(save_path+".jpg", yolo, model, averages, angle_bins)

    results = {}
    results["objects"] = len(detections)

    for objs in range(results["objects"]):
        results[str(objs)] = {}
        results[str(objs)]["width"] = detections[objs][0]
        results[str(objs)]["height"] = detections[objs][1]
        results[str(objs)]["length"] = detections[objs][2]
        results[str(objs)]["dist"] = detections[objs][3]
        results[str(objs)]["class"] = detections[objs][4]
    
    print(json.dumps(results))
    return(json.dumps(results))


    # print("./"+time_stamp+".txt")
    # projection_mat(save_path+".txt")
    # print (content['projection'])
    # print (content['viewmat'])
    


if __name__ == '__main__':
    app.run(host= '0.0.0.0',debug=True,ssl_context="ADHOC")

