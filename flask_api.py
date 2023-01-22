from flask import Flask, request, jsonify
from inference import boundbox_predict, setup_model, extract_bound_boxes

app = Flask(__name__)
print("setup start")
cfg_filepath = "cascade_dit_base.yml"
weights_filepath = "./publaynet_dit-b_cascade.pth"
app.cfg, app.model = setup_model(weights_filepath, cfg_filepath)
print("setup done")

@app.route("/ping/", methods=["GET"])
def pong():
    response = jsonify(message="pong")
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/boundbox/", methods=["POST"])
def predict_bound_boxes():
    img_b64_str = request.form['img_b64_str'].replace("\n","") 
    print(img_b64_str[:30], "*****" ,img_b64_str[-30:])
    img_size, bound_boxes = boundbox_predict(img_b64_str,
                                             app.model,
                                             app.cfg)
    response = jsonify(bboxes=bound_boxes)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, 201

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="80")
