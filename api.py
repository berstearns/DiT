from fastapi import FastAPI, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from inference import boundbox_predict, setup_model, extract_bound_boxes

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    cfg_filepath = "cascade_dit_base.yml"
    weights_filepath = "./publaynet_dit-b_cascade.pth"
    app.cfg, app.model = setup_model(weights_filepath, cfg_filepath)

@app.get("/ping")
def pong():
    return "pong"

@app.post("/boundbox")
async def predict_bound_boxes(request : Request):
    img_b64_str = await request.body()
    # img_b64_str = bytes(str(img_b64_str).replace("\n",""),"utf-8")
    img_b64_str = img_b64_str.replace(b'%0A',b'')
    img_b64_str = img_b64_str.replace(b'img_b64_str=',b'')
    print(img_b64_str[:30], "*****" ,img_b64_str[-30:])
    img_size, bound_boxes = boundbox_predict(img_b64_str,
                                             app.model,
                                             app.cfg)
    return {"bboxes": bound_boxes}
