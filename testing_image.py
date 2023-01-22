# Open the image form working directory
base64_file = "./img_b64"
output_filename = "./img_output.jpg"
import base64
with open(base64_file) as inpf:
    imgdata = base64.b64decode(inpf.read())

with open(output_filename, 'wb') as f:
    f.write(imgdata)    
