from flask import Flask, render_template
from Diffusion.pipeline.prediction_pipline import PredictionPipeline
from Diffusion.utils.utils import save_images ,plot_images
from Diffusion.components.models import *



#model = model.to("cuda")
pipeline = PredictionPipeline(img_size=64)
#result = pipeline.result(model, 5)
# fig_path = '/home/amzad/Desktop/diffusion/fig/result_3.jpg'
# save_images(result,fig_path )


model_ckpt_paths = {
    'anime_face': "/home/amzad/Desktop/diffusion/artifacts/model_ckpt/anime_face_ckpt.pt",
    'flower': "/home/amzad/Desktop/diffusion/artifacts/model_ckpt/flower_ckpt.pt",
    'celeba': "/home/amzad/Desktop/diffusion/artifacts/model_ckpt/celeba_ckpt.pt",
    'pokemon': "/home/amzad/Desktop/diffusion/artifacts/model_ckpt/pokemon_ckpt.pt",
    
    }
model = UNet().to("cuda")
model_ckpt = model_ckpt_paths['anime_face']
model.load_state_dict(torch.load(model_ckpt))

app = Flask(__name__)

@app.route('/') 
def hello_world(): 
    return render_template('index.html')

@app.route('/button-clicked', methods=['POST'])
def button_clicked():
    # Your function code here
    return 'Button clicked!'

@
@app.route('/predict', methods=['POST'])
def predict():
    result = pipeline.result(model, 5)
    fig_path = '/home/amzad/Desktop/diffusion/fig/result_3.jpg'
    save_images(result,fig_path )
    return render_template('index.html', result=fig_path)

if __name__ == '__main__': 
    app.run(debug=True)