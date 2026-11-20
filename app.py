"""Local-only demo: python app.py. No external customer data is sent."""
from functools import lru_cache
from pathlib import Path
import sys, os
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED','False')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from kaggle_yc.inference import Router

@lru_cache(maxsize=1)
def router(): return Router(ROOT,'tfidf','cpu')

def route(text): return router().route(text)

def build_app():
    import gradio as gr
    with gr.Blocks(title='Support request router') as app:
        gr.Markdown('# Support request router\nA research demo trained on a synthetic support dataset. Low-confidence requests are flagged for human review. Scores are uncalibrated; a high score does not establish that a request is in scope.')
        request=gr.Textbox(label='Support request',lines=4,placeholder='I need to change the delivery address on my order.')
        button=gr.Button('Route request',variant='primary')
        output=gr.JSON(label='Routing decision')
        button.click(route,request,output)
        gr.Examples(['Where is my order?','I want to cancel my order and get a refund.','Please schedule an installation appointment.'],inputs=request)
        gr.Markdown('TF-IDF word and character features with logistic regression. Threshold chosen on development data. This demo makes a routing suggestion only; it cannot change an account or execute an action.')
    return app

if __name__=='__main__': build_app().launch(server_name='127.0.0.1',server_port=7860,share=False)
