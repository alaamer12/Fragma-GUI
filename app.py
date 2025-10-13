import gradio as gr
from ml_ui import ml_interface
from dl_ui import dl_interface

with gr.Blocks() as main_app:
    gr.Markdown("## 🤖 Select a Model Interface")
    with gr.Tabs():
        with gr.TabItem("Classical ML"):
            ml_interface.render()
        with gr.TabItem("Deep Learning"):
            dl_interface.render()

# Fixed launch for Google Colab
if __name__ == "__main__":
    main_app.launch(share=True, debug=True)
