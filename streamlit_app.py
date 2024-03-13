import streamlit as st
import streamlit_drawable_canvas 
import numpy as np 
import cv2
import keras
import time
from io import BytesIO
import gtts


file=BytesIO()

st.set_page_config("Digit Recognition Page")
st.title("Digit Recognition ")


hide="""
<style>
footer {visibility : hidden;}
</style>
"""
st.markdown(hide,unsafe_allow_html=True)

col1,col2=st.columns([5,3])


col1.subheader("Please draw digit between 0 and 9 below.")

st.sidebar.title("""RAJESH""")

with col1.container():

    cnvs=streamlit_drawable_canvas.st_canvas(
    fill_color="rgba(255,0,0,0)",
    stroke_color='black',stroke_width=15,drawing_mode='freedraw',
    key='canvas',
    width=250,
    height=250,
    background_color='white'
    )




if st.button("Recognize "):
    
    model=keras.models.load_model("My_model.keras")
    if cnvs.image_data is not None:
        img=cv2.cvtColor(cnvs.image_data,cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(28,28))
        img=np.array(img)
        img=img/255.0
        img=np.where(img<1,1,0)
        img=img.reshape(1,28,28,1)
        pred=model.predict(img)
        

    acc=np.max(pred)
    acc=str(int((acc)*100))+'% '
    d=str(np.argmax(pred))
    
    with col2.container():
        
        prog=col2.progress(0)
        st.subheader("I am Recognizing please wait !")
        for i in range(1,101):
            prog.progress(i,text=(f"{i}%"))
            time.sleep(0.01)
            
        
        col2.success(f"Predicted digit  =  {d}  ")
        col2.success(f"Accuracy achieved = {acc}")
        try:
            speech=gtts.gTTS(f"Digit {d} is recognised and achieved {acc} accuracy !")
            speech.write_to_fp(file)
            col2.write("Click below to play audio")
            col2.audio(file)
            
        except:
            pass
    
