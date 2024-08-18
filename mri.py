import streamlit as st
import pickle
import re
import cv2
import numpy as np
import keras
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
from PIL import Image
import time

modelh5 = tensorflow.keras.models.load_model('model.h5')

def show_page():
    st.write("<h3 style='text-align: center; color: blue;'>سامانه تشخیص تحلیل رفتگی بافت مغز سالمندان 🩺</h3>", unsafe_allow_html=True)
    st.write("<h5 style='text-align: center; color: gray;'>Robo-Ai.ir طراحی شده توسط</h5>", unsafe_allow_html=True)
    st.link_button("Robo-Ai بازگشت به", "https://robo-ai.ir")
    with st.sidebar:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(' ')
        with col2:
            st.image('img.png')
        with col3:
            st.write(' ')
        st.divider()
        st.write("<h4 style='text-align: center; color: black;'>تشخیص زوال عقل یا دمانس زودرس</h4>", unsafe_allow_html=True)
        st.write("<h4 style='text-align: center; color: gray;'>با استفاده از تصاویر اسکن مغزی</h4>", unsafe_allow_html=True)
        st.write("<h4 style='text-align: center; color: gray;'>تحلیل افکار کاربر</h4>", unsafe_allow_html=True)
        st.write("<h4 style='text-align: center; color: gray;'>و بررسی پرسشنامه</h4>", unsafe_allow_html=True)
        st.divider()
        st.write("<h5 style='text-align: center; color: black;'>طراحی و توسعه</h5>", unsafe_allow_html=True)
        st.write("<h5 style='text-align: center; color: black;'>حمیدرضا بهرامی</h5>", unsafe_allow_html=True)


    container = st.container(border=True)
    container.write("<h6 style='text-align: right; color: gray;'>تشخیص آلزایمر و تحلیل رفتن مغز با اسکن مغزی 🧠</h6>", unsafe_allow_html=True)

    image = st.file_uploader('آپلود تصویر', type=['jpg', 'jpeg'])
    button = st.button('تحلیل اسکن مغزی')       
    if image is not None:
        file_bytes = np.array(bytearray(image.read()), dtype= np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, channels= 'BGR', use_column_width= True)
        if button: 
            x = cv2.resize(img, (128, 128))
            x1 = img_to_array(x)
            x1 = x1.reshape((1,) + x1.shape)
            y_pred = modelh5.predict(x1)
            if y_pred == 1:
                with st.chat_message("assistant"):
                    with st.spinner('''در حال تحلیل'''):
                        time.sleep(2)
                        st.success(u'\u2713''تحلیل انجام شد')
                        text1 = 'بر اساس تحلیل من ، بخش هایی از بافت مغز کاربر تحلیل رفته است'
                        text2 = 'این تحلیل رفتگی می تواند ناشی از بروز خاموش زوال عقل ، آلزایمر و بیماری های مغزی مشابه باشد'
                        text3 = 'لطفا در اسرع وقت به پزشک مراجعه کنید'
                        text4 = 'Based on my analysis, owner of this MRI image has partially lost density of their brain tissue'
                        text5 = 'This has occured because of early stages of Dementia , Alzheimer or other brain diseases'
                        text6 = 'Please visit doctor as soon as possible'
                        def stream_data1():
                            for word in text1.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data1)
                        def stream_data2():
                            for word in text2.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data2)
                        def stream_data3():
                            for word in text3.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data3)
                        def stream_data4():
                            for word in text4.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data4)
                        def stream_data5():
                            for word in text5.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data5)
                        def stream_data6():
                            for word in text6.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data6)
        
            elif y_pred == 0:
                with st.chat_message("assistant"):
                    with st.spinner('''در حال تحلیل'''):
                        time.sleep(2)
                        st.success(u'\u2713''تحلیل انجام شد')
                        text1 = 'بر اساس تحلیل من ، بافت مغز در اسکن آپلود شده سالم است'
                        text2 = 'Based on my analysis , brain tissue in this MRI image is healthy and untouched'
                        def stream_data1():
                            for word in text1.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data1)
                        def stream_data2():
                            for word in text2.split(" "):
                                yield word + " "
                                time.sleep(0.09)
                        st.write_stream(stream_data2)

show_page()