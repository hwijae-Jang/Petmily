import os
import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
# from tensorflow.keras.applications import ResNet50 
from tensorflow.keras.preprocessing import image as img_prep
import streamlit as st
import numpy as np
import PIL
from PIL import Image

CLASS_NAMES = [
    "결막염",
    "궤양성각막질환",
    "백내장",
    "비궤양성각막질환",
    "색소침착성각막염",
    "안검내반증",
    "안검염",
    "안검종양",
    "유루증",
    "핵경화"
]

base_model = EfficientNetB0(weights="imagenet", include_top=False)


# 기존 모델에서 출력 레이어를 포함하지 않은 새로운 모델을 만듭니다.
new_base_model = Model(base_model.inputs, base_model.layers[-1].output)

# 출력 레이어 전에 GlobalAveragePooling2D 레이어를 추가합니다.
x = GlobalAveragePooling2D()(new_base_model.output)

# 출력 레이어를 수정하여 클래스 수를 갖도록 합니다.
new_output_layer = Dense(len(CLASS_NAMES), activation='softmax')(x)

# 새로운 출력으로 최종 모델을 만듭니다.
model = Model(new_base_model.inputs, new_output_layer)

output_classes = model.output_shape[-1]  # 모델의 출력 클래스 수를 구합니다.
print("Model output classes: ", output_classes)
print("CLASS_NAMES count: ", len(CLASS_NAMES))

T=len(CLASS_NAMES)

print("Classic Model Complete!\n------------")

print("Model output_shape: ", model.output_shape)
print("Class names length: ", T)


def get_predictions(img):
    img_array = img_prep.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    preds = model.predict(img_array)
    
    return preds[0]


st.title("반려동물 안구질환 예측 모델")
uploaded_file = st.file_uploader("이미지를 선택해주세요...",
                                 type=["jpg", "jpeg", "png"])

def get_sorted_predictions(img):
    preds = get_predictions(img)
    sorted_preds = sorted(enumerate(preds), key=lambda x: x[1], reverse=True)
    
    return sorted_preds

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    resized_image = input_image.resize((224, 224), PIL.Image.ANTIALIAS)
    st.image(input_image, "Uploaded Image", use_column_width=True)
    
    sorted_predictions = get_sorted_predictions(resized_image)
    
    # 각 클래스별 확률을 출력합니다.
    st.write("Class-wise prediction probabilities in descending order:")
    for i, (index, pred) in enumerate(sorted_predictions):
        class_name = CLASS_NAMES[index]
        st.write(f"{i + 1}. {class_name}: {pred * 100:.2f}%")
