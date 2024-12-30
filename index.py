import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
import streamlit as st 

classes = {4: ('nv', 'melanocytic nevi'),
           6: ('mel', 'melanoma'),
           2: ('bkl', 'benign keratosis-like lesions'), 
           1:('bcc', 'basal cell carcinoma'),
           5: ('vasc', 'pyogenic granulomas and hemorrhage'),
           0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
           3: ('df', 'dermatofibroma')}

#load the trained model from a .pkl file
def load_trained_model(model_path='model.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

#format the img & predict
def predict_image(img_path, model):
    img = Image.open(img_path)
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0

    if img_array.shape[-1] != 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    img_array = np.expand_dims(img_array, axis=0) # [1,28,28,3]
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=-1)
    
    return predicted_class[0], prediction[0]

st.set_page_config(page_title="Skin Cancer Detection", page_icon="🔍", layout="wide")

# Initialize session state 
if "page" not in st.session_state:
    st.session_state.page = "Home"  # Default page is 'Home'

col1, col2, col3 = st.columns([1, 2, 1])

#navbar
with col3:
    app_mode = st.pills(
        label="",
        options=["Home", "Recognition", "About"],
        selection_mode="single",
        key="app_mode"
    )

# Check if the page state has changed, and update accordingly
if app_mode != st.session_state.page:
    st.session_state.page = app_mode

if st.session_state.page == "Home":
    st.markdown("""
    <style>
    .welcome-message {
        font-size: 36px;
        text-align: center;
        color: #4e73df;
        font-weight: bold;
        margin-top: 50px;
    }
    .home-description {
        text-align: center;
        font-size: 18px;
        color: #6c757d;
        margin-top: 10px;
    }
    </style>
    <div class="welcome-message">
        Welcome to the Skin Cancer Recognition System!
    </div>
    <div class="home-description">
        Early detection of skin cancer can save lives. Upload your skin image and let our model provide an instant diagnosis.
    </div>
    """, unsafe_allow_html=True)
    
    image_path = "home.jfif"  # Image
    st.image(image_path, caption="Skin Cancer Detection", use_container_width=True)

elif st.session_state.page == "Recognition":
    st.header("Skin Check")
    test_img = st.file_uploader("Choose an image")
    
    if test_img is not None:
        st.image(test_img, use_container_width=False, width=400)  

    if st.button("Predict"):
      if test_img is not None:
          model = load_trained_model('model.pkl')
          predicted_class, prediction_probs = predict_image(test_img, model)
          
          st.markdown(f"""
          <div style="font-size: 20px; color: #ff5733; text-align: center;">
              You have been diagnosed with <span style="font-size: 28px; font-weight: bold;">{classes[predicted_class][1]}</span>.
          </div>
          <div style="font-size: 18px; color: #6c757d; text-align: center;">
              To learn more about the next steps and gain insights on how to manage your condition, visit the <span style="font-weight: bold; color: #4e73df;">About</span> section.
          </div>
          """, unsafe_allow_html=True)
      else:
          st.error("Please upload an image to predict.")

elif st.session_state.page == "About": # Recommendation
    st.header("About the Diseases")
    
    with st.expander("Melanocytic Nevi (nv)") :
        st.write("Melanocytic Nevi are common moles that are usually harmless. They can appear as dark or slightly raised skin lesions.")
        st.write("### What to do:")
        st.write("If the mole changes in size, color, or shape, it's best to consult a dermatologist for further examination.")
    
    with st.expander("Melanoma (mel)") :
        st.write("Melanoma is a serious form of skin cancer that develops in the melanocytes, the cells responsible for producing pigment.")
        st.write("### What to do:")
        st.write("If you notice any new or unusual growths or changes in an existing mole, seek medical attention immediately.")
    
    with st.expander("Benign Keratosis-like Lesions (bkl)") :
        st.write("These are non-cancerous lesions that often appear as rough, scaly patches on the skin.")
        st.write("### What to do:")
        st.write("Though benign, it’s still important to monitor any changes and seek medical advice if they become painful or grow rapidly.")
    
    with st.expander("Basal Cell Carcinoma (bcc)") :
        st.write("Basal Cell Carcinoma is the most common type of skin cancer, appearing as a small, shiny bump or nodule on sun-exposed skin.")
        st.write("### What to do:")
        st.write("If you notice any unusual growths, sores that don’t heal, or changes in the skin, seek medical advice immediately.")
    
    with st.expander("Pyogenic Granulomas and Hemorrhage (vasc)") :
        st.write("These are vascular lesions that appear as red, raised, and often bleeding spots on the skin.")
        st.write("### What to do:")
        st.write("If bleeding or rapid growth occurs, see a dermatologist to rule out any underlying conditions.")
    
    with st.expander("Actinic Keratoses and Intraepithelial Carcinomas (akiec)") :
        st.write("These are precancerous spots or patches that form on sun-damaged skin and can develop into squamous cell carcinoma.")
        st.write("### What to do:")
        st.write("These lesions should be monitored regularly and treated by a dermatologist to prevent progression to skin cancer.")
    
    with st.expander("Dermatofibroma (df)") :
        st.write("Dermatofibroma is a common, benign skin growth that often appears as a small, brown, raised bump.")
        st.write("### What to do:")
        st.write("These growths are typically harmless, but if they become painful or change in appearance, consult a dermatologist.")
