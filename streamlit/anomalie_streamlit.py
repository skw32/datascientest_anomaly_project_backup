import streamlit as st
import keras
import numpy as np
from tensorflow.keras.preprocessing import image


st.title("Projet : Détection d'anomalies dans des pièces industrielles")
st.sidebar.title("Table des matières")

pages=["Présentation de base de données MVTec", "EDA", "Classification de type d'objet", "ML : Détection d'anomalies", "CNN : Classification binaire d'anomalies et segmentation", "CNN: classification multi-classe d'anomalies", "Conclusion et pérspectives"]

page=st.sidebar.radio("Navigation", pages)


if page == pages[0] : 
  # Makhlouf
  st.write("### Présentation de base de données MVTec")







if page == pages[1] : 
  # Suzy
  st.write("## EDA")

  st.write("#### Déséquilibre des classes")
  st.image('EDA_classe_desequilibre.png', caption="Nombre d'images dans les ensembles d'entraînement et de test MVTec pour chaque type d'objet.", 
           width=800, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
  st.image('EDA_classe_desequilibre_transistor.png', caption="Répartition des classes dans les ensembles d'entraînement et de test combinés pour le transistor.", 
           width=600, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

  st.write("#### Diversité dans la taille des images et la taille des défauts")
  st.image('EDA_taille_images.png', caption='Comparaison de la largeur et de la hauteur moyenne des images par catégorie.', 
           width=900, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

  st.image('EDA_taille_anomalies.png', caption="Comparaison des tailles des défauts (pour toutes les classes de défauts) pour tous les types d'objets.", 
           width=900, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

  st.write("#### Analyse de la qualité des données")
  st.image('EDA_MVTec_file_structure.png', caption="Structure des fichiers du jeu de données MVTec et exemple de masque « ground truth »", 
           width=510, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
  st.markdown("- Analyse des pixels pour vérifier qu'il n'y a pas de doublons ou d'images corrompues")
  st.markdown("- Valeurs aberrantes dans les tailles des défauts (par classe de défauts) utilisées pour vérifier qu'il n'y avait pas d'erreur d'étiquetage")






if page == pages[2] : 
  # Suzy
  st.write("## Classification de type d'objet")
  st.image('obj_classification_cm.png', caption="Matrice de confusion pour le classificateur de types d'objets multi-classes CNN", 
           width=800, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

  st.write("#### Démonstration de prédiction de classe d'objet")
  choice = ['image1.png', 'image2.png', 'image3.png']
  chosen_img = st.selectbox('Sélectionnez une image', choice)
  st.write('Image choisie :')
  st.image(chosen_img, caption="",
           width=300, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

  model_obj_classifier = keras.saving.load_model('../models/cnn_obj_classifier_SKW.keras')
  demo_img = image.load_img(chosen_img, target_size = (256, 256))
  demo_img = np.expand_dims(demo_img, axis = 0)
  prediction = model_obj_classifier.predict(demo_img, verbose=0)
  pred_label = int(np.argmax(prediction, axis=-1))
  proba = prediction[0][pred_label]* 100
  categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 
                'toothbrush', 'transistor', 'wood', 'zipper']
  st.write(f"Type d'objet prédit: {categories[pred_label]}")
  st.write(f"Probabilité : {proba} %")







if page == pages[3] : 
  # Karine
  st.write("### ML: Détection d'anomalies")

if page == pages[4] : 
  # Makhlouf
  st.write("### CNN: Classification binaire d'anomalies et segmentation")

if page == pages[5] : 
  # Karine
  st.write("### CNN: Classification multi-classe d'anomalies")

if page == pages[6] : 
  # Toute le monde ?
  st.write("### Conclusion et pérspectives")


