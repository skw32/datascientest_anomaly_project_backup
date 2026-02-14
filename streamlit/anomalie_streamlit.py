import streamlit as st
import keras
import numpy as np
from tensorflow.keras.preprocessing import image
import pandas as pd
from PIL import Image
import os
from pathlib import Path


st.title("Projet : Détection d'anomalies dans des pièces industrielles")
st.sidebar.title("Table des matières")

pages=["Présentation de base de données MVTec", "EDA", "Classification de type d'objet", "ML : Détection d'anomalies", "CNN : Classification binaire d'anomalies et segmentation", "CNN: classification multi-classe d'anomalies", "Conclusion et pérspectives"]

page=st.sidebar.radio("Navigation", pages)


if page == pages[0] : 
  # Makhlouf
  st.write("## Présentation de base de données MVTec")

  st.write("###  Context industrielle du projet:")

  st.write("###  Objectif du projet:")
  st.write("- L’objectif est de développer un système capable de détecter " \
                 "automatiquement des anomalies sur des pièces industrielles à partir d’images en utilisant le dataset MVTec AD.")
  
  # Exemple de table (à adapter avec tes vrais chiffres)
  df_mvtec = pd.DataFrame({
    "Catégorie": ["grid", "bottle", "cable",],
    "Type": ["texture", "object", "object"],
    "Train (good)": [264, 209, 224],
    "Test (total)": [264, 190, 200],
    "Nb types défauts": [5, 3, 4],
    })

  st.subheader("Description du dataset MVTec AD")
  st.dataframe(df_mvtec, use_container_width=True)

   # stats rapides
  st.write("Total images train (good):", df_mvtec["Train (good)"].sum())
  st.write("Total images test:", df_mvtec["Test (total)"].sum())

  st.write("### dataset MVTec AD:")
  st.write("- Dataset de reference pour la detetction d'anomalie industrile par vision par ordinateur:")
  st.write("-  15 categories industrielle")
  st.write("-  10 d'objets")
  st.write("-  5 de textures")
  st.image('repartition_train_test_global.png',
              caption="Répartition des images du dataset MVTec entre les dossiers train et test..", 
           width=500, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
  
  ### ajout de la frisse chronoloigique pour le deroulement des etapes du projet

  st.subheader("Deroulement du projet:")
  st.image('frisse_chronologique_projet.png',
              caption="Frisse chronologique du deroulement des étapes du projet", 
           width=700, use_column_width=None, clamp=False, channels="RGB", output_format="auto")


################################################


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
  st.write("#### Détails du meilleur modèle")
  st.markdown("- CNN architecture peu profonde (1 couche de convolution + 1 couche de pooling + 1 couche dense)")
  st.markdown("- Taille d'images : 256x256")
  st.markdown("- Early stopping callback pour éviter le surapprentissage")
  st.markdown("- Ensemble de données d'entraînement (uniquement des images de classe bonne, sans défauts)")
  st.markdown("- Ensemble de données de validation (mélange d'images de classe bonne et d'images défectueuses)")
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

  st.write("#### Interprétabilité avec SHAP")
  st.image('obj_classification_SHAP.png', caption="", 
           width=900, use_column_width=None, clamp=False, channels="RGB", output_format="auto")







if page == pages[3] : 
  # Karine
  st.write("### ML: Détection d'anomalies")


########################################################
if page == pages[4] : 
  # Makhlouf
  st.write("### CNN: Classification binaire d'anomalies et segmentation")
  st.write("#### 1) Pipeline de préparation des données et d’entraînement du modèle binaire:")
  st.image('Pipeline_complet_de_préparation_des_données_et_de_detection_anomalies_binaire.png',
              caption="ajout de titre", width=700)
  


  st.write("#### 2) Resultats obtenu et demonstration:")

  st.write("## Démonstration – Détection d'anomalies")
  ## Choix des images:
  choice = ['img_defect_1.png', 'img_defect_2.png', 'img_defect_3.png','img_good_1.png','img_good_2.png']
  chosen_img = st.selectbox('Sélectionnez une image', choice, key="chosen_img")

  APP_DIR = Path(__file__).resolve().parent
  img_slot = st.empty()
  grad_slot = st.empty()

  st.write("Image choisie :")
  ##st.image(chosen_img,width=400)
  chosen_path = APP_DIR / chosen_img

  if chosen_path.exists() and chosen_path.stat().st_size > 0:
    with Image.open(chosen_path) as im:
      img_slot.image(im, width=300)
  else:
    img_slot.warning(f"Image choisie introuvable ou vide : {chosen_path}")

  # Chargement modèle
  model_anomaly = keras.saving.load_model('../models/tl_MobileNet_model_binaire_Makhlouf_Hanouti.keras')

  # Préparation image
  demo_img = image.load_img(chosen_img, target_size=(256, 256))
  demo_img = image.img_to_array(demo_img)
  demo_img = np.expand_dims(demo_img, axis=0)

  # Prédiction
  prediction = model_anomaly.predict(demo_img, verbose=0)[0][0]
  proba = prediction * 100

  st.write(f"Probabilité d'anomalie est de : {proba:.4f} %")

  # Décision finale
  threshold = 0.3

  if prediction >= threshold:
    st.error(" Anomalie détectée sur cette pièce")
  else:
    st.success("Pièce normale (aucune anomalie détectée)")

  #  GRAD CAM 
  
  st.write("### Interprétabilité (Grad-CAM)")
  with st.container():
    grad_slot = st.empty()  

    gradcam_path = APP_DIR / chosen_img.replace(".png", "_gradcam.png")
    if gradcam_path.exists() and gradcam_path.stat().st_size > 0:
        with Image.open(gradcam_path) as im:
            grad_slot.image(im, caption="Grad-CAM", use_container_width=True)
    else:
        grad_slot.warning(f"Grad-CAM introuvable ou vide : {gradcam_path}")

  #st.write("### Interprétabilité (Grad-CAM)")

  #gradcam_path = f"{chosen_img.replace('.png', '_gradcam.png')}"

  #st.image(gradcam_path, caption="Grad-CAM (déjà calculé)", width=800)

  ##resultat de la Segmentation:
  st.write("## Modele de Segmentation:")



######################################

if page == pages[5] : 
  # Karine
  st.write("### CNN: Classification multi-classe d'anomalies")

if page == pages[6] : 
  # Toute le monde ?
  st.write("### Conclusion et pérspectives")


