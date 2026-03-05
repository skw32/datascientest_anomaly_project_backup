import streamlit as st
import keras
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.saving import register_keras_serializable
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from keras.layers import Lambda
import pandas as pd
from PIL import Image
import os
from pathlib import Path
import gdown

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image

BASE_DIR = Path(__file__).parent

st.title("Projet : Détection d'anomalies dans des pièces industrielles")
st.sidebar.title("Table des matières")

pages=["Introduction", "EDA", "Classification de type d'objet", "ML : Détection d'anomalies", "CNN : Classification binaire d'anomalies et segmentation", "CNN: classification multi-classe d'anomalies", "Conclusion et pérspectives"]

page=st.sidebar.radio("Navigation", pages)


if page == pages[0] : 
  # Makhlouf
  st.write("## Introduction")
  st.markdown("""Dans l’industrie, garantir la qualité des pièces produites est essentiel pour éviter :
   - retours clients.
  - pertes financières.
  - risques de sécurité.
  - arrêts de production.""")
  st.write("###  Context industrielle du projet:")
  st.markdown("""L’inspection humaine est souvent :
   - lente et couteuse .
   - fatigante.
   - sujette aux erreurs.
    surtout lorsque les défauts sont petits ou difficiles à voir. """)
  st.image( BASE_DIR / 'contexte_industr_1.png',
              caption="Inspection humain vs Ordinateur", 
           width=800, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
  
  st.markdown("""### 3) Apport de la vision par ordinateur
  La vision par ordinateur permet :
   - une inspection automatique et continue.
   - un contrôle en temps réel.
   - une meilleure répétabilité.
   - une réduction des coûts de non-qualité. """)
  st.image( BASE_DIR /'contexte_industr_2.png',
              caption="Exemple inspection de piece industrielle automatiquement", 
           width=800, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
  

  st.write("###  Objectif du projet:")
  st.write("- L’objectif est de développer un système capable de détecter " \
                 "automatiquement des anomalies sur des pièces industrielles à partir d’images en utilisant le dataset MVTec AD.")
  


  st.write("### Dataset MVTec AD:")
  st.markdown("- Dataset de reference pour la detetction d'anomalie industrile par vision par ordinateur:")
  st.markdown("-  15 categories industrielle")
  st.markdown("-  10 d'objets")
  st.markdown("-  5 de textures")
  st.image( BASE_DIR / 'image_exemple_data_MVTec.png',
              caption="Exemple Dataset MVTec", 
           width=800, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
  st.image( BASE_DIR / 'repartition_train_test_global.png',
              caption="Répartition des images du dataset MVTec entre les dossiers train et test..", 
           width=500, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

  
    # Exemple de table (garder le nombre d'anomalies et le type de la categorie)
  df_mvtec = pd.DataFrame({
    "Catégorie": ["grid", "bottle", "capsule","cable","carpet","hazelnut","leather",
                  "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"],
    "Type": ["texture", "object", "object", "object", "texture", "object", "texture", "object", 
             "object", "object", "texture", "object", "object", "texture", "object"],
    "Nb types défauts": [5, 3, 5, 8, 5, 4, 5, 4, 7, 5, 5, 1, 4, 5, 7] ,
    })
  
  st.subheader("Description du dataset MVTec AD")
  st.dataframe(df_mvtec, use_container_width=True)
  ### ajout de la frisse chronoloigique pour le deroulement des etapes du projet

  st.subheader("Deroulement du projet:")
  st.image( BASE_DIR / 'frisse_chronologique_projet.png',
              caption="Frisse chronologique du deroulement des étapes du projet", 
           width=700, use_column_width=None, clamp=False, channels="RGB", output_format="auto")


################################################


if page == pages[1] : 
  # Suzy
  st.write("## EDA")

  st.write("#### Déséquilibre des classes")
  st.image( BASE_DIR / 'EDA_classe_desequilibre.png', caption="Nombre d'images dans les ensembles d'entraînement et de test MVTec pour chaque type d'objet.", 
           width=800, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
  st.image( BASE_DIR / 'EDA_classe_desequilibre_transistor.png', caption="Répartition des classes dans les ensembles d'entraînement et de test combinés pour le transistor.", 
           width=600, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

  st.write("#### Diversité dans la taille des images et la taille des défauts")
  st.image( BASE_DIR / 'EDA_taille_images.png', caption='Comparaison de la largeur et de la hauteur moyenne des images par catégorie.', 
           width=900, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

  st.image( BASE_DIR / 'EDA_taille_anomalies.png', caption="Comparaison des tailles des défauts (pour toutes les classes de défauts) pour tous les types d'objets.", 
           width=900, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

  st.write("#### Analyse de la qualité des données")
  st.image( BASE_DIR / 'EDA_MVTec_file_structure.png', caption="Structure des fichiers du jeu de données MVTec et exemple de masque « ground truth »", 
           width=510, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

  st.markdown("""
  - Analyse des pixels pour vérifier qu'il n'y a pas de doublons ou d'images corrompues.
  - Valeurs aberrantes dans les tailles des défauts (par classe de défauts) utilisées pour vérifier qu'il n'y avait pas d'erreur d'étiquetage.
  """)






if page == pages[2] : 
  # Suzy
  st.write("## Classification de type d'objet")
  st.write("#### Détails du meilleur modèle")
  st.markdown("""
  - CNN architecture peu profonde (1 couche de convolution + 1 couche de pooling + 1 couche dense)
  - Taille d'images : 256x256
  - Early stopping callback pour éviter le surapprentissage
  - Ensemble de données d'entraînement (uniquement des images de classe bonne, sans défauts)
  - Ensemble de données de validation (mélange d'images de classe bonne et d'images défectueuses)
  """)
  
  st.image( BASE_DIR / 'obj_classification_cm.png', caption="Matrice de confusion pour le classificateur de types d'objets multi-classes CNN", 
           width=800, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

  st.write("#### Démonstration de prédiction de classe d'objet")
  choice = [ BASE_DIR/'image1.png',  BASE_DIR/'image2.png',  BASE_DIR/'image3.png']
  chosen_img = st.selectbox('Sélectionnez une image', choice)
  st.write('Image choisie :')
  st.image(chosen_img, caption="",
           width=300, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

  model_obj_classifier = keras.saving.load_model(BASE_DIR.parent / 'models/cnn_obj_classifier_SKW.keras')
  demo_img = image.load_img(chosen_img, target_size = (256, 256))
  demo_img = np.expand_dims(demo_img, axis = 0)
  prediction = model_obj_classifier.predict(demo_img, verbose=0)
  pred_label = int(np.argmax(prediction, axis=-1))
  proba = prediction[0][pred_label]* 100
  categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 
                'toothbrush', 'transistor', 'wood', 'zipper']
  st.markdown(f"Type d'objet prédit: **{categories[pred_label]}**")
  st.markdown(f"Probabilité : **{proba} %**")

  st.write("#### Interprétabilité avec SHAP")
  st.image( BASE_DIR / 'obj_classification_SHAP.png', caption="", 
           width=900, use_column_width=None, clamp=False, channels="RGB", output_format="auto")







if page == pages[3] : 
  # Karine
  st.header("Modèles de Machine Learning")
  st.subheader("Données")

  # Functions
  def displayheatmap(cm, class_names):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,        # show numbers
        fmt=".2f",           # float format with 2 decimal places
        cmap="Blues",      # color map
        cbar=False,
        xticklabels=class_names, yticklabels=class_names
    )
    
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Matrice de confusion")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    st.pyplot(plt)

  def KNN_Oversampling_TrainAndPredict(X_train,y_train,X_test,y_test, is_defectType=False):  
    st.write('Modèle KNN avec suréchantillonnage – Entraînement et Prédiction')
    pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(k_neighbors=5)),
    ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance"))
    ])
    pipeline.fit(X_train, y_train)
    st.write('Précision sur les données d’entraînement: ', pipeline.score(X_train, y_train))
    y_pred = pipeline.predict(X_test)
    st.write('Précision du modèle: ', accuracy_score(y_test, y_pred))
    class_names = ['Good', 'Anomalous'] if not is_defectType else ['crack', 'faulty_imprint', 'good', 'poke','scratch','squeeze']
    st.write('Rapport de classification:')
    st.code(classification_report(y_test, y_pred, target_names = class_names))
    
    cm = confusion_matrix(y_test,y_pred)
    displayheatmap(cm, class_names)

  def SVM_Oversampling_TrainAndPredict(X_train,y_train,X_test,y_test, is_defectType=False):
    st.write('Modèle SVC avec suréchantillonnage – Entraînement et Prédiction')
    pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    #("svm", SVC())    
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    st.write('Précision sur les données d’entraînement: ', pipeline.score(X_train, y_train))
    y_pred = pipeline.predict(X_test)
    st.write('Précision du modèle: ', accuracy_score(y_test, y_pred))
    class_names = ['Good', 'Anomalous'] if not is_defectType else ['crack', 'faulty_imprint', 'good', 'poke','scratch','squeeze']
    st.write('Rapport de classification:')
    st.code(classification_report(y_test, y_pred, target_names = class_names))
    cm= confusion_matrix(y_test,y_pred)
    displayheatmap(cm, class_names)


  def RF_Oversampling_TrainAndPredict(X_train,y_train,X_test,y_test, is_defectType=False):
    st.write('Modèle Random Forest avec suréchantillonnage – Entraînement et Prédiction')
    pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("rf", RandomForestClassifier(n_jobs=-1, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    st.write('Précision sur les données d’entraînement: ', pipeline.score(X_train, y_train))
    y_pred = pipeline.predict(X_test)
    st.write('Précision du modèle: ', accuracy_score(y_test, y_pred))
    class_names = ['Good', 'Anomalous'] if not is_defectType else ['crack', 'faulty_imprint', 'good', 'poke','scratch','squeeze']
    st.write('Rapport de classification:')
    st.code(classification_report(y_test, y_pred, target_names = class_names))

    cm = confusion_matrix(y_test,y_pred)
    displayheatmap(cm, class_names)


  def prediction(classifier, is_defectType=False):
    if classifier == 'KNN':
      KNN_Oversampling_TrainAndPredict(X_train,y_train,X_test,y_test, is_defectType=is_defectType)
    elif classifier == 'SVC':
      SVM_Oversampling_TrainAndPredict(X_train,y_train,X_test,y_test, is_defectType=is_defectType)
    else:
      RF_Oversampling_TrainAndPredict(X_train,y_train,X_test,y_test, is_defectType=is_defectType)
 
  
  # Data
  df = pd.read_csv(BASE_DIR / "DataFiles/mvtec_full_statistiques_features_colour_images.csv")
  df = df.drop(columns=["img", "file_path", "mean", "std","skew" ])
  df = df[df["category_name"] == "capsule"]
  st.write("Capsules anormales")
  st.dataframe(df.head(10))
  st.write("Capsules normales")
  st.dataframe(df.tail(10))

  st.image( BASE_DIR / 'Images/Capsule_1.png', caption="Bonne capsule", 
           width=200, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
  st.image( BASE_DIR / 'Images/Capsule_2.png', caption="Capsule anormale", 
           width=200, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

  # Prepare data for ML models and split into train and test sets
  X = df.select_dtypes(include='number')
  X= X.drop(['label'], axis=1)
  y= df['label']

  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% for testing
    random_state=42,  # reproducible split
    stratify=y        # preserves class distribution
  )

  # Anamoly detection models
  st.subheader("Modèles de détection d’anomalies")

  # Select Models and predict
  choix = ['KNN', 'SVC', 'Random Forest']
  option = st.selectbox('Choix du modèle', choix, key='selectbox_anamoly_detection')
  #st.write('Le modèle choisi est :', option)
  prediction(option)  

  # Defect type detection models
  st.subheader("Modèles de détection de type d’anomalie")
  # Prepare data for ML models and split into train and test sets
  X = df.select_dtypes(include='number')
  y= df['dir_name']

  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% for testing
    random_state=42,  # reproducible split
    stratify=y        # preserves class distribution
  )

  # Select Models and predict
  choix_defectType = ['KNN', 'SVC', 'Random Forest']
  option_defectType = st.selectbox('Choix du modèle', choix_defectType, key='selectbox_defectType_detection')
  #st.write('Le modèle choisi est :', option_defectType)
  prediction(option_defectType, is_defectType=True)

  # Important points
  st.write("#### Points clés:")
  st.markdown("""
  - Images redimensionnées à 256×256 pixels.
  - Jeu de données déséquilibré → utilisation de SMOTE.
  - Évaluation via rappel négatif, précision, rapport de classification et matrice de confusion.
  - Random Forest surpasse KNN et SVC pour la détection d’anomalies et le type de défaut.
  - XGBoost : résultats médiocres.
  - Des filtres (HOG, Canny, ...) n’améliorent pas les performances.
  """)



########################################################
if page == pages[4] : 
  # Makhlouf
 
  st.write("### Pipeline de préparation des données et d’entraînement du modèle binaire:")
  
  st.image( BASE_DIR / 'Pipeline_complet_de_préparation_des_données_et_de_detection_anomalies_binaire.png',
              caption="ajout de titre", width=700)
  


  st.write("## CNN: Classification binaire d'anomalies et segmentation")
  st.write("### Resultats obtenu:")
 
  st.image( BASE_DIR / "resultats_metrics_resume.png", use_container_width=True)
  st.image( BASE_DIR / "resultats_metrics_par_classe.png", use_container_width=True)
  st.image( BASE_DIR / 'matrice_confusion_segmentation.png',caption="ajout de titre", width=700)

  st.write("### Démonstration – Détection d'anomalies")
  ## Choix des images:
  choice = [ BASE_DIR/'img_defect_1.png',  BASE_DIR/'img_defect_2.png',  BASE_DIR/'img_defect_3.png', BASE_DIR/'img_good_2.png']
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
  model_anomaly = keras.saving.load_model(BASE_DIR.parent / 'models/tl_MobileNet_model_binaire_Makhlouf_Hanouti.keras')

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



####### SEGMENTATION #############

  ##resultat de la Segmentation:
  st.write("## Modele de Segmentation:")
  st.write("### Pipeline de préparation des données et d’entraînement du modèle segmentation:")
  st.image( BASE_DIR / 'Pipeline_complet_de_préparation_des_données_et_de_detection_defaut_segmentation.png',
              caption="ajout de titre", width=700)
  
  ##
  st.write("### Démonstration – Détection et localisation d'anomalies")
  ## Choix des images:
  
  choice = [ BASE_DIR / 'img_defect_segmentation_1.png', BASE_DIR / 'img_good_segmentation_1.png']
  
  chosen_img = st.selectbox('Sélectionnez une image', choice, key="chosen_img_seg")
  st.write("Image choisie :")
  ###


  APP_DIR = Path(__file__).resolve().parent
  img_slot = st.empty()
  grad_slot = st.empty()

  ##st.write("Image choisie :")
  ##st.image(chosen_img,width=400)
  chosen_path = APP_DIR / chosen_img

  if chosen_path.exists() and chosen_path.stat().st_size > 0:
    with Image.open(chosen_path) as im:
      img_slot.image(im, width=400)
  else:
    img_slot.warning(f"Image choisie introuvable ou vide : {chosen_path}")

  # Chargement modèle


  FILE_ID = "1xV_GU_H0KLfpFXzUuNvklZTUWzfLb95_"
  MODEL_PATH = Path("model.keras")

  if not MODEL_PATH.exists():
    st.write("Téléchargement du modèle...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, str(MODEL_PATH), quiet=False)
  @register_keras_serializable()
  def preprocess_input(x):
    # version ResNet50
    return tf.keras.applications.resnet50.preprocess_input(x)
  
  def preprocess_lambda_layer(**kwargs):
    # output_shape = identique à l'entrée: (256, 256, 3)
    return Lambda(preprocess_input, output_shape=(256, 256, 3), name="preprocess", **kwargs)


   # Chargement robuste (pas besoin de metrics/loss)
  model_anomaly_segmentation = keras.saving.load_model(MODEL_PATH,compile=False,safe_mode=False, custom_objects={"preprocess_input": preprocess_input,
                                                       "preprocess_lambda_layer": preprocess_lambda_layer,"Lambda": Lambda},)

   # Préparation image
  demo_img = image.load_img(chosen_img, target_size=(256, 256))
  demo_img = image.img_to_array(demo_img)
  demo_img = np.expand_dims(demo_img, axis=0).astype("float32")

  threshold = 0.8

  # Prédiction segmentation
  pred = model_anomaly_segmentation.predict(demo_img, verbose=0)

  # masque 2D
  pred_mask = pred[0, :, :, 0]  # (256,256)

  # convertir en "score défaut" scalaire
  score_defaut = float(pred_mask.max())  

  st.write(f"Probabilité défaut : {score_defaut * 100:.2f} %")

  if score_defaut >= threshold:
    st.error(" Défaut détecté")
  else:
    st.success("Pas de défaut détecté")

  #  Resultat Segmentation: 
  
  st.write("### Interprétabilité (Sortie Segmentation)")
  with st.container():
    grad_slot = st.empty()  

    gradcam_path = APP_DIR / chosen_img.replace(".png", "_segmentation.png")
    if gradcam_path.exists() and gradcam_path.stat().st_size > 0:
        with Image.open(gradcam_path) as im:
            grad_slot.image(im, caption="Resultat de Segmentation", use_container_width=True)
    else:
        grad_slot.warning(f"segmentation introuvable ou vide : {gradcam_path}")



######################################

if page == pages[5] : 
  # Suzy + Karine
  st.write("### CNN: Classification multi-classe d'anomalies de Transistor")

  st.subheader("Augmentation des données")
  st.markdown("""
  - Pré-division des données en ensembles d'entraînement, de validation et de test
  - Ensuite augmentation des images contenant des anomalies pour équilibrer les classes avec Keras ImageDataGenerator
  - Utilisation de techniques d'augmentation d'images telles que la rotation, le zoom et les translations pour augmenter artificiellement la taille de l'ensemble de données
  - Création de nouvelles images à partir des images existantes
  """)
  st.image( BASE_DIR / 'Images/transistor_class_balance.png', 
           caption="Nombre total de transistors bons et anormaux dans l'ensemble de données pour le modèle de détection des types de défauts (avant et après augmentation des données)", 
           width=900, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

  st.image( BASE_DIR / 'Images/grid_data_aug.png', 
           caption="Exemple où certains types de distorsions résultant de l'augmentation de l'image ont induit le modèle en erreur, lui faisant croire que d'autres types de défauts étaient présents.", 
           width=650, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

  st.subheader("Modèle de détection du type de défaut (transfer learning avec VGG16)")

  st.image( BASE_DIR / 'Images/transistor_cm_vgg16.png', caption="Matrice de confusion du modèle de détection du type de défaut (transfer learning avec VGG16)", 
           width=700, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

  st.image( BASE_DIR / 'Images/bent_lead_vs_cut_lead.png', caption="Exemple de défauts de transistor : cut_lead (à gauche) et bent_lead (à droite)", 
           width=500, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
  



  # Load defect type detection model
  st.subheader("Modèle de détection du type de défaut (transfer learning avec MobileNet)")

  st.write("#### Matrice de confusion")
  st.image( BASE_DIR / 'Images/model_defectType_MobileNet_Confusion_matrix.png', caption="Matrice de confusion du modèle de détection du type de défaut (transfer learning avec MobileNet)", 
           width=700, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
  
  loaded_model = load_model(BASE_DIR /  'Models/model_predict_anamoly_MobileNet.keras')
  st.write("Chargement du modèle réussi!")

  # Prediction on new images
  st.write("#### Démonstration de la prédiction du type d’anomalie sur une image de transistor")
  choice = [ BASE_DIR/'Images/transistor_1.png', BASE_DIR/'Images/transistor_2.png', BASE_DIR/'Images/transistor_3.png', BASE_DIR/'Images/transistor_4.png', BASE_DIR/'Images/transistor_5.png']
  chosen_img = st.selectbox('Sélectionnez une image', choice, key='selectbox_model_defectType')
  st.write('Image choisie :')
  st.image(chosen_img, caption="",
           width=300, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

  demo_img = image.load_img(chosen_img, target_size = (256, 256))
  demo_img = np.expand_dims(demo_img, axis = 0)
  demo_img = preprocess_input(demo_img)

  prediction = loaded_model.predict(demo_img, verbose=0)
  pred_label = int(np.argmax(prediction, axis=-1))
  proba = prediction[0][pred_label]* 100
  categories = ['bent_lead', 'cut_lead', 'damaged_case', 'good','misplaced']
  st.markdown(f"Type de défaut prédit: **{categories[pred_label]}**")
  st.markdown(f"Probabilité : **{proba:0.2f} %**")

  # Gtrad-CAM visualization
  st.write("#### Grad-CAM visualization")

  if st.button("Afficher exemple image Grad-CAM"):
    st.image( BASE_DIR / 'Images/Grad-CAM_Transistor.png', caption="Grad-CAM – détection du type de défaut sur une image de transistor", 
           width=500, use_column_width=None, clamp=False, channels="RGB", output_format="auto")  
  

if page == pages[6] : 
  # Toute le monde ?
  st.write("### Conclusion et pérspectives")
  st.write("## Conclusion")

  st.write("""
  Ce projet a permis de mettre en place un pipeline complet de détection d’anomalies industrielles à partir d’images du dataset MVTec AD.

  Plusieurs approches ont été étudiées, depuis des méthodes classiques de machine learning jusqu’aux modèles de deep learning utilisant le transfert leaning 
           Les résultats montrent que les réseaux de neurones convolutifs permettent une détection plus robuste des anomalies, 
           en particulier lorsqu’ils sont combinés avec une étape de segmentation permettant de localiser précisément les défauts.

  Deux pipelines complémentaires ont été proposés :
  - une classification binaire suivie d’une segmentation,
  - une classification multi-classes combinée à la segmentation.

  Ces solutions permettent d’adapter le système aux besoins industriels, que ce soit pour détecter rapidement un défaut ou pour localiser précisément les zones endommagées.

  Ce travail montre ainsi le potentiel du deep learning pour l’automatisation du contrôle qualité industriel, 
           tout en soulignant certaines limites actuelles pour la détection de défauts très petits ou peu visibles.
  """)

  st.write("## Perspectives et améliorations futures")

  st.write("""
  Plusieurs améliorations peuvent être envisagées pour rendre le système plus robuste et plus adapté à un environnement industriel réel.

  Parmi les principales pistes d’amélioration :

  • Optimiser davantage l’entraînement des modèles (learning rate, batch size, nombre d’époques).

  • Tester des architectures plus récentes comme EfficientNet, ConvNeXt ou des modèles spécialisés pour l’analyse d’images industrielles.

  • Utiliser des images de plus haute résolution ou découper les images en patches afin de mieux détecter les petites anomalies.

  • Enrichir les données d’entraînement avec des données réelles ou des images synthétiques générées artificiellement.

  • Explorer des approches non supervisées ou semi-supervisées permettant d’apprendre uniquement à partir d’images normales.

  • Améliorer les modèles de segmentation pour obtenir une localisation plus précise des défauts.

  • Développer un système hybride combinant plusieurs modèles spécialisés selon le type de pièce analysée.

  Ces évolutions permettraient de tendre vers un système de détection fiable capable de fonctionner dans des conditions industrielles réelles.
  """)
    

