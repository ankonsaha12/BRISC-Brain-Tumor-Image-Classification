# BRISC Brain Tumor Classification

A Flask web app from our undergrad capstone. Upload a brain MRI and it predicts glioma, meningioma, pituitary tumor or no tumor, and shows a Grad-CAM heatmap of where the model looked.

The model is a stacked ensemble of InceptionV3, DenseNet201 and Xception, trained on the BRISC dataset.

To run it: `pip install -r requirements.txt`, then `python app.py`.

Student project only, not for medical use.
