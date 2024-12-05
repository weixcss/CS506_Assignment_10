## Demonstration
Watch the video below to see a full demonstration of the Image Search application in action:
[![Image Search](https://img.youtube.com/vi/iY7UeGXafnE/0.jpg)](https://youtu.be/iY7UeGXafnE)

# Assignment 10: Image Search

---

In this assignment, we implemented a simplified version of Google Image Search using a Flask web application. The application allows users to perform three types of searches—`Image query`, `Text query`, and `Hybrid query`—on a dataset of images, using CLIP embeddings for similarity computation.

## Task Overview
- **Core Functionalities**:
  - **Image Query**: Upload an image, and the app returns the top 5 most relevant images from the dataset.
  - **Text Query**: Enter a text query, and the app returns the top 5 most relevant images based on semantic similarity.
  - **Hybrid Query**: Combine both text and image inputs, with a user-defined weighting (λ), to return the top 5 results.

- **Technologies Used**:
  - **Model**: CLIP (ViT-B-32) for computing image and text embeddings.
  - **Framework**: Flask for the web application.
  - **Frontend**: HTML and CSS for the user interface.
  - **Data**: Precomputed image embeddings and the COCO dataset.

---

## Part 0: Setup Environment

You can use the `Makefile` to install all dependencies. In your terminal, simply run:

```bash
make install
```

This will automatically install the necessary packages listed in `requirements.txt`, including:

- flask
- torch
- torchvision
- open-clip-torch
- pandas
- numpy

---

## Part 1: Running the Application

Once the environment is set up, you can start the Flask application by running:

```bash
make run
```