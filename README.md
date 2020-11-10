# 03P_Facial_Expression_Recoginition

Facial Expression Recognition with Keras!

## FER Dataset
__Facial Expression Recognition Challenge__ [click here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

## :heavy_check_mark: Objectives

   :one: Develop a facial expression recognition model in Keras
   
   :two: Build and train a convolutional neural network (CNN)
   
   :three: Deploy the trained model to a web interface with Flask
   
   :four: Apply the model to real-time video streams and image data

<img src= 'https://miro.medium.com/max/1000/1*cQMgkngnYdIRHcZ2cJUnAg.jpeg' width = 900 height = 400>

Emotion recognition is the process of identifying human emotion, most typically from facial expressions as well as from verbal expressions. This is both something that humans do automatically but computational methodologies have also been developed.

Human expression recognition plays an important role in the interpersonal relationship. … Expression are reflected from speech, hand and gestures of the body and through facial expressions. Hence extracting and understanding of expression has a high importance of the interaction between human and machine communication.

we will build and train a convolutional neural network (CNN) in Keras from scratch to recognize facial expressions.

:red_circle: :one: __Introduction and Overview__

    - Introduction to the data and and overview of the project.
    - See a demo of the final product you will build by the end of this project.
    - Introduction to the Rhyme interface.
    - Import essential modules and helper functions from NumPy, Matplotlib, and Keras.
  
:red_circle: :two: __Explore the Dataset__

    - Display some images from every expression type in the Emotion FER dataset.
    - Check for class imbalance problems in the training data.
   
:red_circle: :three: __Generate Training and Validation Batches__

    - Generate batches of tensor image data with real-time data augmentation.
    - Specify paths to training and validation image directories and generates batches of augmented data.
   
:red_circle: :four: __Create a Convolutional Neural Network (CNN) Model__

    - Design a convolutional neural network with 4 convolution layers and 2 fully connected layers to predict 7 types of facial expressions.
    - Use Adam as the optimizer, categorical crossentropy as the loss function, and accuracy as the evaluation metric.
    
:red_circle: :five: __Train and Evaluate Model__

    - Train the CNN by invoking the `model.fit()` method.
    - Use `ModelCheckpoint()` to save the weights associated with the higher validation accuracy.
    - Observe live training loss and accuracy plots in Jupyter Notebook for Keras.

:red_circle: :six: __Represent Model as JSON String__

    - Sometimes, you are only interested in the architecture of the model, and you don't need to save the weight values or the optimizer.
    - Use to `json()`, which uses a JSON string, to store the model architecture.

:red_circle: :seven: __Create a Flask App to Serve Predictions__
     
     - Use open-source code from "[Video Streaming with Flask Example](https://github.com/log0/video_streaming_with_flask_example)" to create a flask app to serve the model's prediction images directly to a web interface.
     
:red_circle: :eight: __Design an HTML Template for the Flask App__

    - Create a FacialExpressionModel class to load the model from the JSON file, load the trained weights into the model, and predict facial expressions.
    
:red_circle: :nine: __Use Model to Recognize Facial Expressions in Videos__

    - Run the `main.py` script to create the Flask app and serve the model's predictions to a web interface.
    - Apply the model to saved videos on disk.
    
--- 

### Connect with me:


[<img align="left" alt="codeSTACKr | Twitter" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/twitter.svg" />][twitter]
[<img align="left" alt="codeSTACKr | LinkedIn" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/linkedin.svg" />][linkedin]
[<img align="left" alt="codeSTACKr.com" width="22px" src="https://raw.githubusercontent.com/iconic/open-iconic/master/svg/globe.svg" />][StackExchange AI]

[twitter]: https://twitter.com/F4izy
[linkedin]: https://www.linkedin.com/in/faizy-mohd-836573122/
[StackExchange AI]: https://ai.stackexchange.com/users/36737/cypher


---


![Faizy's github stats](https://github-readme-stats.vercel.app/api?username=mohd-faizy&show_icons=true)


[![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=mohd-faizy&layout=compact)](https://github.com/mohd-faizy/github-readme-stats)
