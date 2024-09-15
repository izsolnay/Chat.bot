#!/usr/bin/env python
# coding: utf-8

# # Basic Chatbot
# This basic chatbot was designed simply to get a feel for the basic programming of conversational chatbots and how the call and response method works.
# 
# All code at https://github.com/izsolnay

# In[1]:


#Import libraries
import nltk
import random
import string
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[2]:


nltk.download('punkt')  # if don't have it


# ## Step 1: Add training data

# In[3]:


# Some training data
training_sentences = [
    {
        "greeting": [
            "Hello",
            "Hi there",
            "Good morning",
            "Hey!",
            "What's up?"
        ],
        "goodbye": [
            "Goodbye",
            "See you later",
            "Take care",
            "Farewell",
            "Catch you later"
        ],
        "thanks": [
            "Thank you",
            "Thanks",
            "I appreciate it",
            "Thanks a lot",
            "Many thanks"
        ],
        "help_request": [
            "Can you help me?",
            "I need assistance",
            "Please help me",
            "I need some help",
            "Can you assist me?"
        ],
        "joke_request": [
            "Can you tell me a joke?",
            "I need a laugh, do you have any jokes?",
            "Make me laugh with a funny joke.",
            "Do you know any good jokes?"
        ],
        "small_talk": [
            "How's your day going?",
            "What have you been up to lately?",
            "Any fun plans for the weekend?",
            "What do you think about the current events?"
        ],
        "weather_inquiry": [
            "What's the weather like today?",
            "Can you tell me the forecast for this week?",
            "Is it going to rain tomorrow?",
            "What temperature is it outside right now?"
        ],
        "time_inquiry": [
            "What time is it?",
            "Can you tell me the current time?",
            "How many hours until noon?",
            "What time does the sun set today?"
        ],
        "location_inquiry": [
            "Where is the nearest coffee shop?",
            "Can you help me find a good restaurant around here?",
            "What’s the best way to get to the park?",
            "How far is it to the nearest bus station?"
        ],
        "advice_request": [
            "Can you give me some advice on studying?",
            "What should I do if I'm feeling stressed?",
            "Do you have any tips for improving my writing?",
            "How can I better manage my time?"
        ],
        "feedback": [
            "I’d like to give some feedback on your service.",
            "How can I submit my suggestions?",
            "What’s the best way to provide comments?",
            "I have some thoughts on how to improve this."
        ]
    }
]


# ## Step 2: Add corresponding intents

# In[4]:


# Corresponding intents
intents = [
    "greeting",
    "goodbye",
    "thanks",
    "help_request",
    "joke_request",
    "small_talk",
    "weather_inquiry",
    "time_inquiry",
    "location_inquiry",
    "advice_request",
    "feedback"
]


# #### Discussion of intents
# In the context of chatbots and natural language processing (NLP), "intents" refer to the purpose or goal behind a user's input. Each intent represents a specific action or request that a user might have when interacting with the chatbot. Understanding intents is crucial for the chatbot to respond appropriately and effectively.
# 
# Particular kinds of intents:
# * Purpose Identification: Intents help identify what the user wants to achieve with their message. For example, if a user says "What's the weather like today?" the intent is to inquire about the weather.
# 
# * Categorization: Intents are often categorized to help the chatbot understand and respond to various types of interactions. Common categories include greetings, farewells, questions, requests for information, and more.
# 
# * Training Data: Intents are usually associated with training phrases or examples that represent how users might express that intent. For example, for the "greeting" intent, training phrases might include "Hello," "Hi there," or "Good morning."
# 
# * Response Generation: Once the chatbot identifies the user's intent, it can generate an appropriate response based on the intent and any relevant context or information.
# 
# * Improvement Over Time: As users interact with the chatbot, the system can learn from new inputs and improve its understanding of intents, leading to better responses in the future.

# ## Step 3: Prepare data
# * **Flatten**
#   * Create loop which builds 2 lists
#     * flattened_sentences to contain all the sentences from all intents in a single list
#     * labels to contain corresponding intent for each sentence in flattened_sentences
# 
# 
# * **Vectorization**
#   * `CountVectorizer()` analyzes the text and identifies all unique words (tokens) present in the dataset. It creates a mapping of these words to feature indices.
# 
#   * `fit_transform method`
#     * **fit**_transform(): model learns vocabulary of the input text data (flattened_sentences).
#     * fit_**transform**(): after learning vocabulary, model transforms the input text into a numerical format, creating a sparse matrix where each row corresponds to a document (in this case, each sentence), and each column corresponds to a unique word from the vocabulary. The values in the matrix represent the count of each word in the respective document.
# 
#   * `vectorizer class` converts a collection of text documents into a matrix of token counts
# 

# In[5]:


# Flatten training sentences into a list of strings 
# Create corresponding label for each sentences in flattened_sentences
flattened_sentences = []
labels = []

# Create look that goes through each key-value pair in dictionary
# Key takes on value of intent and sentences associated with intent
for key, sentences in training_sentences[0].items():  # list contains 1 dictionary with 1st (& only) element at index 0
                                                      # items() display list of dics tuple pairs, iterate over each intents & associated sents.
    flattened_sentences.extend(sentences)  # extend() add all elements to flattened_sentences to flatten into 1 list
    
    labels.extend([key] * len(sentences))  # add the intent key as the label for each sentence
                                           # intent key is repeated as many times(*) as sentences in intent
                                           # extend() adds to flattened_sentences


# In[6]:


# Vectorization
vectorizer = CountVectorizer()  # convert data into matrix of tokens where each word becomes a feature and the count of each word form entries 
X_train = vectorizer.fit_transform(flattened_sentences)  # fit to vectorizer 

y_train = labels  # labels corresponding to training data


# ## Step 4: Build predicative Naive Bayes classifier
# The Naive Bayes classifier is trained on the intents and can then predict the intent of unseen messages by putting user input into the predefined categories to recognize. 
# 
# A Naive Bayes classifier is often used in *basic* chatbots for:
# 
# * Simplicity and Efficiency: Naive Bayes is easy to implement and computationally efficient, making it suitable for applications with limited resources. It can quickly classify input text based on learned probabilities.
# * Probabilistic Approach: It operates on the principle of Bayes' theorem, which allows it to predict the category of a given input based on the likelihood of the input belonging to each category. This is particularly useful in chatbots for understanding user intents.
# * Handling Large Datasets: Naive Bayes can effectively handle large datasets, which is beneficial when training chatbots on diverse conversational data.
# * Performance with Text Data: It works well with text classification tasks, such as spam detection or intent recognition, due to its capability to consider the frequency of words and phrases.
# * Independence Assumption: Although the assumption that features are independent may not hold true in real-world scenarios, Naive Bayes often performs surprisingly well in practice, making it a popular choice for initial implementations.

# In[7]:


# Train a Naive Bayes classifier
# Call classifer
model = MultinomialNB()

# Fit model
model.fit(X_train, y_train)


# ## Map Responses
# Response Mapping\
# Once the intent is identified, the chatbot looks up the associated responses for that intent 
# * define a list of possible responses for each intent. These must match
# 
# Response Generation\
# Chatbot then selects one of the predefined responses to send back to the user 
# * this can be randomized from the list of responses or a specific one based on context

# In[8]:


# Some responses based on intents
responses = {
    "greeting": [
        "Hello! How can I assist you today?",
        "Hi there! What can I do for you?",
        "Hey! How's it going?",
        "Greetings! What brings you here?",
        "Good to see you! How can I help?",
        "Welcome! What would you like to know?",
        "Hi! I'm here to assist you.",
        "Hello! What can I help you with today?"
    ],
    "goodbye": [
        "Goodbye! Have a great day!",
        "See you later!",
        "Take care!",
        "Farewell! Come back anytime.",
        "Goodbye! Wishing you all the best!",
        "See you next time!",
        "Until we meet again!",
        "Bye! Don't hesitate to return."
    ],
    "thanks": [
        "You're welcome!",
        "No problem!",
        "Glad to help!",
        "Anytime! I'm here for you.",
        "Happy to assist!",
        "You're very welcome!",
        "It was my pleasure!",
        "Don't mention it!"
    ],
    "help_request": [
        "Of course! What do you need help with?",
        "I'm here to help! Please tell me your question.",
        "Sure! What can I assist you with?",
        "Let me know how I can help you.",
        "I'd be happy to help! What do you need?",
        "Just ask, and I'll do my best to assist you!",
        "How can I support you today?",
        "What do you need assistance with?"
    ],
    "joke_request": [
        "Sure! Why don't scientists trust atoms? Because they make up everything!",
        "Here's a joke: Why did the scarecrow win an award? Because he was outstanding in his field!",
        "Want to hear a funny one? What do you call fake spaghetti? An impasta!",
        "Okay! Why did the bicycle fall over? Because it was two-tired!",
        "Here's one: What do you get when you cross a snowman and a vampire? Frostbite!",
        "Want another? Why did the math book look sad? Because it had too many problems!",
        "Sure! How does a penguin build its house? Igloos it together!",
        "Here's a classic: Why did the cookie go to the doctor? Because it felt crummy!"
    ],
    "small_talk": [
        "How's your day going?",
        "What have you been up to lately?",
        "Do you have any plans for the weekend?",
        "What’s your favorite hobby?",
        "What kind of music do you enjoy?",
        "Have you seen any good movies recently?",
        "What’s your favorite book?",
        "Do you like to travel? Where to?"
    ],
    "weather_inquiry": [
        "I can't check the weather, but you can use a weather app or website for that!",
        "It's best to check a reliable weather service for the latest updates.",
        "I recommend looking at a weather website for the most accurate forecast.",
        "You might want to check your local news for weather updates.",
        "For the latest weather, try asking a virtual assistant or checking a weather app."
    ],
    "time_inquiry": [
        "I can't tell the time, but you can check your device for the current time.",
        "For the exact time, please refer to your clock or smartphone.",
        "You can easily find out the time by looking at your watch or phone.",
        "I suggest checking a clock or your device for the time."
    ],
    "location_inquiry": [
        "I'm here in the digital realm, but where are you located?",
        "I exist online, but I can help you find information about any location!",
        "I don't have a physical location, but I can assist you with directions or local info.",
        "I’m all about the virtual world! How can I help you with locations?"
    ],
    "advice_request": [
        "Sure! What kind of advice are you looking for?",
        "I'm here to help! Please share your situation.",
        "I'd be happy to offer advice! What's on your mind?",
        "Let me know what you need advice about, and I'll do my best to help.",
        "What kind of guidance do you need? I'm here for you!"
    ],
    "feedback": [
        "I appreciate your feedback! What would you like to share?",
        "Your thoughts are valuable! Please let me know your feedback.",
        "I'm all ears! What feedback do you have for me?",
        "Thank you for your input!"
    ]
}


# In[9]:


# Function to process user input
def chatbot_response(user_input):    # define 1 parameter function
    
    user_input = user_input.lower()  # convert (user) input to lowercase
    
    user_input = user_input.translate(str.maketrans("", "", string.punctuation))  # remove punctuation to standardize 
    
    input_vector = vectorizer.transform([user_input])   # convert text into a numerical vector
    
    predicted_intent = model.predict(input_vector)[0]   # pass to model for predictions; 0 = 1st prediction 
    
    return random.choice(responses.get(predicted_intent, ["I'm sorry, I didn't understand that."])) # randomly select reponse from list
                                                         # "I'm sorry" is 2nd argument to default to 


# In[ ]:


# Create main loop
print("Chatbot: Hello! Type 'exit' to end the chat.")  # display welcome message

while True:                        # while loop = infinite display until loop is cut off
    user_input = input("You: ")    # string to capture
    
    if user_input.lower() == "exit":      # exit condition
        print("Chatbot: Goodbye!")
        
        break
        
    response = chatbot_response(user_input)
    
    print("Chatbot:", response)


# #### Discussion of response = chatbot_response(user_input)
# The chatbot is calling a function named `chatbot_response`, which is intended to generate a response based on the user's input. It could either repeat the user's input or provide a more meaningful and context-aware response - here it will not.
# 
# Here are a few possibilities for what this function could (otherwise) do:
# 
# * Echo Response: If the function is designed to simply return the user's input, then yes, it would repeat what the user typed. This is often used in very basic chatbots for testing purposes.
# * Predefined Responses: The function could be programmed to recognize certain keywords or phrases and return predefined responses. For example, if the user types "Hello," the function might return "Hi there! How can I help you today?"
# * Dynamic Responses: The function could use more complex logic or machine learning models to generate responses that are contextually relevant or conversationally appropriate, making the interaction feel more natural.
# * Sentiment Analysis: The function might analyze the sentiment of the user's input and respond accordingly. For instance, if the user expresses frustration, the chatbot might respond with empathy.
# * Fallback Response: If the input does not match any recognized patterns or keywords, the function might return a fallback response like "I'm not sure how to respond to that. Can you ask something else?"

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




