
# This project was built during the "Naive Bayes Algorithm" module of Udacity's Data Science Nanodegree.

# Imports.
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Get Data.
df = pd.read_table("sms.txt", sep='\t', names=["label", "sms_message"])

# Replace "ham" and "spam" values at the column label for numerical values.
df = df.replace({"label": {"ham":0, "spam":1}})

# The data is spllited into (x, y) training and (x, y) testing data.
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], df['label'], random_state=1)

# Format and transform the sentences into numerical values to better fit the naive bayes algorithm.
count_vector = CountVectorizer()
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)

# Trains the model.
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

# Tests the model using the testing data created previously.
predictions = naive_bayes.predict(testing_data)
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))
print('*************\n\n')

# Get users input and uses the model to classify it into ham or spam.
user_input = ''

# Interacts with the user for him to type a message and the algorithm predict if it's spam or ham.
while True:
    user_input = input("Enter an SMS text and we will tell you if it's spam or ham (or type quit to leave): ")
    if user_input.strip() == "quit": exit()  
    formatted_input = count_vector.transform([user_input])
    prediction = naive_bayes.predict(formatted_input)[0]

    if prediction == 0:
        print("\nMmmm that's ham!\n")
        print("*************\n\n")
    else:
        print("\nAwww that's spam!\n")
        print("*************\n\n")





