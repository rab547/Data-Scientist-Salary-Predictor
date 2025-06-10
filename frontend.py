import streamlit as st
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import joblib

model_5 = joblib.load("models/saved_models/search_05p.pkl")
model_50 = joblib.load("models/saved_models/search_50p.pkl")
model_95 = joblib.load("models/saved_models/search_95p.pkl")
expected_columns = joblib.load("models/saved_models/feature_columns.pkl")

df = pd.read_csv('Data/ds_salaries.csv')

st.title("Predicting Data Science Salary")
st.subheader("Group Name: Algo Warriors")
st.write("Team Members: Arnav, Aden, and Rahul")

# About Us Section
st.header("About Us")
col1, col2, col3 = st.columns(3)

with col1:
    # st.image("member1.jpg", width=150)
    st.write("**Arnav**")
    st.write("Hey, I'm Arnav and I'm a member of the Machine Learning Engineering subteam of Cornell Data Science, and I am a sophomore studying Computer Science!")

with col2:
    # st.image("member2.jpg", width=150)
    st.write("**Aden**")
    st.write("I'm Aden, I'm on the Machine Learning Engineering subteam of Cornell Data Science, and I'm a freshman studying Computer Science in the College of Arts and Sciences.")

with col3:
    # st.image("member3.jpg", width=150)
    st.write("**Rahul**")
    st.write("Hi! I'm Rahul and I am on the data science subteam of Cornell Data Science and am a freshman studying computer science in the college of engineering.")

# Project Introduction
st.header("Project Introduction")
st.write("""Salaries are confusing. Even though we haven't entered the job market yet, we've seen how complicated salary negotiation can be. 
        Salary information on job postings is often hidden or inconsistent, and they can vary a lot based on factors like location, experience, or company size.
        This uncertainty can make it hard for candidates to know what's fair, especially early in their careers. 
""")
st.write("""We wanted to explore a simple question: Can we use data to predict what someone should be paid for a data science job, based on features like experience, location, and job title?""")
st.write("""Our project will learn from a dataset contaning data scientist salaries alongside distinguishing information such as
        experience level, employment type, job_title, salary, salary_currency, etc,
        and our goal is to utilize it in order to predict the salaries of incoming data scientists in order to detirmine
        if they are being underpaid.""")

st.header("Data Manipulation")
st.write("""Our data cleansing process consisted of discovering all NAN values and replacing them with relavant summary statistics.
         In order to perform regression analysis, we converted all the categorical features using one hot encoding. We chose this method
         as it provided the most representative measure of our features compared to other encodings.""")

st.header("Data Visualization")
st.write("""Below are some key plots related to our dataset. The first shows the frequency of each salary in USD, and represents the main
         classification/regression target variable of our project. Secondly, we can see a scatterplot of the currency and salary in USD. 
         The main takeaway from this is that we can understand that linear regression is not the ideal approach for predicting salary, due to 
         the categorical nature of our dataset.""")
st.image("data/figures/Histogram of salary_in_usd.png")
# st.image("data/figures/lin_reg/curr_sal.png")

st.header("💼 Predict Your Data Science Salary")
work_year = st.selectbox("Work Year", df["work_year"].unique())
experience_level = st.selectbox("Experience Level", df["experience_level"].unique())
employment_type = st.selectbox("Employment Type", df["employment_type"].unique())
job_title = st.selectbox("Job Title", df['job_title'].unique())
employee_residence = st.selectbox("Employee Residence", df["employee_residence"].unique())
company_size = st.selectbox("Company Size", ["S", "M", "L"])
company_location = st.selectbox("Company Location", df["company_location"].unique())
remote_ratio = st.selectbox("Remote Work Level", df['remote_ratio'].unique())

# When button clicked:
# Warning: very ugly code; categorical features are not working properly
if st.button("Predict Salary"):
    # Construct a single-row DataFrame
    user_input = pd.DataFrame([{
        'work_year': work_year,
        'experience_level': experience_level,
        'employment_type': employment_type,
        'job_title': job_title,
        'employee_residence': employee_residence,
        'company_size': company_size,
        'company_location': company_location,
        'remote_ratio': remote_ratio
    }])

    ordinal_cols = ["company_size"]
    ordinal_order = [["S", "M", "L"]]
    user_input[ordinal_cols] = OrdinalEncoder(categories=ordinal_order).fit_transform(user_input[ordinal_cols])

    user_input_prepped = pd.get_dummies(user_input, columns=["experience_level", "employment_type", "job_title", "employee_residence", "company_location"], drop_first=True)
    user_input_prepped = user_input_prepped.reindex(columns=expected_columns, fill_value=0)

    # Predict
    median_salary = model_50.predict(user_input_prepped)
    lower = model_5.predict(user_input_prepped)
    upper = model_95.predict(user_input_prepped)

    # Display results
    st.success(f"Estimated salary: **${median_salary[0]:,.0f}**")
    st.caption(f"90% range: ${lower[0]:,.0f} - ${upper[0]:,.0f}")


st.header("Frontend")
st.write("""The framework we chose for our frontend is Streamlit. We found that Streamlit provides an efficient, elegant solution for us 
         to present our analysis and findings from this project. Streamlit is perfect for data science projects.""")

st.header("Continuous Integration")
st.write("""We are utilizing GitHub Actions as our main CI framework. Specifically, we are using it to ensure that each push to our repository 
         pases all baseline requirements. These include dataset verification, unit testing, and code formatting/linting.""")

st.header("KNN")
st.write("""KNN models simply classify points based on their nearest neighbors. When given a test point, the model determines the k closest points 
         by some measure of distance (usually Euclidean or Manhattan). It then assigns the majority class as the prediction for the given point.""")
st.write("""As k increases, the prediction becomes less dependent on the nearby points and simply converges to the class with the greatest number of datapoints in the test dataset.""")
st.write("""When the number of input features decreases, the feature space \"flattens\" making the predictions less specific to a given points features.""")
st.write("""Since all of our features are categorical, normalization and scaling significantly degrade the accuracy of our KNN model.""")
st.image("data/figures/knn_accuracy.png")
st.image("data/figures/knn_scale.png")

st.header("SVR")
st.write("""1. SVR's work by taking data vectors, possibly raising them to a higher dimension, and trying to find a hyperplane which best approximates the data.""")
st.write("""2. It is different from a linear regression due to kernels. Kernels allow the data points to be raised to higher dimensions and creates the possibility for a non-linear regression.""")
st.write("""3. C controls the tradeoff for a misclassified point and minimizing error, while gamma affects how strongly a single point will affect the boundary""")
st.image("data/figures/SVR.png")

st.header("Decision Tree Regressor")
st.write("""Decision trees try to predict the value of a target value by learning “rules” from the data features. It starts from a root question and continuously splits data by asking about certain features until it reaches a leaf node with a final prediction.""")
st.write("""Increasing the depth means the decision tree regressor can “ask more questions” about the data to help it create better predictions, which is reflected in how increasing max depth from 3 to 10 decreases the validation MSE. However, when we set max_depth to None, our decision tree overfits on our training data and actually performs worse on the validation data.""")
st.image("data/figures/DTR_1.png")
st.write("""Strangely, our decision tree regressor gets worse when we use more features. I would have thought that using more features would give the model more information to create predictions from, but I guess not. I am not sure why this is the case.""")
st.image("data/figures/DTR_2.png")
st.write("""With Decision Tree Regressors, normalizing the data has no effect on the accuracy of the model. This makes intuitive sense since as long as we train our decision tree on the normalized data, the rules it learns should be the same (only with the scaling changed). Thus, the model will make the same decisions.""")
st.image("data/figures/DTR_3.png")


st.header("Model Performance")
st.write("""As seen above, the model that performed the best was KNN. We believe it had the lowest validation loss due to the 
         heavily categorical nature of our feature set. Due to this, the \"nearby\" points shared similar feature classes,
         leading to similar prediction classes. It context, it logically shows that job candidates are likely to be paid the same as those with similar backgrounds and company features.""")
st.write("""Since our KNN model performed significantly well compared to the others, we expect to use this in almost all scenarios. 
         Since KNN has an extremely minimal training time, it is especially useful for scenarios that require a short initiation time. 
         However, KNNs have a very intensive test classification time so it may not be as useful for quick prediction requirements.""")

st.header("Project Takeaways")
st.write("Our project taught us a lot about different types of supervised learning models and how they compare with each other. For example, SVM vs. Decision Tree Regressor")
st.write("""Decision Trees are very easy to interpret, fast to train, and need minimal data preprocessing. However, they are also prone to overfitting.
         On the other hand, SVM have good performance in high-dimensional problems and are fairly accurate, but can have high training complexity and are often hard to interpet.""")
st.write("""Another key insight we realized is the importance of trying new things. We cast a wide net with the models we tested, and we because of that we were able to achieve pretty good results.
         We even played with the idea of returning a range of possible salaries instead of an exact amount, but the 90 percent onfidence intervals we were returning were too wide to be particularly useful.""")
st.image("data/figures/conf_interval.png")

st.header("Conclusion")
st.write("""Although our KNN model showed promising results, our lack of success with SVR and Decision Tree Regression suggest that there is a lot of room for improvement." \
        One change we could have made to our approach was predicting a salary range instead of an exact value. For example, instead of using 
        Decision Tree Regression, perhaps using Random Forests with Quantile Regression would lead to better results.""")

# Run this script with `streamlit run frontend.py`
