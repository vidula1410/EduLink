import streamlit as st
import pandas as pd 
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
class GCNModel(nn.Module):
    def _init_(self, num_node_features, hidden_channels, num_classes):
        super(GCNModel, self)._init_()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
@st.cache(allow_output_mutation=True)
def load_data():
    source_path1 = os.path.join("data/coursera-courses-overview.csv")
    source_path2 = os.path.join("data/coursera-individual-courses.csv")
    try:
        df_overview = pd.read_csv(source_path1, encoding='utf-8')
        df_individual = pd.read_csv(source_path2, encoding='utf-8')
    except UnicodeDecodeError:
        df_overview = pd.read_csv(source_path1, encoding='latin-1')
        df_individual = pd.read_csv(source_path2, encoding='latin-1')
    df = pd.concat([df_overview, df_individual], axis=1)
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    df['skills'] = df['skills'].fillna('Missing')
    df['instructors'] = df['instructors'].fillna('Missing')
    df['skills'] = df['skills'].apply(lambda x: x.split(','))
    df['description'] = df['description'].fillna('').apply(preprocess_text)
    return df
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(tokens)
def construct_multipartite_graph(df):
    G = nx.Graph()
    for i, row in df.iterrows():
        course = row['course_name']
        skills = row['skills']
        description = row['description']
        G.add_node(course, bipartite=0)
        for skill in skills:
            G.add_node(skill, bipartite=1)
            G.add_edge(course, skill)
        G.add_node(description, bipartite=2)
        G.add_edge(course, description)
    return G
def filter_courses_based_on_skills(df, chosen_skills):
    G = construct_multipartite_graph(df)
    recommended_courses = set()
    for skill in chosen_skills:
        if skill in G:
            recommended_courses.update(G.neighbors(skill))
    courses_with_all_skills = []
    for course in recommended_courses:
        skills_of_course = set(df[df['course_name'] == course]['skills'].iloc[0])
        if set(chosen_skills).issubset(skills_of_course):
            courses_with_all_skills.append(course)
    return courses_with_all_skills
def prep_for_cbr(df):
    st.header("Content-based Recommendation")
    st.sidebar.header("Filter on Preferences")
    st.write("Choose course from 'Select Course' dropdown on the sidebar")
    query = st.text_input("Enter your query", "")
    processed_query = preprocess_text(query)
    if 'skills' in df.columns:
        skills_avail = []
        for i in range(len(df)):
            try:
                skills_avail = skills_avail + df['skills'].iloc[i]
            except AttributeError:
                continue
        skills_avail = list(set(skills_avail))
        skills_select = st.sidebar.multiselect("Select Skills", skills_avail)
        input_course = st.sidebar.selectbox("Select Course", df['course_name'], key='courses')
        if st.sidebar.button("Filter Courses"):
            temp = filter_courses_based_on_skills(df, skills_select)
            skill_filtered = df[df['course_name'].isin(temp)].reset_index()
            courses = skill_filtered['course_name']
            st.write(skill_filtered)
            st.write("Number of programmes filtered:",skill_filtered.shape[0])
            st.write("Number of courses:",
                skill_filtered[skill_filtered['learning_product_type']=='COURSE'].shape[0])
            st.write("Number of professional degrees:",
                skill_filtered[skill_filtered['learning_product_type']=='PROFESSIONAL CERTIFICATE'].shape[0])
            st.write("Number of specializations:",
                skill_filtered[skill_filtered['learning_product_type']=='SPECIALIZATION'].shape[0])
    else:
        st.write("No 'skills' column found in the DataFrame.")
def train_gcn(df):
    G = construct_multipartite_graph(df)
    edge_index = []
    for edge in G.edges:
        edge_index.append(edge)
    edge_index = torch.tensor(edge_index).t().contiguous()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['description'].fillna(''))
    description_features = tfidf_matrix.toarray()
    num_courses = len(df['course_name'])
    num_skills = len(df['skills'].explode().unique())
    num_descriptions = len(description_features)
    x = torch.eye(num_courses + num_skills + num_descriptions)
    data = Data(x=x, edge_index=edge_index)
    model = GCNModel(num_node_features=x.size(1), hidden_channels=16, num_classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[:num_courses], torch.zeros(num_courses))
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
    with torch.no_grad():
        embeddings = model(data).numpy()
    return model, embeddings
def main():
    st.title("MLGA")
    st.write("Exploring Courses on Coursera")
    st.sidebar.title("Set your Parameters")
    st.sidebar.header("Preliminary Inspection")
    df = load_data()
    st.header("Dataset Used")
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
    prep_for_cbr(X_train)
    model, embeddings = train_gcn(X_train)
if __name__=="__main__":
    main()
